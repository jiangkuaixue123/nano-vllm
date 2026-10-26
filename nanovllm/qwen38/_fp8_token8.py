"""Opt-in prepared FP8 consumers; selection is fixed, decode eligibility is per call."""
from ._fp8_token8_build import Binary, CASES, POOL80_CASES, policy_symbols, require

OWNER_CASE = {
    'linear_attn.in_proj_qkvz': 'gdn_input', 'self_attn.qkv_proj': 'fa_input',
    'mlp.gate_up_proj': 'gate_up', 'mlp.down_proj': 'down',
    'linear_attn.out_proj': 'gdn_output',
}
ROUTES = {'input_rms': {'gdn_input', 'fa_input'}, 'postnorm': {'gate_up'},
          'silu': {'down'}, 'gdn_outnorm': {'gdn_output'}}
COUNTS = {'gate_up': 64, 'gdn_input': 48, 'fa_input': 16, 'gdn_output': 48, 'down': 64}


def route_identity(model):
    """Validate the four prepared producers before stateful execution."""
    layers=model.model.layers
    require(len(layers)==64, 'FP8 token8 decoder layer coverage')
    routes=[]
    for index,layer in enumerate(layers):
        selected=(layer.input_rms_fp8_backend,layer.mlp.postnorm_fp8_backend,
                  layer.mlp.silu_fp8_backend,layer.mlp.activation_backend)
        require(selected==('triton',)*4, 'FP8 token8 requires prepared input/postnorm/SwiGLU producers and activation')
        if index%4!=3:
            gdn=layer.linear_attn
            selected+=(gdn.outnorm_fp8_backend,gdn.norm.outnorm_backend)
            require(selected[-2:]==('triton','triton'), 'FP8 token8 requires prepared GDN output producer')
        routes.append((id(layer),selected))
    return tuple(routes)


def tensor_key(tensor):
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device, tensor._version)


def validate_weights(torch, projection, case, device):
    spec=CASES[case];n,k=spec['N'],spec['K']
    require(projection.in_features==k and projection.out_features==n and projection.use_flashinfer,
            'FP8 token8 fixed projection backend/geometry')
    for tensor,shape,dtype in ((projection.weight,(n,k),torch.float8_e4m3fn),
                              (projection.weight_scale_inv,(n//128,k//128),torch.float32)):
        require(tensor.shape==shape and tensor.dtype==dtype and tensor.device==device
                and tensor.is_contiguous() and not tensor.requires_grad and tensor.data_ptr()%16==0,
                'FP8 token8 original checkpoint ABI')
    require(device.type=='cuda' and not getattr(torch.version,'hip',None)
            and torch.cuda.get_device_capability(device)==(9,0),'FP8 token8 SM90 CUDA')


def validate_input(torch,case,q,scales,device):
    k=CASES[case]['K']
    require(q.shape==(1,k) and q.dtype==torch.float8_e4m3fn and q.device==device
            and q.is_contiguous() and not q.requires_grad and q.data_ptr()%16==0,
            'FP8 token8 M1 activation ABI')
    require(scales.shape==(1,k//128) and scales.dtype==torch.float32 and scales.device==device
            and scales.stride()==(1,4) and scales.storage_offset()==0 and not scales.requires_grad
            and scales.data_ptr()%16==0 and scales.untyped_storage().nbytes()>=k//128*16,
            'FP8 token8 original padded live scale ABI')


class Consumer:
    def __init__(self,torch,projection,case,binary):
        self.torch,self.projection,self.case,self.binary=torch,projection,case,binary
        require(case in binary.cases, 'FP8 token8 policy/owner geometry')
        self.device=projection.weight.device
        validate_weights(torch,projection,case,self.device)
        self.weights=(tensor_key(projection.weight),tensor_key(projection.weight_scale_inv))
        self.identity=(binary.identity['source_sha256'],binary.identity['binary_sha256'],binary.device,case,
                       binary.policy,binary.launch_name,self.weights)
        self.ready=None

    def check(self,device,*,capturing):
        p=self.projection
        require(p.fp8_decode_consumer=='token8' and device==self.device
                and self.torch.cuda.current_device()==self.binary.device,'FP8 token8 fixed selection/device')
        require(self.binary.identity['resource_admitted'] and self.binary.identity['policy']==self.binary.policy
            and self.case in self.binary.cases and self.identity==(
            self.binary.identity['source_sha256'],self.binary.identity['binary_sha256'],self.binary.device,self.case,
            self.binary.policy,self.binary.launch_name,
            (tensor_key(p.weight),tensor_key(p.weight_scale_inv))), 'FP8 token8 prepared binary/geometry/weight identity')
        if capturing:require(self.ready==self.identity,'Warm the selected token8 consumer eagerly before CUDA Graph capture')

    def __call__(self,values,weight,*,input_scale,weight_scale,out_dtype):
        torch=self.torch;p=self.projection
        require(p._token8_active and weight is p.weight and weight_scale is p.weight_scale_inv
                and out_dtype==torch.bfloat16,'FP8 token8 bound prequantized owner only')
        self.check(values.device,capturing=torch.cuda.is_current_stream_capturing())
        validate_input(torch,self.case,values,input_scale,self.device)
        output=torch.empty((1,CASES[self.case]['N']),device=self.device,dtype=torch.bfloat16)
        self.binary.call(self.binary.launch_name,CASES[self.case]['id'],1,values.data_ptr(),input_scale.data_ptr(),
            weight.data_ptr(),weight_scale.data_ptr(),output.data_ptr(),torch.cuda.current_stream(self.device).cuda_stream)
        self.ready=self.identity
        return output


def consumer_for(projection,device,route):
    binding=getattr(projection,'_token8_binding',None)
    require(binding is not None and binding.case in ROUTES[route],'FP8 token8 owner/entry was not prepared')
    binding.check(device,capturing=projection._token8_active and binding.torch.cuda.is_current_stream_capturing())
    return binding


class ModelConsumers:
    def __init__(self,torch,model,owners,binary,pool80=None):
        self.torch,self.model_id,self.owners,self.binary=torch,id(model),tuple(owners),binary
        self.pool80=pool80
        self.routes=route_identity(model)

    def begin_forward(self,model,state,input_ids):
        require(id(model)==self.model_id,'FP8 token8 model ownership')
        active=(input_ids.shape[1]==1 and not (state.length==0 and state.positions is None))
        capturing=self.torch.cuda.is_current_stream_capturing()
        require(route_identity(model)==self.routes, 'FP8 token8 prepared producer routes changed')
        # Validate every selected owner before any layer KV/conv/recurrent write.
        for _,p in self.owners:
            require(p.fp8_decode_consumer=='token8','FP8 token8 selection changed after loading')
            if active:p._token8_binding.check(input_ids.device,capturing=capturing)
        for _,p in self.owners:p._token8_active=active

    def snapshot(self):
        return dict(backend='token8',prepared=True,owners=len(self.owners),case_counts=dict(COUNTS),
            policy='pool80' if self.pool80 is not None else 'original',
            case_policies={case:'pool80' if self.pool80 is not None and case in POOL80_CASES else 'original' for case in CASES},
            identity=dict(self.binary.identity),resources=self.binary.resource_rows,
            compiled_resources=self.binary.compiled_resources,resource_accounting=self.binary.resource_accounting,
            pool80=None if self.pool80 is None else dict(identity=dict(self.pool80.identity),
                resources=self.pool80.resource_rows,compiled_resources=self.pool80.compiled_resources,
                resource_accounting=self.pool80.resource_accounting,
                owners=sum(p._token8_binding.binary is self.pool80 for _,p in self.owners)),
            ready_owners=sum(p._token8_binding.ready is not None for _,p in self.owners),
            host_calls={case:sum(p.kernel_calls['token8'] for _,p in self.owners if p._token8_binding.case==case) for case in CASES})


def prepare_model(model,*,policy='original'):
    """Called once after original weights load, before any model forward/capture."""
    import torch
    from .quant import FP8Linear
    policy_symbols(policy)
    require(not torch.cuda.is_current_stream_capturing(),'Prepare token8 eagerly before model execution')
    require(not hasattr(model,'_fp8_token8_runtime'),'FP8 token8 model already prepared')
    route_identity(model)
    owners=[];counts={case:0 for case in CASES};excluded=0
    for name,p in model.named_modules():
        if not isinstance(p,FP8Linear):continue
        require(p.fp8_decode_consumer=='token8','Mixed immutable FP8 decode consumer selections')
        parts=name.split('.',3)
        require(len(parts)==4 and parts[:2]==['model','layers'],'FP8 token8 requires decoder projection ownership')
        layer=int(parts[2]);suffix=parts[3]
        if suffix=='self_attn.o_proj':
            require(0<=layer<64 and layer%4==3 and p.in_features==6144 and p.out_features==5120,'native FA output ownership')
            excluded+=1;continue
        require(suffix in OWNER_CASE and 0<=layer<64,'FP8 token8 unsupported projection owner')
        case=OWNER_CASE[suffix]
        require((case!='fa_input' or layer%4==3) and (case not in ('gdn_input','gdn_output') or layer%4!=3),
                'FP8 token8 layer/family ownership')
        validate_weights(torch,p,case,p.weight.device)
        owners.append((name,p));counts[case]+=1
    require(counts==COUNTS and excluded==16 and len({n for n,_ in owners})==240,'FP8 token8 exact240/16 projection coverage')
    require(all(p.weight.device==owners[0][1].weight.device for _,p in owners), 'FP8 token8 one-device model')
    with torch.cuda.device(owners[0][1].weight.device):
        binary=Binary(torch)
        pool80=Binary(torch,policy='pool80') if policy=='pool80' else None
        for name,p in owners:
            case=OWNER_CASE[name.split('.',3)[3]]
            selected=pool80 if pool80 is not None and case in POOL80_CASES else binary
            p._token8_binding=Consumer(torch,p,case,selected)
            p._token8_active=False
    runtime=ModelConsumers(torch,model,owners,binary,pool80)
    model._fp8_token8_runtime=runtime
    return runtime.snapshot()
