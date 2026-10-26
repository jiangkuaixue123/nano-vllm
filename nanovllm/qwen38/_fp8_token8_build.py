"""Model-owned SM90a build and exact per-function admission; stdlib only."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

SOURCE=Path(__file__).with_name('csrc')/'fp8_token8'
"""Five fixed geometry cases; no parameter search or runtime shape selection."""
CASES = {
    'gate_up': {'id':0,'N':34816,'K':5120,'family':'mlp_gate_up','first':0,'last':63},
    'gdn_input': {'id':1,'N':16384,'K':5120,'family':'gdn_input_qkvz','first':0,'last':62},
    'fa_input': {'id':2,'N':14336,'K':5120,'family':'attention_input_qkv','first':3,'last':63},
    'gdn_output': {'id':3,'N':5120,'K':6144,'family':'gdn_output','first':0,'last':62},
    'down': {'id':4,'N':5120,'K':17408,'family':'mlp_down','first':0,'last':63},
}
def dynamic_bytes(case,token_tile):
    k=CASES[case]['K']
    return 128*token_tile*2+8*(128*128+token_tile*128+128)+((k//128*4+7)//8)*8+128

DEVICE_FLAGS = (
    "-std=c++17", "--gpu-architecture=compute_90a",
    "--gpu-code=sm_90a,compute_90a",
    "--ptxas-options=-allow-expensive-optimizations=true",
    "--ptxas-options=--register-usage-level=10", "--diag-suppress=161,174,177,940",
    "-D__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__=1",
    "-D__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__=1", "-O3",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi",
)


RESOURCE_KEYS = ('registers', 'static_shared_bytes', 'local_bytes', 'max_threads',
                 'max_dynamic_shared_bytes', 'requested_dynamic_bytes',
                 'device_optin_shared_bytes', 'device_registers_per_block', 'sms', 'binary_version')

# Every case/arm binds one exact full-kernel function. No shape aliasing.
def kernel_symbol(case, token_tile):
    n,k=CASES[case]['N'],CASES[case]['K']
    return ('_ZN24probe_fp8_token_n8_batch22fp8_gemm_kernel_swapAB'
            f'ILj{n}ELj{k}ELj128ELj{token_tile}ELj128ELj1ELj8ELj128ELj128ELj1E'
            f'NS_21NormalSchedulerSwapABILj{n}ELj128ELj{token_tile}ELj1ELj1ELj{n//128}ELj16EEE'
            'NS_26NormalSchedulerInputSwapABEEEvP13__nv_bfloat16PfT10_14CUtensorMap_stS8_S8_S8_')
KERNEL_SYMBOLS = {case+':'+arm:kernel_symbol(case,width) for case in CASES
                  for arm,width in (('control128',16),('token8',8))}


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def check_resources(rows):
    require(set(rows) == set(KERNEL_SYMBOLS), 'ten fixed compiled resource records')
    for name, row in rows.items():
        case,arm=name.split(':')
        require(row['requested_dynamic_bytes'] == dynamic_bytes(case,8 if arm=='token8' else 16)
                and row['sms'] == 78 and row['binary_version'] == 90
                and row['registers'] > 0 and row['registers'] * 384 <= row['device_registers_per_block']
                and row['static_shared_bytes'] + row['requested_dynamic_bytes'] <= row['device_optin_shared_bytes']
                and row['local_bytes'] == 0 and row['max_threads'] >= 384,
                'actual static/dynamic/register/local resource admission')


def check_compiled_resources(rows, compiled):
    """Keep API and binary quantities distinct; conservatively charge both."""
    check_resources(rows)
    require(set(compiled) == set(KERNEL_SYMBOLS), 'ten named compiled resource records')
    accounting = {}
    for label, row in rows.items():
        raw = compiled[label]
        require(set(raw) == {'symbol', 'registers', 'shared_bytes', 'stack_bytes', 'local_bytes'}
                and raw['symbol'] == KERNEL_SYMBOLS[label]
                and all(type(raw[k]) is int and raw[k] >= 0
                        for k in ('registers', 'shared_bytes', 'stack_bytes', 'local_bytes'))
                and type(row['static_shared_bytes']) is int and row['static_shared_bytes'] >= 0
                and raw['registers'] == row['registers']
                and raw['stack_bytes'] == raw['local_bytes'] == 0
                and row['max_dynamic_shared_bytes'] >= row['requested_dynamic_bytes'],
                'bound compiled register/local/stack and dynamic attribute evidence')
        total = row['static_shared_bytes'] + raw['shared_bytes'] + row['requested_dynamic_bytes']
        require(total <= row['device_optin_shared_bytes'], 'conservative combined shared-memory capacity')
        accounting[label] = dict(cuda_static_bytes=row['static_shared_bytes'],
                                 cuobjdump_shared_bytes=raw['shared_bytes'],
                                 requested_dynamic_bytes=row['requested_dynamic_bytes'],
                                 conservative_total_shared_bytes=total,
                                 device_optin_shared_bytes=row['device_optin_shared_bytes'])
    return accounting


def compiled_evidence(rows, resource_text, ptx, sass):
    """Parse retained evidence for the ten exact functions; no CUDA execution."""
    functions = re.findall(r' Function ([^\n]+):\n\s*(REG:[^\n]+)', resource_text)
    require(len(functions) == 10 and {name for name, _ in functions} == set(KERNEL_SYMBOLS.values()),
            'exact ten fixed compiled resource symbols')
    compiled = {}
    for name, text in functions:
        label = next(k for k, v in KERNEL_SYMBOLS.items() if v == name)
        pairs = re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)', text)
        values = {key: int(value) for key, value in pairs}
        require(len(pairs) == 4 and set(values) == {'REG', 'STACK', 'SHARED', 'LOCAL'},
                'complete unique compiled resource fields')
        compiled[label] = dict(symbol=name, registers=values['REG'], shared_bytes=values['SHARED'],
                               stack_bytes=values['STACK'], local_bytes=values['LOCAL'])
    accounting = check_compiled_resources(rows, compiled)
    ptx_parts = re.split(r'(?m)^\.visible \.entry ([^\s(]+)\(', ptx)
    sass_parts = re.split(r'(?m)^\s*Function : (\S+)\s*$', sass)
    require(len(ptx_parts) == len(sass_parts) == 21
            and set(ptx_parts[1::2]) == set(sass_parts[1::2]) == set(KERNEL_SYMBOLS.values())
            and re.findall(r'(?m)^\.target (\S+)', ptx) == ['sm_90a'],
            'same exact PTX/SASS functions and architecture-specific PTX')
    ptx_bodies = dict(zip(ptx_parts[1::2], ptx_parts[2::2]))
    sass_bodies = dict(zip(sass_parts[1::2], sass_parts[2::2]))
    for label, symbol in KERNEL_SYMBOLS.items():
        width = 8 if label.endswith(':token8') else 16
        ptx_ops = re.findall(r'\bwgmma\.mma_async\.[A-Za-z0-9_.]+', ptx_bodies[symbol])
        sass_ops = re.findall(r'\b(?:[A-Z]*GMMA|WGMMA)(?:\.[A-Za-z0-9_]+)+', sass_bodies[symbol])
        require(ptx_ops and set(ptx_ops) == {f'wgmma.mma_async.sync.aligned.m64n{width}k32.f32.e4m3.e4m3'}
                and sass_ops and set(sass_ops) == {f'QGMMA.64x{width}x32.F32.E4M3.E4M3'}
                and not re.search(r'\b(?:LDL|STL)(?:\.|\b)', sass_bodies[symbol]),
                'each bound function has selected FP8 E4M3 WGMMA shape and no local spill instructions')
        stores = re.findall(r'\bstmatrix\.sync\.[A-Za-z0-9_.]+', ptx_bodies[symbol])
        vector = 2 if label.endswith(':token8') else 4
        require(stores and set(stores) == {f'stmatrix.sync.aligned.x{vector}.m8n8.shared.b16.trans'},
                'selected full-kernel transposed shared output store')
    return compiled, accounting


def file_hash(path):
    with Path(path).open('rb') as stream:
        digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    expected=json.loads((SOURCE/'source-manifest.json').read_text())
    actual={str(p.relative_to(SOURCE)):file_hash(p) for p in sorted(SOURCE.rglob('*'))
            if p.is_file() and p.name!='source-manifest.json'}
    require(actual==expected,'FP8 token8 packaged source identity changed')
    host=Path(__file__)
    actual['python/'+host.name]=file_hash(host)
    actual['python/_fp8_token8.py']=file_hash(host.with_name('_fp8_token8.py'))
    return hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def build():
    """One eager build per model; owned files stay alive with its library handle.

    No installed source lookup, native-library prerequisite, cache reuse or
    diagnostic log dependency. The enclosing process owns compilation lifetime.
    """
    for name in ('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS','CPATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH'):
        require(not os.environ.get(name),'FP8 token8 rejects ambient compiler injection: '+name)
    source_key=source_identity()
    cuda=Path(os.environ.get('CUDA_HOME','/usr/local/cuda')).resolve(strict=True)
    host=shutil.which(os.environ.get('CXX','c++'))
    require(host is not None,'FP8 token8 requires a host C++ compiler')
    tools=[cuda/'bin'/name for name in ('nvcc','ptxas','fatbinary','nvlink','cudafe++','cuobjdump')]
    tools += [cuda/'nvvm/bin/cicc',Path(host)] + sorted((cuda/'nvvm/libdevice').glob('*.bc'))
    require(any(p.suffix=='.bc' for p in tools),'FP8 token8 requires CUDA libdevice')
    tool_ids={str(p.resolve(strict=True)):file_hash(p) for p in tools}
    cache=Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'nanovllm'/'fp8_token8'
    cache.mkdir(parents=True,exist_ok=True)
    work=tempfile.TemporaryDirectory(prefix=source_key[:16]+'-',dir=cache)
    root=Path(work.name);copied=root/'source';shutil.copytree(SOURCE,copied)
    require(source_identity()==source_key and
            {str(p.relative_to(SOURCE)):file_hash(p) for p in SOURCE.rglob('*') if p.is_file()} ==
            {str(p.relative_to(copied)):file_hash(p) for p in copied.rglob('*') if p.is_file()},
            'FP8 token8 source changed during preparation')
    command=[str(cuda/'bin/nvcc'),str(copied/'launch.cu'),*DEVICE_FLAGS,'--compiler-bindir',host]
    dep=root/'dependencies.make';library=root/'kernel.so'
    environment={**os.environ,'TMPDIR':str(root)}
    try:
        subprocess.run(command+['-M','-o',str(dep)],cwd=root,env=environment,check=True,capture_output=True)
        text=dep.read_text().replace('\\\n',' ');_,separator,paths=text.partition(':')
        require(bool(separator),'FP8 token8 compiler dependency output')
        dependencies={str(Path(p).resolve(strict=True)):file_hash(p) for p in shlex.split(paths)}
        require(bool(dependencies),'FP8 token8 empty dependency closure')
        subprocess.run(command+['-shared','--cudart=shared','-Xcompiler=-fvisibility=hidden','-o',str(library)],
                       cwd=root,env=environment,check=True,capture_output=True)
        require({p:file_hash(p) for p in dependencies}==dependencies
                and {p:file_hash(p) for p in tool_ids}==tool_ids and source_identity()==source_key,
                'FP8 token8 source/compiler dependency drift')
        return work,library,cuda/'bin/cuobjdump',dict(source_sha256=source_key,binary_sha256=file_hash(library),
            tool_sha256={Path(p).name:h for p,h in tool_ids.items()},dependency_count=len(dependencies),
            dependency_sha256=hashlib.sha256(json.dumps(dependencies,sort_keys=True).encode()).hexdigest(),
            disk_cache_reused=False)
    except BaseException:
        work.cleanup()
        raise


class Binary:
    """Pin both CDLL and its owned source/binary directory for graph lifetime."""
    def __init__(self,torch):
        self.device=torch.cuda.current_device()
        self.work,self.path,tool,self.identity=build()
        self.library=ctypes.CDLL(str(self.path))
        require(file_hash(self.path)==self.identity['binary_sha256'],'FP8 token8 loaded binary drift')
        self.library.token_n8_resources.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_ulonglong),ctypes.c_char_p,ctypes.c_size_t]
        self.library.token_n8_prepare.argtypes=[ctypes.c_char_p,ctypes.c_size_t]
        self.library.token_n8_launch.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*6+[ctypes.c_char_p,ctypes.c_size_t]
        for name in ('token_n8_resources','token_n8_prepare','token_n8_launch'):getattr(self.library,name).restype=ctypes.c_int
        before=self.resources();check_resources(before);self.call('token_n8_prepare')
        self.resource_rows=self.resources();check_resources(self.resource_rows)
        texts=[]
        for flag in ('--dump-resource-usage','--dump-ptx','--dump-sass'):
            result=subprocess.run([str(tool),flag,str(self.path)],check=True,capture_output=True,text=True,timeout=30)
            texts.append(result.stdout)
        self.compiled_resources,self.resource_accounting=compiled_evidence(self.resource_rows,*texts)
        require(file_hash(tool)==self.identity['tool_sha256']['cuobjdump']
                and file_hash(self.path)==self.identity['binary_sha256'],'FP8 token8 admitted binary/tool drift')
        self.identity.update(device=self.device,resource_admitted=True,
            compiler_evidence_sha256=[hashlib.sha256(t.encode()).hexdigest() for t in texts])

    def call(self,name,*args):
        error=ctypes.create_string_buffer(2048)
        status=getattr(self.library,name)(*args,error,len(error))
        require(status==0,error.value.decode(errors='replace'))

    def resources(self):
        rows={}
        for case,spec in CASES.items():
            for selected,arm in enumerate(('control128','token8')):
                values=(ctypes.c_ulonglong*len(RESOURCE_KEYS))()
                self.call('token_n8_resources',spec['id'],selected,values)
                rows[case+':'+arm]=dict(zip(RESOURCE_KEYS,map(int,values)))
        return rows
