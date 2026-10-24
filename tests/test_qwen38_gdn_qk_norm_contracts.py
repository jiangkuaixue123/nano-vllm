"""Metadata and actual-source AST controls; no Torch/Triton or numeric oracle."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / 'nanovllm/qwen38'


def source(name):
    return ast.parse((QWEN / name).read_text())


def execute(nodes, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(tree, '<actual-source-ast>', 'exec'), namespace)  # noqa: S102
    return namespace


class Tensor:
    def __init__(self, shape=(1, 33, 16, 128), strides=None, dtype='bf16', device='cuda'):
        self.shape, self.ndim, self.dtype = shape, len(shape), dtype
        self.device, self.requires_grad = SimpleNamespace(type=device), False
        if strides is None:
            sizes = [1]
            for size in reversed(shape[1:]):
                sizes.insert(0, sizes[0] * size)
            strides = tuple(sizes)
        self.strides = strides

    def stride(self):
        return self.strides

    def __getattr__(self, name):
        raise AssertionError('Unexpected tensor-data or metadata access: ' + name)


@pytest.fixture
def dispatch():
    torch = SimpleNamespace(bfloat16='bf16', version=SimpleNamespace(hip=None),
                            cuda=SimpleNamespace(get_device_capability=lambda device: (9, 0)))
    return execute([n for n in source('_gdn_qk_norm.py').body if isinstance(n, ast.FunctionDef)],
                   {'torch': torch})


@pytest.mark.parametrize('tokens', [1, 15, 16, 17, 33, 512, 8192])
@pytest.mark.parametrize('layout', ['canonical', 'token_minor', 'padded'])
def test_pair_metadata_is_read_only(dispatch, tokens, layout):
    q = Tensor((1, tokens, 16, 128))
    if layout == 'token_minor':
        q.strides = (tokens, 1, 128 * tokens, tokens)
    elif layout == 'padded':
        q.strides = (1, 2, 128 * (2 * tokens + 3), 2 * tokens + 3)
    k = Tensor(q.shape)
    before = copy.deepcopy((vars(q), vars(k)))
    result = dispatch['execution'](q, k, 'triton')
    assert result['selected_backend'] == 'triton'
    assert result['cross_backend_comparison'] == 'diagnostic_reduction_order_differs'
    assert result['executed'] is False and (vars(q), vars(k)) == before


@pytest.mark.parametrize('change,reason', [
    ('cpu', 'requires_nvidia_cuda'), ('hip', 'requires_nvidia_cuda'),
    ('device', 'device_mismatch'), ('architecture', 'requires_sm90'),
    ('q_dtype', 'requires_bf16_inputs'), ('k_dtype', 'requires_bf16_inputs'),
    ('rank', 'unsupported_grouped_prefill_shape'), ('batch', 'unsupported_grouped_prefill_shape'),
    ('heads', 'unsupported_grouped_prefill_shape'), ('width', 'unsupported_grouped_prefill_shape'),
    ('empty', 'unsupported_grouped_prefill_shape'), ('long', 'unsupported_grouped_prefill_shape'),
    ('k_shape', 'unsupported_grouped_prefill_shape'),
    ('q_grad', 'requires_inference_operands'), ('k_grad', 'requires_inference_operands'),
    ('q_overlap', 'unsupported_read_layout'), ('k_overlap', 'unsupported_read_layout'),
    ('span', 'unsupported_read_layout'),
])
def test_explicit_fallback_without_tensor_reads_or_imports(dispatch, change, reason):
    q, k = Tensor(), Tensor()
    if change == 'cpu':
        q.device.type = 'cpu'
    elif change == 'hip':
        dispatch['torch'].version.hip = 'test'
    elif change == 'device':
        k.device.type = 'cpu'
    elif change == 'architecture':
        dispatch['torch'].cuda.get_device_capability = lambda device: (8, 0)
    elif change in ('q_dtype', 'k_dtype'):
        (q if change == 'q_dtype' else k).dtype = 'f32'
    elif change in ('rank', 'batch', 'heads', 'width', 'empty', 'long'):
        shapes = {'rank': (33, 16, 128), 'batch': (2, 33, 16, 128),
                  'heads': (1, 33, 48, 128), 'width': (1, 33, 16, 64),
                  'empty': (1, 0, 16, 128), 'long': (1, 8193, 16, 128)}
        q = Tensor(shapes[change])
    elif change == 'k_shape':
        k = Tensor((1, 34, 16, 128))
    elif change in ('q_grad', 'k_grad'):
        (q if change == 'q_grad' else k).requires_grad = True
    elif change in ('q_overlap', 'k_overlap'):
        (q if change == 'q_overlap' else k).strides = (1, 0, 128, 1)
    elif change == 'span':
        q.strides = (1, 2**31, 128, 1)
    result = dispatch['execution'](q, k, 'triton')
    assert result['selected_backend'] == 'native' and result['fallback_reason'] == reason
    assert not result['executed']


def test_native_default_and_typo(dispatch):
    q, k = Tensor(), Tensor()
    assert dispatch['execution'](q, k, 'native')['fallback_reason'] == 'configured_native'
    with pytest.raises(ValueError):
        dispatch['execution'](q, k, 'typo')


@pytest.mark.parametrize('requested', ['native', 'triton', 'typo'])
@pytest.mark.parametrize('gdn,layout', [('native', 'expanded'), ('flashinfer', 'expanded'),
                                      ('flashinfer', 'grouped')])
def test_actual_constructor_selection(requested, gdn, layout):
    delta = next(n for n in source('model.py').body if isinstance(n, ast.ClassDef) and n.name == 'DeltaNet')
    init = next(n for n in delta.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    nodes = [n for n in init.body if 'gdn_qk_norm_backend' in ast.unparse(n)]
    reads = []
    env = SimpleNamespace(get=lambda name, default: reads.append((name, default)) or requested)
    self = SimpleNamespace(gdn_backend=gdn, gdn_prefill_qk_layout=layout)
    valid = requested == 'native' or (requested == 'triton' and (gdn, layout) == ('flashinfer', 'grouped'))
    if valid:
        execute(nodes, {'self': self, 'os': SimpleNamespace(environ=env)})
        assert self.gdn_qk_norm_backend == requested
    else:
        with pytest.raises(ValueError):
            execute(nodes, {'self': self, 'os': SimpleNamespace(environ=env)})
    assert reads == [('NANOVLLM_GDN_QK_NORM', 'native')]


@pytest.mark.parametrize('backends', [('native',), ('triton',), ('native', 'triton')])
def test_actual_loader_report_and_prewrite_rejection(backends):
    load = next(n for n in source('loader.py').body if isinstance(n, ast.FunctionDef) and n.name == 'load_weights')
    nodes = [n for n in load.body if ast.unparse(n).startswith(('gdn_qk_norms =', 'if len(gdn_qk_norms)'))]
    report = next(n for n in ast.walk(load) if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == "report['gdn_qk_normalization']")
    ns = {'gdn_modules': [SimpleNamespace(gdn_qk_norm_backend=b) for b in backends], 'report': {}}
    if len(set(backends)) > 1:
        with pytest.raises(ValueError):
            execute(nodes, ns)
    else:
        execute([*nodes, report], ns)
        result = ns['report']['gdn_qk_normalization']
        assert result['requested_backend'] == backends[0] and result['execution_observed'] is False
        assert result['cross_backend_comparison'] == 'diagnostic_reduction_order_differs'
    assert load.body.index(nodes[-1]) < next(i for i, n in enumerate(load.body)
                                           if ast.unparse(n).startswith('parameters ='))


@pytest.mark.parametrize('tokens', [1, 15, 16, 17, 33])
@pytest.mark.parametrize('layout', ['canonical', 'token_minor', 'recorded_stride'])
def test_actual_kernel_addresses_two_independent_inputs_and_outputs(tokens, layout):
    kernel = copy.deepcopy(next(n for n in source('_triton_gdn_qk_norm.py').body
                                if isinstance(n, ast.FunctionDef) and n.name == '_paired_l2'))
    kernel.decorator_list = []
    for arg in kernel.args.args:
        arg.annotation = None
    qt, qh, qf = (2048, 128, 1) if layout == 'canonical' else (1, 128 * tokens, tokens)
    qbase, kbase, qout, kout = 100, 10**8 + 16777216, 3 * 10**8, 4 * 10**8
    if layout == 'recorded_stride':
        qt, qh, qf = 1, 1048576, 8192
    kt, kh, kf = 2 * 2048 + 3, 256, 2
    reads, writes, math_calls = [], [], []

    class Array(np.ndarray):
        def to(self, dtype):
            return self.astype(dtype).view(Array)

    def array(value):
        return np.asarray(value).view(Array)

    def load(address, mask, other):
        chosen = np.asarray(address)[np.broadcast_to(mask, np.shape(address))]
        reads.extend(chosen.tolist())
        return np.zeros(np.shape(address), dtype=np.float32).view(Array)

    def store(address, value, mask):
        writes.extend(np.asarray(address)[np.broadcast_to(mask, np.shape(address))].tolist())

    def l2(value):
        math_calls.append(value.shape)
        return value

    ids = [0, 0]
    tl = SimpleNamespace(program_id=lambda i: array(ids[i]), arange=lambda a, b: array(np.arange(a, b)),
                         int64=np.int64, load=load, store=store)
    ns = execute([kernel], {'tl': tl, '_l2': l2})
    for tile in range((tokens + 15) // 16):
        for head in range(16):
            ids[:] = [tile, head]
            ns['_paired_l2'](qbase, kbase, qout, kout, tokens, qt, qh, qf, kt, kh, kf, 16)
    expected_q = sorted(qbase + t * qt + h * qh + f * qf
                        for t in range(tokens) for h in range(16) for f in range(128))
    expected_k = sorted(kbase + t * kt + h * kh + f * kf
                        for t in range(tokens) for h in range(16) for f in range(128))
    assert sorted(reads) == expected_q + expected_k
    assert sorted(writes) == [*range(qout, qout + tokens * 2048), *range(kout, kout + tokens * 2048)]
    assert math_calls == [(16, 128)] * (((tokens + 15) // 16) * 16 * 2)


def test_declared_math_boundaries_and_live_store_dependency():
    tree = source('_triton_gdn_qk_norm.py')
    math = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_l2')
    assignments = {ast.unparse(n.targets[0]): ast.unparse(n.value)
                   for n in math.body if isinstance(n, ast.Assign)}
    assert assignments == {'work': 'values.to(tl.float32)', 'squared': 'work * work',
                           'total': 'tl.sum(squared, axis=1)', 'epsilon_added': 'total + 1e-06',
                           'inverse': 'libdevice.rsqrt(epsilon_added)',
                           'product': 'work * inverse[:, None]'}
    assert ast.unparse(math.body[-1]) == "return product.to(tl.bfloat16, fp_downcast_rounding='rtne')"
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not set(calls) & {'tl.sqrt', 'tl.fma', 'tl.dot', 'tl.rsqrt'}
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_paired_l2')
    stores = [n for n in ast.walk(kernel) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'tl.store']
    assert [ast.unparse(n.args[1]) for n in stores] == ['_l2(query)', '_l2(key)']


def test_actual_wrapper_live_strides_ownership_and_no_fp_fusion():
    wrapper = next(n for n in source('_triton_gdn_qk_norm.py').body
                   if isinstance(n, ast.FunctionDef) and n.name == 'normalize_pair')
    q = Tensor((1, 8192, 16, 128), (8192, 1, 1048576, 8192))
    k = Tensor(q.shape)
    allocations, launches = [], []

    def empty(shape, device, dtype):
        assert shape == q.shape and device == q.device and dtype == 'bf16'
        result = object()
        allocations.append(result)
        return result

    class Device:
        def __init__(self, device):
            assert device is q.device

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))
            return launch

    ns = execute([wrapper], {'torch': SimpleNamespace(empty=empty, bfloat16='bf16',
                                                      cuda=SimpleNamespace(device=Device)),
                             'triton': SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
                             '_paired_l2': Kernel()})
    result = ns['normalize_pair'](q, k)
    assert tuple(allocations) == result and result[0] is not result[1]
    assert launches == [((512, 16), (q, k, *result, 8192, 1, 1048576, 8192,
                                   2048, 128, 1, 16),
                        {'num_warps': 4, 'num_stages': 2, 'enable_fp_fusion': False})]


@pytest.mark.parametrize('backend', ['native', 'triton'])
def test_actual_bridge_preserves_other_operands_and_native_order(backend):
    prefill = copy.deepcopy(next(n for n in source('_flashinfer_gdn.py').body
                                 if isinstance(n, ast.FunctionDef) and n.name == 'prefill'))

    class RemoveImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

    prefill = RemoveImports().visit(prefill)
    events, received = [], []

    class Value:
        dtype = 'bf16'
        shape = (1, 33, 16, 128)
        device = 'synthetic'

        def __init__(self, label):
            self.label = label

        def float(self):
            events.append(self.label + '.float')
            return self

        def __mul__(self, other):
            events.append(self.label + ('.square' if self is other else '.multiply'))
            return Value(self.label)

        def sum(self, dim, keepdim):
            assert (dim, keepdim) == (-1, True)
            events.append(self.label + '.sum')
            return self

        def __add__(self, value):
            assert value == 1e-6
            events.append(self.label + '.eps')
            return self

        def to(self, dtype):
            assert dtype == 'bf16'
            events.append(self.label + '.bf16')
            return self

        def squeeze(self, dim):
            assert dim == 0
            return self

        def contiguous(self):
            events.append(self.label + '.contiguous')
            return self

        def exp(self):
            events.append(self.label + '.exp')
            return self

        def transpose(self, a, b):
            assert (a, b) == (-1, -2)
            events.append(self.label + '.transpose')
            return self

        def unsqueeze(self, dim):
            assert dim == 0
            return self

    def rsqrt(value):
        events.append(value.label + '.rsqrt')
        return value

    def pair(q, k, selected):
        assert (q.label, k.label, selected) == ('Q', 'K', 'triton')
        events.append('paired')
        return Value('Qf'), Value('Kf')

    def core(**kwargs):
        received.append(kwargs)
        return Value('output'), Value('updated')

    torch = SimpleNamespace(rsqrt=rsqrt, int64='i64', tensor=lambda *a, **k: Value('seqlens'))
    ns = execute([prefill], {'torch': torch, 'normalize_pair': pair, 'chunk_gated_delta_rule': core})
    values = [Value(n) for n in ('Q', 'K', 'V', 'decay', 'beta', 'initial')]
    ns['prefill'](*values, qk_norm_backend=backend)
    prefix = ['initial.transpose', 'initial.contiguous']
    if backend == 'native':
        prefix += [who + '.' + op for who in ('Q', 'K')
                   for op in ('float', 'square', 'sum', 'eps', 'rsqrt', 'multiply', 'bf16', 'contiguous')]
    else:
        prefix += ['paired', 'Qf.contiguous', 'Kf.contiguous']
    assert events == prefix + ['V.contiguous', 'decay.float', 'decay.exp', 'decay.contiguous',
                               'beta.float', 'beta.contiguous', 'updated.transpose', 'updated.contiguous']
    call = received[0]
    assert call['v'] is values[2] and call['g'] is values[3] and call['beta'] is values[4]
    assert call['initial_state'] is values[5]
    assert call['backend'] == 'flashinfer' and call['use_qk_l2norm_in_kernel'] is False
    assert call['output_final_state'] is True


def test_native_fallback_expression_and_error_propagation():
    bridge = next(n for n in source('_flashinfer_gdn.py').body
                  if isinstance(n, ast.FunctionDef) and n.name == 'prefill')
    native = next(n for n in bridge.body if isinstance(n, ast.FunctionDef) and n.name == 'normalized')
    helper = source('_gdn_qk_norm.py')
    fallback = next(n for n in ast.walk(helper) if isinstance(n, ast.FunctionDef) and n.name == 'normalized')
    assert ast.dump(native) == ast.dump(fallback)
    assert not any(isinstance(n, ast.Try) for n in ast.walk(helper))
    observer = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == 'execution')
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(observer))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('item', 'cpu', 'numpy', 'tolist', 'float', 'copy_')
                   for n in ast.walk(observer))
