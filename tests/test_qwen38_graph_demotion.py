"""CPU regression for hybrid state ownership at a decode graph boundary."""

from types import SimpleNamespace

import pytest
from test_qwen38_runner import Logits, MODULE, RequestRunner, Row


class Buffer:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return Buffer(self.values[key[2]])

    def clone(self):
        return Buffer(self.values.copy())


@pytest.mark.parametrize("linear_first", [True, False])
def test_graph_boundary_preserves_hybrid_state_and_invalidates_pool(linear_first):
    key, value = Buffer([1, 2, 99]), Buffer([3, 4, 99])
    conv, recurrent = object(), object()
    attention = SimpleNamespace(capacity=2, key=key, value=value)
    linear = SimpleNamespace(capacity=2, key=None, value=None,
                             conv=conv, recurrent=recurrent, fresh=False)
    layers = [linear, attention] if linear_first else [attention, linear]
    state = SimpleNamespace(layers=layers, capacity=2, positions=object(), length=1)

    class Model:
        def __call__(self, inputs, *, state):
            assert state.capacity == 0 and state.positions is None
            assert all(layer.capacity == 0 for layer in state.layers)
            assert linear.key is linear.value is None
            assert linear.conv is conv and linear.recurrent is recurrent
            assert not linear.fresh
            assert attention.key is not key and attention.value is not value
            assert attention.key.values == [1, 2]
            assert attention.value.values == [3, 4]
            state.length += len(inputs[0])
            return Logits([Row([0., 1.])], 2)

    runner = RequestRunner(Model(), max_context_tokens=4, tensor_factory=lambda x: x)
    graph = SimpleNamespace(capacity=2)
    request = MODULE._Request(state, length=2, graph=graph, graph_bucket=2)
    runner._requests["boundary"] = request
    runner._graph_pool[2] = graph
    runner._graph_busy.add(2)
    runner.append("boundary", [1])
    assert request.length == state.length == 3
    assert request.graph is None and request.graph_bucket == 0
    assert not runner._graph_pool and not runner._graph_busy
    assert runner._acquire_graph_state(2) is None
    runner.append("boundary", [1])
    assert request.length == state.length == 4
    assert runner.release("boundary")
    assert not runner.active_request_ids
