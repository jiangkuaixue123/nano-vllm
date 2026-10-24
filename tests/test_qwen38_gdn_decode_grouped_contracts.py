"""Exercise construction and actual repeat routing without importing Torch."""
# ruff: noqa: S102 -- executes selected AST from this checkout only.
import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TREE = ast.parse((Path(__file__).parents[1] / 'nanovllm/qwen38/model.py').read_text())
DELTA = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'DeltaNet')


def run(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual_delta', 'exec'), namespace)


class Heads:
    def __init__(self):
        self.heads = list(range(16))

    def repeat_interleave(self, count, axis):
        if count != 3 or axis != 2:
            raise AssertionError('unexpected repetition')
        return [h for h in self.heads for _ in range(count)]


class GroupedDecodeControls(unittest.TestCase):
    def test_constructor_selection_and_scope(self):
        init = next(n for n in DELTA.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == 'self.gdn_decode_qk_layout')
        config = SimpleNamespace(linear_num_key_heads=16, linear_num_value_heads=48)
        for mode in (None, 'expanded', 'grouped', 'wrong'):
            for backend in ('native', 'flashinfer'):
                module = SimpleNamespace(gdn_backend=backend)
                env = {} if mode is None else {'NANOVLLM_GDN_DECODE_QK_LAYOUT': mode}
                with patch.dict(os.environ, env, clear=True):
                    if mode == 'wrong' or (mode == 'grouped' and backend == 'native'):
                        with self.assertRaises(ValueError):
                            run(init.body[start:start + 3], dict(self=module, config=config, os=os))
                    else:
                        run(init.body[start:start + 3], dict(self=module, config=config, os=os))
                        self.assertEqual(module.gdn_decode_qk_layout, mode or 'expanded')
        config.linear_num_key_heads = 15
        with patch.dict(os.environ, {'NANOVLLM_GDN_DECODE_QK_LAYOUT': 'grouped'}):
            with self.assertRaises(ValueError):
                run(init.body[start:start + 3], dict(self=module, config=config, os=os))

    def test_actual_forward_routes(self):
        forward = next(n for n in DELTA.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        start = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == 'decode_layout')
        for prefill in ('expanded', 'grouped'):
            for decode in ('expanded', 'grouped'):
                for previous in (False, True):
                    for count in (1, 2, 8192):
                        query, key = Heads(), Heads()
                        scope = dict(self=SimpleNamespace(gdn_prefill_qk_layout=prefill,
                                     gdn_decode_qk_layout=decode), previous=previous,
                                     count=count, repeats=3, query=query, key=key)
                        run(forward.body[start:start + 2], scope)
                        expanded = (decode if previous and count == 1 else prefill) == 'expanded'
                        if expanded:
                            self.assertEqual(scope['query'], [h // 3 for h in range(48)])
                            self.assertEqual(scope['key'], scope['query'])
                        else:
                            self.assertIs(scope['query'], query)
                            self.assertIs(scope['key'], key)


if __name__ == '__main__':
    unittest.main()
