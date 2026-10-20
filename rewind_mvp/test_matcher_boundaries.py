"""Acceptance/rejection tests for the current matcher, without rewriting graphs.

Most cases export executable PyTorch expressions. Metadata-corruption tests are
explicitly synthetic: they test conservative rejection of untrusted/incomplete IR.
"""

import unittest

import torch

from rewind_mvp.hybrid import TinyHybridWindow, example_inputs
from rewind_mvp.matcher import find_updates, match_state_update


def decay_part(a, matrix):
    return a * matrix


def outer_part(v, k):
    return torch.outer(v, k)


class Pattern(torch.nn.Module):
    def __init__(self, variant="exact"):
        super().__init__()
        self.variant = variant

    def forward(self, matrix, a, v, k):
        if self.variant == "helpers":
            scaled = decay_part(a, matrix)
            unrelated = torch.sin(v).sum()
            return scaled + outer_part(v, k), unrelated
        if self.variant == "literal_scale":
            return 0.5 * matrix + torch.outer(v, k)
        if self.variant == "reversed_mul":
            return matrix * a + torch.outer(v, k)
        scaled = a * matrix
        product = torch.outer(v, k)
        if self.variant == "two_independent":
            first = scaled + product
            second = a * matrix + torch.outer(v + 1, k)
            return first, second
        if self.variant == "alpha_two":
            return torch.add(scaled, product, alpha=2)
        if self.variant == "reversed_add":
            return product + scaled
        if self.variant == "subtraction":
            return scaled - product
        if self.variant == "plain_bias":
            return scaled + matrix
        if self.variant == "nonlinear_scaled":
            return torch.sigmoid(scaled) + product
        if self.variant == "broadcast_outer":
            return scaled + v[:, None] * k[None, :]
        if self.variant == "matmul_outer":
            return scaled + v[:, None] @ k[None, :]
        return scaled + product


def export_pattern(variant="exact", dtype=torch.float32, shape=(4, 3), scale_shape=()):
    generator = torch.Generator().manual_seed(71)
    m, n = shape
    args = (torch.randn(m, n, generator=generator, dtype=dtype),
            torch.full(scale_shape, 0.5, dtype=dtype),
            torch.randn(m, generator=generator, dtype=dtype),
            torch.randn(n, generator=generator, dtype=dtype))
    return torch.export.export(Pattern(variant), args)


class MatcherBoundaryTests(unittest.TestCase):
    def test_exact_match_returns_correct_operand_references(self):
        graph = export_pattern().graph_module.graph
        match, = find_updates(graph)
        matrix, a, v, k = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertIs(match.previous, matrix)
        self.assertIs(match.scale, a)
        self.assertIs(match.value, v)
        self.assertIs(match.key, k)
        self.assertIs(match.result.args[0].args[1], matrix)

    def test_helper_functions_and_unrelated_work_do_not_hide_match(self):
        self.assertEqual(len(find_updates(export_pattern("helpers").graph_module.graph)), 1)

    def test_two_independent_expressions_are_both_candidates(self):
        # Expected limitation: neither expression is a recurrent chain.
        matches = find_updates(export_pattern("two_independent").graph_module.graph)
        self.assertEqual(len(matches), 2)
        self.assertIs(matches[0].previous, matches[1].previous)

    def test_generated_node_names_do_not_matter(self):
        graph = export_pattern().graph_module.graph
        for i, node in enumerate(graph.nodes):
            node.name = f"renamed_{i}"
        graph.lint()
        self.assertEqual(len(find_updates(graph)), 1)

    def test_every_non_add_node_is_rejected_as_a_root(self):
        graph = torch.export.export(TinyHybridWindow(), example_inputs()).graph_module.graph
        checked = 0
        for node in graph.nodes:
            if node.op != "call_function" or node.target != torch.ops.aten.add.Tensor:
                self.assertIsNone(match_state_update(node), node.name)
                checked += 1
        self.assertGreater(checked, 50)

    def test_full_graph_nodes_edges_metadata_and_code_are_unchanged(self):
        exported = torch.export.export(TinyHybridWindow(), example_inputs())
        graph = exported.graph_module.graph
        nodes = tuple(graph.nodes)
        edges = [(n.args, dict(n.kwargs), tuple(n.users)) for n in nodes]
        metadata = [dict(n.meta) for n in nodes]
        code = exported.graph_module.code
        first = find_updates(graph)
        second = find_updates(graph)
        self.assertEqual(len(first), 3)
        self.assertEqual(first, second)
        self.assertEqual(tuple(graph.nodes), nodes)
        self.assertEqual(edges, [(n.args, dict(n.kwargs), tuple(n.users)) for n in nodes])
        self.assertEqual(code, exported.graph_module.code)
        for node, before in zip(nodes, metadata):
            self.assertEqual(set(node.meta), set(before))
            for key, value in before.items():
                self.assertIs(node.meta[key], value)

    def test_missing_metadata_on_any_operand_rejects(self):
        for field in range(7):
            with self.subTest(operand=field):
                graph = export_pattern().graph_module.graph
                match, = find_updates(graph)
                root = match.result
                nodes = [root, root.args[0], root.args[1], match.previous,
                         match.scale, match.value, match.key]
                nodes[field].meta.pop("val")
                self.assertIsNone(match_state_update(root))

    def test_non_tensor_metadata_rejects(self):
        graph = export_pattern().graph_module.graph
        match, = find_updates(graph)
        match.scale.meta["val"] = 0.5
        self.assertIsNone(match_state_update(match.result))

    def test_inconsistent_result_shape_rejects(self):
        graph = export_pattern().graph_module.graph
        match, = find_updates(graph)
        match.result.meta["val"] = torch.empty(4, 4)
        self.assertIsNone(match_state_update(match.result))

    def test_mixed_dtype_metadata_rejects(self):
        graph = export_pattern().graph_module.graph
        match, = find_updates(graph)
        match.key.meta["val"] = torch.empty(3, dtype=torch.float64)
        self.assertIsNone(match_state_update(match.result))

    def test_mixed_device_metadata_rejects(self):
        graph = export_pattern().graph_module.graph
        match, = find_updates(graph)
        match.key.meta["val"] = torch.empty(3, device="meta")
        self.assertIsNone(match_state_update(match.result))

    def test_integer_tensor_pattern_rejects(self):
        args = (torch.ones(4, 3, dtype=torch.int64), torch.tensor(2),
                torch.ones(4, dtype=torch.int64), torch.ones(3, dtype=torch.int64))
        exported = torch.export.export(Pattern(), args)
        self.assertEqual(find_updates(exported.graph_module.graph), [])

    def test_symbolic_dimension_rejects(self):
        m = torch.export.Dim("rows", min=2, max=8)
        args = (torch.ones(4, 3), torch.tensor(0.5), torch.ones(4), torch.ones(3))
        exported = torch.export.export(Pattern(), args,
                                      dynamic_shapes=({0: m}, {}, {0: m}, {}))
        self.assertEqual(find_updates(exported.graph_module.graph), [])


def positive_test(dtype, shape):
    def check(self):
        self.assertEqual(len(find_updates(export_pattern(dtype=dtype, shape=shape)
                                         .graph_module.graph)), 1)
    return check


def rejection_test(variant, scale_shape=()):
    def check(self):
        exported = export_pattern(variant, scale_shape=scale_shape)
        self.assertEqual(find_updates(exported.graph_module.graph), [])
    return check


for name, dtype, shape in [
    ("float32_nonsquare", torch.float32, (4, 3)),
    ("float64_nonsquare", torch.float64, (3, 7)),
    ("unit_dimensions", torch.float32, (1, 1)),
]:
    setattr(MatcherBoundaryTests, f"test_accept_{name}", positive_test(dtype, shape))

for name, variant, scale_shape in [
    ("alpha_two", "alpha_two", ()),
    ("reversed_add", "reversed_add", ()),
    ("reversed_mul", "reversed_mul", ()),
    ("subtraction", "subtraction", ()),
    ("plain_bias", "plain_bias", ()),
    ("nonlinear_scaled", "nonlinear_scaled", ()),
    ("broadcast_outer", "broadcast_outer", ()),
    ("matmul_outer", "matmul_outer", ()),
    ("literal_scale", "literal_scale", ()),
    ("matrix_scale", "exact", (4, 3)),
    ("one_element_vector_scale", "exact", (1,)),
]:
    setattr(MatcherBoundaryTests, f"test_reject_{name}", rejection_test(variant, scale_shape))


if __name__ == "__main__":
    unittest.main()
