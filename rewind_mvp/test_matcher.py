import unittest

import torch

from rewind_mvp.hybrid import TinyHybridWindow, example_inputs
from rewind_mvp.matcher import find_updates, match_state_update


class Expression(torch.nn.Module):
    def __init__(self, variant="supported"):
        super().__init__()
        self.variant = variant

    def forward(self, matrix, coefficient, left, right):
        scaled = coefficient * matrix
        if self.variant == "elementwise":
            return scaled + left[:, None] * right[None, :]
        return torch.add(scaled, torch.outer(left, right),
                         alpha=2 if self.variant == "alpha_two" else 1)


class MatcherTests(unittest.TestCase):
    def test_hybrid_update_candidates_and_links_without_mutation(self):
        exported = torch.export.export(TinyHybridWindow(), example_inputs())
        graph = exported.graph_module.graph
        before = str(graph)
        matches = find_updates(graph)
        self.assertEqual(len(matches), 3)
        # Check actual edges rather than generated node names.
        for previous, current in zip(matches, matches[1:]):
            self.assertIs(current.previous, previous.result)
        self.assertEqual(str(graph), before)

    def test_shapes_and_supported_spelling(self):
        matrix = torch.randn(4, 3)
        for variant, coefficient, expected in [
            ("supported", torch.tensor(0.5), True),
            ("alpha_two", torch.tensor(0.5), False),
            ("elementwise", torch.tensor(0.5), False),
            ("supported", torch.randn(4, 3), False),
        ]:
            with self.subTest(variant=variant, scale_shape=tuple(coefficient.shape)):
                exported = torch.export.export(Expression(variant),
                    (matrix, coefficient, torch.randn(4), torch.randn(3)))
                output = next(n for n in exported.graph_module.graph.nodes if n.op == "output")
                match = match_state_update(output.args[0][0])
                self.assertEqual(match is not None, expected)

    def test_missing_metadata_is_not_guessed(self):
        graph = torch.export.export(TinyHybridWindow(), example_inputs()).graph_module.graph
        match = find_updates(graph)[0]
        match.scale.meta.pop("val")
        self.assertIsNone(match_state_update(match.result))


if __name__ == "__main__":
    unittest.main()
