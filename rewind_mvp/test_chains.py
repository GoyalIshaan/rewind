from dataclasses import replace
import unittest

import torch

from rewind_mvp.chains import describe_chains, form_chains
from rewind_mvp.hybrid import TinyHybridWindow, example_inputs
from rewind_mvp.matcher import find_updates


class ChainFixture(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant

    def forward(self, matrix, a, v, k):
        first = a * matrix + torch.outer(v, k)
        if self.variant == "single":
            return first
        if self.variant == "independent":
            return first, a * matrix + torch.outer(v + 1, k)
        if self.variant == "scale_dependency":
            return first, first.sum() * (matrix + 1) + torch.outer(v, k)
        if self.variant in ("tanh_gap", "clone_gap"):
            middle = torch.tanh(first) if self.variant == "tanh_gap" else first.clone()
            return first, a * middle + torch.outer(v, k)
        second = a * first + torch.outer(v + 1, k)
        if self.variant in ("fork", "fork_and_independent"):
            third = a * first + torch.outer(v + 2, k)
            if self.variant == "fork":
                return second, third
            other = a * matrix + torch.outer(v + 3, k)
            return second, third, other
        if self.variant == "two_chains":
            other_first = a * matrix + torch.outer(v + 2, k)
            other_second = a * other_first + torch.outer(v + 3, k)
            return second, other_second
        return second, torch.sin(first).sum()


def fixture_matches(variant):
    args = (torch.ones(4, 3), torch.tensor(0.5), torch.ones(4), torch.ones(3))
    graph = torch.export.export(ChainFixture(variant), args).graph_module.graph
    return graph, find_updates(graph)


class ChainTests(unittest.TestCase):
    def test_empty_input(self):
        analysis = form_chains([])
        self.assertEqual(analysis.chains, ())
        self.assertEqual(analysis.skipped_components, ())

    def test_single_candidate(self):
        _, matches = fixture_matches("single")
        chain, = form_chains(matches).chains
        self.assertEqual(chain.updates, tuple(matches))
        self.assertIs(chain.incoming_state, matches[0].previous)

    def test_hybrid_has_one_three_update_chain(self):
        graph = torch.export.export(TinyHybridWindow(), example_inputs()).graph_module.graph
        matches = find_updates(graph)
        before = str(graph)
        nodes = tuple(graph.nodes)
        users = [tuple(node.users) for node in nodes]
        analysis = form_chains(matches)
        chain, = analysis.chains
        self.assertEqual(chain.updates, tuple(matches))
        self.assertEqual(len(chain.updates), 3)
        self.assertIs(chain.incoming_state, matches[0].previous)
        self.assertEqual(analysis.skipped_components, ())
        self.assertEqual(str(graph), before)
        self.assertEqual(tuple(graph.nodes), nodes)
        self.assertEqual([tuple(node.users) for node in nodes], users)
        self.assertEqual(form_chains(matches), analysis)

    def test_input_order_does_not_determine_state_order(self):
        _, matches = fixture_matches("linear")
        original = matches.copy()
        chain, = form_chains(list(reversed(matches))).chains
        self.assertEqual(chain.updates, tuple(original))
        self.assertEqual(matches, original)

    def test_same_incoming_matrix_does_not_link_independent_updates(self):
        _, matches = fixture_matches("independent")
        analysis = form_chains(matches)
        self.assertEqual([len(chain.updates) for chain in analysis.chains], [1, 1])
        self.assertIs(analysis.chains[0].incoming_state, analysis.chains[1].incoming_state)

    def test_two_chains_stay_separate(self):
        _, matches = fixture_matches("two_chains")
        analysis = form_chains(matches)
        self.assertEqual([len(chain.updates) for chain in analysis.chains], [2, 2])
        for chain in analysis.chains:
            self.assertIs(chain.updates[1].previous, chain.updates[0].result)

    def test_dependency_through_scale_is_not_a_state_edge(self):
        _, matches = fixture_matches("scale_dependency")
        analysis = form_chains(matches)
        self.assertEqual([len(chain.updates) for chain in analysis.chains], [1, 1])

    def test_unsupported_intermediate_operations_are_not_skipped(self):
        for variant in ("tanh_gap", "clone_gap"):
            with self.subTest(variant=variant):
                _, matches = fixture_matches(variant)
                self.assertEqual([len(chain.updates) for chain in form_chains(matches).chains],
                                 [1, 1])

    def test_non_state_consumers_do_not_count_as_forks(self):
        _, matches = fixture_matches("linear")
        analysis = form_chains(matches)
        self.assertEqual(len(analysis.chains[0].updates), 2)
        self.assertEqual(analysis.skipped_components, ())

    def test_fork_skips_whole_component(self):
        _, matches = fixture_matches("fork")
        analysis = form_chains(matches)
        self.assertEqual(analysis.chains, ())
        component, = analysis.skipped_components
        self.assertEqual(len(component.updates), 3)
        fork, = component.forks
        self.assertIs(fork.state, matches[0].result)
        self.assertEqual(fork.successors, tuple(matches[1:]))

    def test_unbranched_component_survives_beside_a_fork(self):
        _, matches = fixture_matches("fork_and_independent")
        analysis = form_chains(matches)
        self.assertEqual(len(analysis.skipped_components), 1)
        chain, = analysis.chains
        self.assertEqual(chain.updates, (matches[-1],))
        self.assertIn("state fork unsupported", describe_chains(analysis))

    def test_names_do_not_determine_links(self):
        graph, matches = fixture_matches("linear")
        for i, node in enumerate(graph.nodes):
            node.name = f"arbitrary_{i}"
        chain, = form_chains(matches).chains
        self.assertEqual(chain.updates, tuple(matches))

    def test_duplicate_results_rejected(self):
        _, matches = fixture_matches("single")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            form_chains(matches + matches)

    def test_different_graphs_rejected(self):
        _, first = fixture_matches("single")
        _, second = fixture_matches("single")
        with self.assertRaisesRegex(ValueError, "same graph"):
            form_chains(first + second)

    def test_invalid_cyclic_match_records_rejected(self):
        # Synthetic bad analysis input, not a valid exported graph.
        _, matches = fixture_matches("linear")
        cyclic = [replace(matches[0], previous=matches[1].result), matches[1]]
        with self.assertRaisesRegex(ValueError, "cycle"):
            form_chains(cyclic)


if __name__ == "__main__":
    unittest.main()
