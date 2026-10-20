import unittest

import torch

from rewind_mvp.chains import form_chains
from rewind_mvp.history import describe_histories, find_stack_histories
from rewind_mvp.hybrid import TinyHybridWindow, example_inputs
from rewind_mvp.matcher import find_updates


class HistoryFixture(torch.nn.Module):
    def __init__(self, variant="exact"):
        super().__init__()
        self.variant = variant

    def forward(self, matrix, a, v, k, index):
        first = a * matrix + torch.outer(v, k)
        second = a * first + torch.outer(v + 1, k)
        outputs = torch.stack([first @ k, second @ k])
        states = [matrix, first, second]
        if self.variant == "missing_initial":
            states = [first, second]
        elif self.variant == "missing_middle":
            states = [matrix, second]
        elif self.variant == "reordered":
            states = [matrix, second, first]
        elif self.variant == "duplicate":
            states = [matrix, first, first]
        elif self.variant == "extra_state":
            states = [matrix, first, second, matrix * 2]
        elif self.variant == "unrelated_state":
            states = [matrix, first, matrix + 1]
        elif self.variant == "no_stack":
            return outputs, second
        elif self.variant == "cat_storage":
            history = torch.cat([s.unsqueeze(0) for s in states])
            return outputs, history.index_select(0, index).squeeze(0)

        if self.variant == "stack_dim_one":
            history = torch.stack(states, dim=1)
            return outputs, history.index_select(1, index).squeeze(1)
        history = torch.stack(states)
        if self.variant == "return_history":
            return outputs, history.index_select(0, index).squeeze(0), history
        if self.variant == "history_mean":
            return outputs, history.index_select(0, index).squeeze(0), history.mean()
        if self.variant == "unsupported_consumer":
            return outputs, history.sum()
        if self.variant == "two_selections":
            first_selection = history.index_select(0, index).squeeze(0)
            second_selection = history.index_select(0, (index + 1) % 3).squeeze(0)
            return outputs, first_selection, second_selection
        dim = 1 if self.variant == "selection_dim_one" else 0
        selected = history.index_select(dim, index)
        if self.variant == "no_squeeze":
            return outputs, selected
        if self.variant == "reshape_instead":
            return outputs, selected.reshape(4, 3)
        dim = 1 if self.variant == "squeeze_dim_one" else 0
        retrieved = selected.squeeze(dim)
        if self.variant == "selection_escapes":
            return outputs, retrieved, selected
        if self.variant == "state_escapes":
            return outputs, retrieved, first
        if self.variant == "selected_used_twice":
            return outputs, retrieved, retrieved.sum()
        return outputs, retrieved


def exported_fixture(variant="exact", index=None):
    args = (torch.ones(4, 3), torch.tensor(0.5), torch.ones(4), torch.ones(3),
            torch.tensor([1], dtype=torch.int64) if index is None else index)
    return torch.export.export(HistoryFixture(variant), args)


def analyze(exported):
    graph = exported.graph_module.graph
    return find_stack_histories(form_chains(find_updates(graph)).chains)


class HistoryTests(unittest.TestCase):
    def test_hybrid_match_record_has_correct_connections(self):
        exported = torch.export.export(TinyHybridWindow(), example_inputs())
        analysis = analyze(exported)
        self.assertEqual(analysis.rejections, ())
        match, = analysis.matches
        self.assertEqual(len(match.chain.updates), 3)
        self.assertIs(match.states[0], match.chain.incoming_state)
        self.assertEqual(match.states[1:], tuple(u.result for u in match.chain.updates))
        self.assertIs(match.selection_node.args[0], match.storage_node)
        self.assertIs(match.selection_node.args[2], match.selected_index)
        self.assertIs(match.selected_state.args[0], match.selection_node)
        self.assertEqual(match.required_index_range, (0, 3))

    def test_runtime_index_not_specialized_and_all_boundaries_work(self):
        exported = exported_fixture()
        match, = analyze(exported).matches
        self.assertEqual(match.selected_index.op, "placeholder")
        module = exported.module()
        s, a, v, k = torch.ones(4, 3), torch.tensor(0.5), torch.ones(4), torch.ones(3)
        first = a * s + torch.outer(v, k)
        second = a * first + torch.outer(v + 1, k)
        for count, expected in enumerate([s, first, second]):
            torch.testing.assert_close(module(s, a, v, k, torch.tensor([count]))[1], expected)

    def test_read_only_including_execution_and_metadata(self):
        exported = exported_fixture()
        graph = exported.graph_module.graph
        nodes = tuple(graph.nodes)
        before = str(graph)
        edges = [(n.args, dict(n.kwargs), tuple(n.users)) for n in nodes]
        metadata = [dict(n.meta) for n in nodes]
        args = (torch.ones(4, 3), torch.tensor(0.5), torch.ones(4), torch.ones(3), torch.tensor([1]))
        outputs_before = exported.module()(*args)
        first = analyze(exported)
        self.assertEqual(first, analyze(exported))
        self.assertEqual(str(graph), before)
        self.assertEqual(tuple(graph.nodes), nodes)
        self.assertEqual(edges, [(n.args, dict(n.kwargs), tuple(n.users)) for n in nodes])
        for node, old in zip(nodes, metadata):
            self.assertEqual(set(node.meta), set(old))
            for key, value in old.items():
                self.assertIs(node.meta[key], value)
        for actual, expected in zip(exported.module()(*args), outputs_before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_names_are_not_used_for_matching(self):
        exported = exported_fixture()
        for i, node in enumerate(exported.graph_module.graph.nodes):
            node.name = f"renamed_{i}"
        self.assertEqual(len(analyze(exported).matches), 1)

    def test_empty_chain_input(self):
        analysis = find_stack_histories(())
        self.assertEqual(analysis.matches, ())
        self.assertEqual(analysis.rejections, ())

    def test_missing_metadata_rejects_each_history_stage(self):
        for field in ("storage_node", "selection_node", "selected_index", "selected_state"):
            with self.subTest(field=field):
                exported = exported_fixture()
                match, = analyze(exported).matches
                getattr(match, field).meta.pop("val")
                result = analyze(exported)
                self.assertEqual(result.matches, ())
                self.assertTrue(result.rejections)

    def test_inconsistent_retrieved_shape_rejects(self):
        exported = exported_fixture()
        match, = analyze(exported).matches
        match.selected_state.meta["val"] = torch.empty(3, 4)
        self.assertEqual(analyze(exported).matches, ())

    def test_printed_record_exposes_all_detected_nodes(self):
        analysis = analyze(exported_fixture())
        match, = analysis.matches
        text = describe_histories(analysis)
        for field in ("storage_node", "selection_node", "selected_index", "selected_state"):
            self.assertIn(f"{field}: {getattr(match, field).name}", text)
        self.assertIn("not proven here", text)

    def test_multiple_indices_rejects(self):
        exported = exported_fixture(index=torch.tensor([0, 1]))
        result = analyze(exported)
        self.assertEqual(result.matches, ())
        self.assertIn("shape-[1]", result.rejections[0].reason)

    def test_int32_index_is_conservatively_rejected(self):
        self.assertEqual(analyze(exported_fixture(index=torch.tensor([1], dtype=torch.int32)))
                         .matches, ())


def supported_test(variant):
    def check(self):
        result = analyze(exported_fixture(variant))
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.rejections, ())
    return check


def rejected_test(variant, reason):
    def check(self):
        result = analyze(exported_fixture(variant))
        self.assertEqual(result.matches, ())
        self.assertTrue(any(reason in r.reason for r in result.rejections),
                        describe_histories(result))
    return check


for variant in ("exact", "state_escapes", "selected_used_twice"):
    setattr(HistoryTests, f"test_accept_{variant}", supported_test(variant))

for variant, reason in [
    ("missing_initial", "every chain result in order"),
    ("missing_middle", "every chain result in order"),
    ("reordered", "every chain result in order"),
    ("duplicate", "every chain result in order"),
    ("extra_state", "every chain result in order"),
    ("unrelated_state", "every chain result in order"),
    ("no_stack", "no direct stack"),
    ("cat_storage", "no direct stack"),
    ("stack_dim_one", "stack dimension"),
    ("return_history", "exactly one direct consumer"),
    ("history_mean", "exactly one direct consumer"),
    ("two_selections", "exactly one direct consumer"),
    ("unsupported_consumer", "history consumer must be"),
    ("selection_dim_one", "selection dimension"),
    ("no_squeeze", "selection consumer must be"),
    ("reshape_instead", "selection consumer must be"),
    ("squeeze_dim_one", "squeeze dimension"),
    ("selection_escapes", "exactly one squeeze consumer"),
]:
    setattr(HistoryTests, f"test_reject_{variant}", rejected_test(variant, reason))


if __name__ == "__main__":
    unittest.main()
