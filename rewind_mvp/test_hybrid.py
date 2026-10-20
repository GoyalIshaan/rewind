import unittest

import torch

from rewind_mvp.hybrid import TinyHybridWindow, accepted_prefix, example_inputs


class HybridTests(unittest.TestCase):
    def test_prefix_stops_at_first_mismatch(self):
        predictions = torch.eye(4, dtype=torch.float64) * 10
        for count in range(4):
            tokens = torch.tensor([0, 1, 2])
            if count < 3:
                tokens[count] = (tokens[count] + 1) % 4
            actual = accepted_prefix(predictions[0], predictions[1:], tokens)
            self.assertEqual(actual.item(), count)

    def test_every_boundary_restores_state_and_cache_length(self):
        model = TinyHybridWindow()
        inputs = example_inputs()
        greedy = inputs[0].clone()
        greedy[0] = inputs[-1].argmax()
        # Causality lets us construct a fully accepted proposal sequentially.
        for t in range(1, len(greedy)):
            logits = model(greedy, *inputs[1:])[0]
            greedy[t] = logits[t - 1].argmax()
        for count in range(len(greedy) + 1):
            tokens = greedy.clone()
            if count < len(tokens):
                tokens[count] = (tokens[count] + 1) % model.embedding.shape[0]
            with self.subTest(count=count):
                result = model(tokens, *inputs[1:])
                expected = inputs[1].clone()
                for token in tokens[:count]:
                    x = model.embedding[token]
                    expected = (torch.sigmoid(x @ model.decay) * expected
                                + torch.outer(x @ model.recurrent_value,
                                              x @ model.recurrent_key))
                torch.testing.assert_close(result[1], expected)
                self.assertEqual(result[-1].item(), count)
                self.assertEqual(result[-2].item(), inputs[2].shape[0] + count)

    def test_future_token_does_not_change_earlier_logits(self):
        model = TinyHybridWindow()
        inputs = example_inputs()
        changed = inputs[0].clone()
        changed[-1] = (changed[-1] + 1) % model.embedding.shape[0]
        torch.testing.assert_close(model(*inputs)[0][:-1],
                                   model(changed, *inputs[1:])[0][:-1], rtol=0, atol=0)

    def test_export_matches_eager_and_inputs_are_preserved(self):
        model = TinyHybridWindow()
        inputs = example_inputs()
        originals = [value.clone() for value in inputs]
        exported = torch.export.export(model, inputs).module()
        for actual, expected in zip(exported(*inputs), model(*inputs)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(inputs, originals):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
