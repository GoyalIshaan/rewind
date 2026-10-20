import unittest

import torch

from rewind_mvp.reference import Inputs, random_case, replay, run


class ReferenceTests(unittest.TestCase):
    def test_hand_computed_case(self):
        state = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        inputs = Inputs(
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([[2., -1.]], dtype=torch.float64),
            torch.tensor([[3., 1.]], dtype=torch.float64),
            torch.tensor([[1., 2.]], dtype=torch.float64),
        )
        states, outputs = run(state, inputs)
        torch.testing.assert_close(states[-1], torch.tensor(
            [[6.5, 3.], [-1.5, 1.]], dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(outputs[-1], torch.tensor(
            [12.5, 0.5], dtype=torch.float64), rtol=0, atol=0)

    def test_every_rejection_boundary_and_continuation(self):
        for seed in range(5):
            for values, keys in [(1, 1), (5, 3), (3, 7)]:
                initial, inputs = random_case(seed, values=values, keys=keys)
                original = initial.clone()
                states, _ = run(initial, inputs)
                _, continuation = random_case(seed + 100, tokens=4,
                                               values=values, keys=keys)
                for accepted in range(len(inputs.a) + 1):
                    with self.subTest(seed=seed, shape=(values, keys), accepted=accepted):
                        restored = replay(initial, inputs, accepted)
                        torch.testing.assert_close(restored, states[accepted], rtol=0, atol=0)
                        # The continuation is independently chosen, not the rejected suffix.
                        actual_states, actual_outputs = run(restored, continuation)
                        expected_states, expected_outputs = run(states[accepted], continuation)
                        for actual, expected in zip(actual_states + actual_outputs,
                                                    expected_states + expected_outputs):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(initial, original, rtol=0, atol=0)

    def test_invalid_accepted_length(self):
        initial, inputs = random_case()
        for accepted in [-1, len(inputs.a) + 1]:
            with self.assertRaises(ValueError):
                replay(initial, inputs, accepted)


if __name__ == "__main__":
    unittest.main()
