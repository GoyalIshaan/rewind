import unittest

import torch

from rewind_mvp.program import speculative_recurrent_window
from rewind_mvp.reference import random_case, run


class ProgramTests(unittest.TestCase):
    def test_all_accepted_prefixes(self):
        for tokens, values, keys in [(1, 1, 1), (4, 5, 3), (8, 3, 7)]:
            state, inputs = random_case(tokens=tokens, values=values, keys=keys)
            arguments = (state, inputs.a, inputs.k, inputs.v, inputs.q)
            original_values = [x.clone() for x in arguments]
            states, outputs = run(state, inputs)
            for accepted in range(tokens + 1):
                with self.subTest(tokens=tokens, accepted=accepted):
                    actual_outputs, actual_state = speculative_recurrent_window(
                        *arguments, torch.tensor([accepted], dtype=torch.int64))
                    torch.testing.assert_close(actual_outputs, torch.stack(outputs),
                                               rtol=0, atol=0)
                    torch.testing.assert_close(actual_state, states[accepted],
                                               rtol=0, atol=0)
            for actual, original in zip(arguments, original_values):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
