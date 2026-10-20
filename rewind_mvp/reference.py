"""Executable semantics for one synthetic recurrent layer; no compiler yet."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Inputs:
    # a: [T]; v: [T, V]; k, q: [T, K]; state: [V, K].
    a: torch.Tensor
    v: torch.Tensor
    k: torch.Tensor
    q: torch.Tensor

    def prefix(self, length: int) -> "Inputs":
        if not 0 <= length <= len(self.a):
            raise ValueError("prefix length outside input window")
        return Inputs(*(x[:length] for x in (self.a, self.v, self.k, self.q)))


def random_case(seed: int = 0, tokens: int = 8, values: int = 5, keys: int = 3):
    """CPU float64 fixture, with a private generator for reproducibility."""
    g = torch.Generator().manual_seed(seed)
    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)
    state = randn(values, keys)
    inputs = Inputs(
        torch.rand(tokens, generator=g, dtype=torch.float64),
        randn(tokens, values), randn(tokens, keys), randn(tokens, keys),
    )
    return state, inputs


def step(state: torch.Tensor, a, v, k, q):
    """Pure update: the incoming state is never overwritten."""
    updated = a * state + torch.outer(v, k)
    output = updated @ q
    return updated, output


def run(state: torch.Tensor, inputs: Inputs):
    """Oracle: return every output and every state boundary, including S_0."""
    states = [state.clone()]
    outputs = []
    for a, v, k, q in zip(inputs.a, inputs.v, inputs.k, inputs.q, strict=True):
        state, output = step(state, a, v, k, q)
        states.append(state.clone())
        outputs.append(output)
    return states, outputs


def replay(checkpoint: torch.Tensor, inputs: Inputs, accepted: int):
    """Reconstruct only the accepted boundary, retaining no state history.

    This deliberately preserves the oracle's update order. It is a reference
    strategy, not generated code, output-only verification, or a speed claim.
    Callers must retain the checkpoint and recurrence inputs unchanged.
    """
    prefix = inputs.prefix(accepted)
    state = checkpoint.clone()
    for a, v, k in zip(prefix.a, prefix.v, prefix.k, strict=True):
        state = a * state + torch.outer(v, k)
    return state


if __name__ == "__main__":
    initial, inputs = random_case()
    states, outputs = run(initial, inputs)
    for accepted in range(len(inputs.a) + 1):
        restored = replay(initial, inputs, accepted)
        torch.testing.assert_close(restored, states[accepted], rtol=0, atol=0)
    print(f"Checked all {len(states)} accepted-prefix boundaries on CPU float64.")
    print(f"State shape: {tuple(initial.shape)}; output shape: {tuple(outputs[0].shape)}")
    print("Reference only: no compiler, generated kernels, or performance result yet.")
