"""The single PyTorch function our first compiler pass will target."""

import torch


def speculative_recurrent_window(
    initial_state: torch.Tensor,
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    accepted_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute T tentative steps and return outputs plus the accepted state.

    Shapes: initial_state [V, K], a [T], k/q [T, K], v [T, V].
    T >= 1. Floating tensors share a dtype and device. accepted_count is an
    int64 tensor of shape [1] on that device, with a value in [0, T].
    Inputs remain unchanged during the call. There is no model or verifier:
    accepted_count is supplied by the caller for this controlled fixture.
    """
    state = initial_state.clone()
    snapshots = [state]
    outputs = []

    for t in range(a.shape[0]):
        # Each update creates a fresh tensor, preserving earlier snapshots.
        state = a[t] * state + torch.outer(v[t], k[t])
        output = state @ q[t]
        snapshots.append(state)
        outputs.append(output)

    # Snapshot 0 is the initial state; snapshot m follows m accepted tokens.
    history = torch.stack(snapshots)
    accepted_state = torch.index_select(history, 0, accepted_count).squeeze(0)
    return torch.stack(outputs), accepted_state
