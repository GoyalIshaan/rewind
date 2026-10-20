"""Read-only recognition of stack -> index_select -> squeeze state histories."""

from dataclasses import dataclass

import torch
from torch.fx import Node

from rewind_mvp.chains import RecurrenceChain, form_chains
from rewind_mvp.matcher import find_updates, get_tensor_metadata, is_call


@dataclass(frozen=True)
class HistoryMatch:
    chain: RecurrenceChain
    states: tuple[Node, ...]
    storage_node: Node
    selection_node: Node
    selected_index: Node
    selected_state: Node
    required_index_range: tuple[int, int]


@dataclass(frozen=True)
class HistoryRejection:
    chain: RecurrenceChain
    storage_node: Node | None
    reason: str


@dataclass(frozen=True)
class HistoryAnalysis:
    matches: tuple[HistoryMatch, ...]
    rejections: tuple[HistoryRejection, ...]


def argument(node: Node, position: int, name: str, default=None):
    """Read an operator argument whether supplied positionally or by keyword."""
    return node.args[position] if len(node.args) > position else node.kwargs.get(name, default)


def same_tensor_description(node, expected_shape, reference):
    tensor = get_tensor_metadata(node)
    if tensor is None or any(type(size) is not int for size in tensor.shape):
        return False
    return (tuple(tensor.shape) == expected_shape and tensor.dtype == reference.dtype
            and tensor.device == reference.device)


def match_stack_history(chain: RecurrenceChain, storage: Node) -> HistoryMatch | HistoryRejection:
    """Recognize one strict pattern; do not prove index bounds, effects, or profit.

    Inputs are fresh chains from form_chains over one unchanged straight-line
    graph. An index in [0, number of updates] remains a rewrite obligation.
    """
    def reject(reason):
        return HistoryRejection(chain, storage, reason)

    if not is_call(storage, torch.ops.aten.stack.default):
        return reject("storage is not aten.stack.default")
    dim = argument(storage, 1, "dim", 0)
    if type(dim) is not int or dim != 0:
        return reject("stack dimension must be exactly 0")
    expected_states = (chain.incoming_state,) + tuple(u.result for u in chain.updates)
    stored_states = argument(storage, 0, "tensors")
    if (not isinstance(stored_states, (list, tuple))
            or len(stored_states) != len(expected_states)
            or any(actual is not expected for actual, expected
                   in zip(stored_states, expected_states))):
        return reject("stack must contain initial state and every chain result in order")

    state_tensor = get_tensor_metadata(chain.incoming_state)
    if (state_tensor is None or state_tensor.ndim != 2
            or any(type(size) is not int for size in state_tensor.shape)):
        return reject("initial state needs concrete matrix metadata")
    state_shape = tuple(state_tensor.shape)
    if any(not same_tensor_description(state, state_shape, state_tensor)
           for state in expected_states):
        return reject("state metadata is missing or inconsistent")
    if not same_tensor_description(storage, (len(expected_states),) + state_shape, state_tensor):
        return reject("history metadata is missing or inconsistent")

    users = tuple(storage.users)
    if len(users) != 1:
        return reject("history must have exactly one direct consumer")
    selection = users[0]
    if not is_call(selection, torch.ops.aten.index_select.default):
        return reject("history consumer must be aten.index_select.default")
    if argument(selection, 0, "self") is not storage:
        return reject("history must be the selection's source tensor")
    dim = argument(selection, 1, "dim")
    if type(dim) is not int or dim != 0:
        return reject("selection dimension must be exactly 0")
    index = argument(selection, 2, "index")
    index_tensor = get_tensor_metadata(index)
    if (not isinstance(index, Node) or index_tensor is None
            or any(type(size) is not int for size in index_tensor.shape)
            or tuple(index_tensor.shape) != (1,) or index_tensor.dtype != torch.int64
            or index_tensor.device != state_tensor.device):
        return reject("index must be a shape-[1] int64 tensor on the state device")
    if not same_tensor_description(selection, (1,) + state_shape, state_tensor):
        return reject("selection metadata is missing or inconsistent")

    users = tuple(selection.users)
    if len(users) != 1:
        return reject("selection must have exactly one squeeze consumer")
    selected_state = users[0]
    if not is_call(selected_state, torch.ops.aten.squeeze.dim):
        return reject("selection consumer must be aten.squeeze.dim")
    if argument(selected_state, 0, "self") is not selection:
        return reject("squeeze must consume the selection")
    dim = argument(selected_state, 1, "dim")
    if type(dim) is not int or dim != 0:
        return reject("squeeze dimension must be exactly 0")
    if not same_tensor_description(selected_state, state_shape, state_tensor):
        return reject("retrieved-state metadata is missing or inconsistent")

    # Users of this final ordinary matrix are retained, not rewritten here.
    return HistoryMatch(chain, expected_states, storage, selection, index,
                        selected_state, (0, len(chain.updates)))


def find_stack_histories(chains: tuple[RecurrenceChain, ...]) -> HistoryAnalysis:
    matches = []
    rejections = []
    for chain in chains:
        candidates = []
        seen = set()
        # Check every state, so stacks missing the initial state are also diagnosed.
        states = (chain.incoming_state,) + tuple(u.result for u in chain.updates)
        for state in states:
            for user in state.users:
                if user not in seen and is_call(user, torch.ops.aten.stack.default):
                    seen.add(user)
                    candidates.append(user)
        if not candidates:
            rejections.append(HistoryRejection(chain, None, "no direct stack history candidate"))
        for storage in candidates:
            result = match_stack_history(chain, storage)
            if isinstance(result, HistoryMatch):
                matches.append(result)
            else:
                rejections.append(result)
    return HistoryAnalysis(tuple(matches), tuple(rejections))


def describe_histories(analysis: HistoryAnalysis) -> str:
    lines = []
    for i, match in enumerate(analysis.matches):
        lines.extend([
            f"HistoryMatch {i}:",
            f"  incoming_state: {match.chain.incoming_state.name}",
            "  updates: " + " -> ".join(u.result.name for u in match.chain.updates),
            "  stored_states: [" + ", ".join(n.name for n in match.states) + "]",
            f"  storage_node: {match.storage_node.name}",
            f"  selection_node: {match.selection_node.name}",
            f"  selected_index: {match.selected_index.name}",
            f"  selected_state: {match.selected_state.name}",
            f"  required_index_range: {match.required_index_range} (not proven here)",
        ])
    for rejection in analysis.rejections:
        storage = rejection.storage_node.name if rejection.storage_node is not None else "<none>"
        lines.append(f"Rejected history {storage} (incoming={rejection.chain.incoming_state.name}): "
                     + rejection.reason)
    lines.append(f"{len(analysis.matches)} matches; {len(analysis.rejections)} rejected candidates.")
    lines.append("Read-only pattern detection; no replay-legality or profitability claim.")
    return "\n".join(lines)


if __name__ == "__main__":
    from rewind_mvp.hybrid import TinyHybridWindow, example_inputs

    exported = torch.export.export(TinyHybridWindow(), example_inputs())
    graph = exported.graph_module.graph
    chains = form_chains(find_updates(graph))
    before = str(graph)
    print(describe_histories(find_stack_histories(chains.chains)))
    assert str(graph) == before, "History detection must not change the graph"
