"""Read-only recognition of one supported state-update expression in ATen/FX."""

from dataclasses import dataclass
from typing import Any, Callable, TypeGuard

import torch
from torch.fx import Graph, Node


@dataclass(frozen=True)
class UpdateMatch:
    result: Node
    previous: Node
    scale: Node
    value: Node
    key: Node


def is_call(node: object, target: Callable[..., Any]) -> TypeGuard[Node]:
    return (isinstance(node, Node) and node.op == "call_function"
            and node.target == target)


def get_tensor_metadata(node: object) -> torch.Tensor | None:
    # Export attaches tensor metadata (usually FakeTensor); no tensor executes here.
    value: object = node.meta.get("val") if isinstance(node, Node) else None
    return value if isinstance(value, torch.Tensor) else None


def match_state_update(node: Node) -> UpdateMatch | None:
    """Match add(mul(scale, previous), outer(value, key)), with static shapes.

    This is deliberately one spelling, with positional tensor operands. A match
    is an expression candidate, not a recurrence/alias/effect/rewriting proof.
    """

    # is this an addition node? and does it have 2 args?
    if not is_call(node, torch.ops.aten.add.Tensor) or len(node.args) != 2: return None
    # makse sure that the additional isnt scaled additon
    if node.kwargs.get("alpha", 1) != 1: return None

    scaled_arg, product_arg = node.args
    if not isinstance(scaled_arg, Node) or not isinstance(product_arg, Node): return None
    scaled, product = scaled_arg, product_arg
    if not is_call(scaled, torch.ops.aten.mul.Tensor) or len(scaled.args) != 2: return None
    if not is_call(product, torch.ops.aten.outer.default) or len(product.args) != 2: return None

    scale_factor, previous_state = scaled.args
    value_tensor, key_tensor = product.args
    if (not isinstance(scale_factor, Node) or not isinstance(previous_state, Node)
            or not isinstance(value_tensor, Node) or not isinstance(key_tensor, Node)):
        return None

    scale, previous = scale_factor, previous_state
    value, key = value_tensor, key_tensor

    result_t = get_tensor_metadata(node)
    scaled_t = get_tensor_metadata(scaled)
    product_t = get_tensor_metadata(product)
    previous_t = get_tensor_metadata(previous)
    scale_t = get_tensor_metadata(scale)
    value_t = get_tensor_metadata(value)
    key_t = get_tensor_metadata(key)
    if (result_t is None or scaled_t is None or product_t is None
            or previous_t is None or scale_t is None or value_t is None or key_t is None):
        return None
    operands = (result_t, scaled_t, product_t, previous_t, scale_t, value_t, key_t)
    # V0 handles concrete shapes and one common floating dtype/device only.
    if any(type(size) is not int for tensor in operands for size in tensor.shape):
        return None
    result_t, scaled_t, product_t, previous_t, scale_t, value_t, key_t = operands
    if any(not tensor.is_floating_point() or tensor.dtype != previous_t.dtype
           or tensor.device != previous_t.device for tensor in operands):
        return None
    if previous_t.ndim != 2 or scale_t.ndim != 0:
        return None
    if value_t.ndim != 1 or key_t.ndim != 1:
        return None
    expected = (value_t.shape[0], key_t.shape[0])
    if any(tuple(tensor.shape) != expected
           for tensor in (previous_t, scaled_t, product_t, result_t)):
        return None

    return UpdateMatch(node, previous, scale, value, key)


def find_updates(graph: Graph) -> list[UpdateMatch]:
    matches: list[UpdateMatch] = []
    node: Node
    for node in graph.nodes:
        match: UpdateMatch | None = match_state_update(node)
        if match is not None:
            matches.append(match)
    return matches


if __name__ == "__main__":
    from rewind_mvp.hybrid import TinyHybridWindow, example_inputs

    exported: torch.export.ExportedProgram = torch.export.export(
        TinyHybridWindow(), example_inputs()
    )
    graph: Graph = exported.graph_module.graph
    before: str = str(graph)
    matches: list[UpdateMatch] = find_updates(graph)
    match: UpdateMatch
    for match in matches:
        print(f"{match.result.name} = {match.scale.name} * {match.previous.name}"
              f" + outer({match.value.name}, {match.key.name})")
    assert str(graph) == before, "Recognition must not change the graph"
    print(f"{len(matches)} update candidates; graph unchanged. No rewrite performed.")
