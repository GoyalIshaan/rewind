"""Read-only state-edge analysis over matched update expressions.

An edge A -> B means B.previous is A.result. No aliases or intermediate
operations are skipped. Results describe connectivity, not replay legality.
"""

from dataclasses import dataclass

from torch.fx import Node

from rewind_mvp.matcher import UpdateMatch, find_updates


@dataclass(frozen=True)
class RecurrenceChain:
    incoming_state: Node
    updates: tuple[UpdateMatch, ...]


@dataclass(frozen=True)
class StateFork:
    state: Node
    successors: tuple[UpdateMatch, ...]


@dataclass(frozen=True)
class BranchedComponent:
    incoming_state: Node
    updates: tuple[UpdateMatch, ...]
    forks: tuple[StateFork, ...]


@dataclass(frozen=True)
class ChainAnalysis:
    chains: tuple[RecurrenceChain, ...]
    skipped_components: tuple[BranchedComponent, ...]


def form_chains(matches: list[UpdateMatch]) -> ChainAnalysis:
    """Link matches in expected O(n) time and O(n) space using node identities.

    Input order need not be topological. Independent groups follow input root
    order; each linear chain follows its actual state edges. Single updates are
    retained as length-one candidates. Skip an entire connected component if it
    forks, rather than silently treating its arms as independent complete chains.
    Matches must refer to one valid, unchanged, straight-line FX graph.
    """
    producer_by_result: dict[Node, UpdateMatch] = {}
    graph = matches[0].result.graph if matches else None
    for match in matches:
        if match.result in producer_by_result:
            raise ValueError("Duplicate update result in match list")
        if any(node.graph is not graph for node in
               (match.result, match.previous, match.scale, match.value, match.key)):
            raise ValueError("All match nodes must belong to the same graph")
        producer_by_result[match.result] = match

    successors: dict[Node, list[UpdateMatch]] = {node: [] for node in producer_by_result}
    roots: list[UpdateMatch] = []
    for match in matches:
        if match.previous in producer_by_result:
            successors[match.previous].append(match)
        else:
            roots.append(match)

    chains: list[RecurrenceChain] = []
    skipped: list[BranchedComponent] = []
    visited: set[Node] = set()
    for root in roots:
        component: list[UpdateMatch] = []
        forks: list[StateFork] = []
        pending = [root]
        while pending:
            current = pending.pop()
            if current.result in visited:
                raise ValueError("State edges must form an acyclic forest")
            visited.add(current.result)
            component.append(current)
            children = successors[current.result]
            if len(children) > 1:
                forks.append(StateFork(current.result, tuple(children)))
            pending.extend(reversed(children))
        if forks:
            skipped.append(BranchedComponent(root.previous, tuple(component), tuple(forks)))
        else:
            chains.append(RecurrenceChain(root.previous, tuple(component)))

    if len(visited) != len(matches):
        raise ValueError("State-edge cycle or invalid match list has no reachable root")
    return ChainAnalysis(tuple(chains), tuple(skipped))


def describe_chains(analysis: ChainAnalysis) -> str:
    lines = []
    for i, chain in enumerate(analysis.chains):
        path = " -> ".join([chain.incoming_state.name]
                           + [update.result.name for update in chain.updates])
        lines.append(f"Chain {i}: {path} ({len(chain.updates)} updates)")
    for i, component in enumerate(analysis.skipped_components):
        lines.append(f"Skipped component {i}: incoming={component.incoming_state.name}, "
                     f"{len(component.updates)} updates; state fork unsupported")
        for fork in component.forks:
            children = ", ".join(update.result.name for update in fork.successors)
            lines.append(f"  {fork.state.name} -> [{children}]")
    lines.append("Connectivity analysis only; graph unchanged, no replay rewrite.")
    return "\n".join(lines)


if __name__ == "__main__":
    import torch

    from rewind_mvp.hybrid import TinyHybridWindow, example_inputs

    exported = torch.export.export(TinyHybridWindow(), example_inputs())
    print(describe_chains(form_chains(find_updates(exported.graph_module.graph))))
