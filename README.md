# Rewind

A learning and research project exploring compiler analysis for recurrent-state
history and checkpoint/replay during speculative inference.

## Current status

The current MVP exports a small synthetic hybrid PyTorch computation containing
recurrent updates, causal attention, verification logits, and greedy accepted-prefix
selection. Read-only analyses recognize:

1. Updates of the form `a * state + outer(v, k)`.
2. Linear state-dependency chains, reporting unsupported forks.
3. Ordered state-history stacks and their `index_select` / `squeeze` retrieval.

Matches include graph-node references and are printed for debugging. Recognition
does not yet prove replay legality or perform a history rewrite. There is no MLIR
dialect, generated optimized GPU kernel, or measured speedup yet.

The synthetic fixture is not a pretrained model or complete speculative serving
engine. `reference.py` includes a hand-written replay reference for correctness
experiments; that is separate from an automatic compiler rewrite.

## Run

Use a Python environment supported by the pinned PyTorch version, preferably a
virtual environment. From the repository root:

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s rewind_mvp -v
python3 -m rewind_mvp.matcher
python3 -m rewind_mvp.chains
python3 -m rewind_mvp.history
```

Export the fixture's graph:

```sh
python3 -m rewind_mvp.export_hybrid
```

This writes `rewind_mvp/hybrid_graph.txt`. The fixed-size Python loop currently
exports as an unrolled ATen graph. Node names in reports are labels; analyses use
actual dependencies and node identity.

## Project files

| File | Purpose |
| --- | --- |
| `rewind_mvp/hybrid.py` | Synthetic hybrid verification fixture |
| `rewind_mvp/program.py` | Earlier minimal recurrence-window function |
| `rewind_mvp/reference.py` | Executable oracle and manual replay reference |
| `rewind_mvp/matcher.py` | Exact update-expression recognition |
| `rewind_mvp/chains.py` | State-edge connectivity analysis |
| `rewind_mvp/history.py` | Stack-history and retrieval detection |
| `rewind_mvp/test_*.py` | Positive, negative, and non-mutation checks |
| `output/` | Existing inference study worksheets and solutions |
| `work/` | Existing traces from an earlier MoE-cache experiment |

The worksheets and MoE traces predate this compiler MVP. They are supporting
workspace artifacts, not Rewind benchmark results.

## Development

GitHub Actions runs the test suite and analysis demos on pushes and pull requests.
Changes are committed and pushed as coherent, validated steps. The initial commit
is a snapshot of existing work; earlier implementation dates are not reconstructed
as artificial commits. Local concepts/ideas tracking notes are excluded from Git.
