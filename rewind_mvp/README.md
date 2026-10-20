# Rewind: first recurrent-layer fixture

This is the executable specification to build a compiler against. It is separate
from the earlier MoE cache experiment. There is no compiler or dialect yet.

For each token:

    S_next = a * S + outer(v, k)
    y = S_next @ q

`S` has shape `[V, K]`, `a` is a scalar decay, `v` has shape `[V]`, and `k`
and `q` have shape `[K]`. Inputs are already projected recurrence inputs;
this is a synthetic operator, not a complete Mamba or Gated DeltaNet layer.

Run from the repository root, with the existing PyTorch environment:

```sh
python3 -m rewind_mvp.reference
python3 -m unittest rewind_mvp.test_reference -v
```

The oracle saves all state boundaries. The replay reference restores an accepted
prefix from the initial checkpoint and retained recurrence inputs. Checks cover
zero accepted tokens, full acceptance, every intermediate boundary, and a new
continuation after rejection. Exact equality is scoped to these CPU float64
fixtures and this unchanged arithmetic order.

## Next compiler slice

1. Express the update using typed SSA operations: scalar/tensor multiply, outer
   product, add, and matrix-vector product, inside an explicit sequence loop.
2. Interpret that IR and compare every output and state boundary to `reference.py`.
3. Recognize precisely this affine update, including shapes, state dependencies,
   and loop-invariant versus per-token operands. Unmatched patterns stay unchanged.
4. Define a speculative-window operation whose observations are all verification
   outputs plus the selected accepted state. The accepted length is a runtime input.
5. Rewrite snapshot storage into checkpoint plus retained recurrence inputs,
   followed by replay of the accepted prefix. Preserve outputs and update order.

Separate matching, legality checking, rewriting, and later cost-based selection.
Do not rewrite arbitrary Python or infer legality from a few matching random
outputs. Log pass decisions and dump IR before and after each pass.

State history is unnecessary only when it is not observed. Verification outputs
still require computation. A replay plan trades state storage against retained
inputs and recomputation; speed and memory savings must be measured, not assumed.
Neither in-place buffer reuse nor reassociation is authorized by this first pass.
