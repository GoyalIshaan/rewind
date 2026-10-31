# Rewind MLIR dialect and history lowering

The dialect supports parsing, printing, type checking, and conservative effects.
`--rewind-lower-history` lowers local straight-line history operations to checked
replay using the standard `tensor`, `arith`, `scf`, and `cf` dialects. The tests
run the result on the CPU through the installed MLIR tools. PyTorch graph import,
automatic snapshot-history rewriting, CUDA lowering, and performance measurement
remain separate unfinished steps.

## Build and check

Requires an LLVM/MLIR development installation, CMake, a C++17 compiler, and
Python 3 for the test runner. Initially validated with LLVM/MLIR 22.1.5.
The Python analysis MVP does not require this build.

From the repository root, substitute your installation's MLIR package path:

```sh
cmake -S mlir -B build/mlir \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
# CMake also writes build/mlir/compile_commands.json for clangd / the editor.
cmake --build build/mlir -j 4
ctest --test-dir build/mlir --output-on-failure
build/mlir/rewind-opt mlir/test/history.mlir
build/mlir/rewind-opt mlir/test/history.mlir --rewind-lower-history --verify-each
```

On this Apple Silicon workstation the successful configuration uses:

```sh
cmake -S mlir -B build/mlir \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DCMAKE_AR=/usr/bin/ar \
  -DCMAKE_RANLIB=/usr/bin/ranlib \
  -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  -DCMAKE_BUILD_TYPE=Release
```

The explicit Xcode SDK avoids the local default Apple compiler/Command Line
Tools SDK mismatch. The Apple archive tools avoid picking up GNU binutils from
PATH when creating Mach-O libraries. These paths are machine-specific, not
project requirements.

`CMAKE_EXPORT_COMPILE_COMMANDS` is a cache option, so CMake writes
`build/mlir/compile_commands.json`. The build also copies that file into
`mlir/compile_commands.json` (gitignored) because Cursor's Microsoft C++
extension only reads a database inside the workspace. `mlir/.clangd` and
`mlir/.vscode/` point at that copy. After a build, reload the window
if includes still look unresolved.

## Read the code in this order

The handwritten dialect files live together in `Rewind/`.

1. `Rewind/RewindDialect.td`: declares the `rewind` namespace.
2. `Rewind/RewindTypes.td`: declares the store handle type and its
   matrix-type parameter.
3. `Rewind/RewindOps.td`: declares operation inputs, outputs, assembly
   syntax, semantic contracts, and effects.
4. `Rewind/RewindDialect.cpp`: registers generated classes and supplies verifiers
   for relationships between types and shapes.
5. `tools/rewind-opt.cpp`: registers the dialect with a small MLIR driver.
6. `test/history.mlir`: a handwritten program using the operations.
7. `Rewind/LowerHistory.cpp`: supported-input checks, three lowering helpers,
   replay arithmetic, and the ordered pass driver.

The build runs `mlir-tblgen` on `.td` files to generate C++ `.inc` files under
`build/mlir/include/Rewind/`. The handwritten header and implementation include
these generated declarations and definitions. Generated files are not committed.
`RewindIR` exports include paths for both `Rewind/` and the generated
`include/Rewind/` directory, so grouped sources can use short includes such as
`Rewind.h` and `RewindTypes.h.inc`. `Rewind.h` collects the external MLIR/LLVM
headers through `Rewind/System.h`.
The definitions use MLIR's
[operation definitions](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
and [type definitions](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/).

## V0 contract

For `!rewind.store<tensor<DxKxF>>`, `D` and `K` are static and `F` is a floating
element type. The state and update tensors have no encoding. The handle itself
is not a tensor; its parameter describes the matrix retrieval returns.

| Operation | Inputs | Outputs | Meaning |
| --- | --- | --- | --- |
| `initialize_store` | Initial `tensor<DxKxF>` | Fresh store handle | Capture the initial value as boundary zero |
| `append` | Store, `tensor<F>` scale, `tensor<DxF>` value, `tensor<KxF>` key | None | Capture one update's operands in order |
| `retrieve` | Store, scalar `index` | `tensor<DxKxF>` | Reconstruct the requested boundary |

All operands use the same floating element type; there are no implicit casts.
The rank-zero tensor scale matches the existing ATen matcher. The MLIR `index`
operand is different from PyTorch's shape-[1] int64 index tensor; a future
importer must explicitly bridge that representation.

After `n` appends, the existing boundaries are `0..n`. Retrieval of zero returns
the initial value. For each replayed step, separately compute:

```text
scaled[d,k]  = round_F(a * previous[d,k])
product[d,k] = round_F(v[d] * k[k])
next[d,k]    = round_F(scaled[d,k] + product[d,k])
```

No contraction, reassociation, or implicit type promotion is permitted by this
initial contract. Numerical equivalence to an eventual execution backend still
needs testing, including that backend's floating-point environment.

Initialization and append capture tensor values, not externally mutable buffer
references. Later appends preserve all earlier entries. Appending does not
compute a state matrix: the original forward recurrence and its output users
remain separate from this history log.

An out-of-range index causes a runtime failure after lowering. Type verification
alone still accepts a constant invalid index: the lowering inserts signed lower
and upper bounds checks and a `cf.assert` at each original retrieval position.
It does not assume valid bounds or silently clamp an index. Even an unused
retrieval must preserve this failure.

## Lowering scope and function walkthrough

The first pass supports stores initialized directly in a single-block function
body. Every handle use must be an append or retrieval in that same block.
It supports any statically represented number of appends, runtime indices,
multiple retrievals, and independent stores, including stores sharing an initial
tensor. Branch-spanning histories, nested history operations, and handles passed
to or returned from functions are rejected. Rejection happens before any IR is
changed anywhere in the module.

This is the supported-input contract for lowering. A future original-graph-to-
Rewind transformation must separately prove its semantic legality and retain the
original computation when that proof fails. The lowering does not recover a
PyTorch graph after unsupported Rewind operations have been introduced.

`HistoryMap` maps each handle to a `HistoryInfo`: an initial SSA value and a
`SmallVector` of `(scale, value, key)` SSA references. This is compiler memory,
not a runtime tensor container. Tensor values must still remain available at
runtime; ordinary tensor bufferization manages their physical storage.

The call flow in `Rewind/LowerHistory.cpp` is:

1. `validateInput(module)` checks that all histories fit the supported scope.
2. `runOnOperation()` walks each function's operations in program order.
3. `lowerInitializeStore(op, histories)` records the initial value and an empty
   update sequence for the handle. It emits no numerical computation.
4. `lowerAppend(op, histories)` appends operand references to that handle's
   sequence. It does not calculate the next state or remove earlier entries.
5. `lowerRetrieve(op, history, rewriter)` checks bounds against the updates
   encountered so far, emits replay, and replaces uses of the retrieved matrix.
6. `emitReplayStep(builder, loc, previous, update)` emits one `tensor.generate`.
   Its body extracts the required elements and emits two `arith.mulf` operations
   followed by `arith.addf`, without fast-math flags.
7. The pass erases the original appends and initializations in reverse order,
   after their handle users have been removed.

The ordered walk performs the bookkeeping we discussed for `collectHistory`
without rescanning the block for every retrieval. It also reads later operands
after any earlier retrieval-result replacements, so a retrieved matrix can
safely initialize another store. Future appends cannot affect earlier bounds.

For two recorded updates, retrieval generates computation equivalent to:

```text
assert 0 <= accepted <= 2
state = S0
if accepted > 0:
    state = replay_step(state, update0)
if accepted > 1:
    state = replay_step(state, update1)
return state
```

The C++ pass emits these branches during compilation. The comparisons, tensor
elements, and replay arithmetic execute later. Only the selected prefix performs
arithmetic. Each retrieval starts from its checkpoint; records are not consumed.
The pass does not modify other forward computations or matrix consumers.

This emits one conditional replay site per available update per retrieval.
It is a simple first implementation, not a profitability decision. It does not
unroll matrix dimensions into C++ statements: `tensor.generate` represents the
element space. Downstream bufferization and loop lowering create the actual
allocations and loops, with their own costs.

## Run a complete CPU example

`examples/replay.mlir` uses two updates on a rectangular matrix. Its `main`
returns element `[0,0]` after both updates, which is `0.5`.

After building, run from the repository root (the paths below match this Mac):

```sh
build/mlir/rewind-opt mlir/examples/replay.mlir \
  --rewind-lower-history --verify-each -o build/mlir/replay.tensor.mlir
/opt/homebrew/opt/llvm/bin/mlir-opt build/mlir/replay.tensor.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref --convert-linalg-to-loops \
  --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
  --finalize-memref-to-llvm --convert-func-to-llvm \
  --reconcile-unrealized-casts -o build/mlir/replay.llvm.mlir
/opt/homebrew/opt/llvm/bin/mlir-runner build/mlir/replay.llvm.mlir \
  -e main --entry-point-result=f32 \
  --shared-libs=/opt/homebrew/opt/llvm/lib/libmlir_c_runner_utils.dylib
```

The runner prints `5.000000e-01`. The intermediate files are under the ignored
build directory. The C runner utility library supplies the `memrefCopy` routine
used by downstream bufferization; this is not a Rewind history runtime.

`test/test_lowering.py` checks structure and unsupported-input diagnostics. When
`mlir-opt`, `mlir-runner` (or `mlir-cpu-runner`), and `mlir_c_runner_utils` are
available, CTest also runs the execution cases; otherwise those cases are marked
skipped. Numerical execution is checked for `f32` and `f64`, including every
boundary of zero-, one-, and three-update histories, repeated and interleaved
retrievals, independent stores, negative/too-large indices, immutable checkpoint
capture, and the ordering of bounds failure before a consumer. Every returned
element is compared bitwise with a reference that rounds each operation
separately; NaNs are checked by classification. Adversarial cancellation cases
also run with canonicalization, CSE, and LLVM optimization to detect contraction.
The lowering's structural tests also cover `f16` and `bf16`; those two types are
not covered by the CPU numerical execution tests.

## Effects and unfinished work

Initialization allocates/writes a fresh history; append reads/writes its history;
retrieval reads its history. Retrieval also declares a conservative global write
effect to preserve its possible observable bounds failure during generic
elimination/reordering. That effect is not a mutation of the history. None of
these operations is marked `Pure` or always speculatable.

This is deliberately conservative. Precise failure resources, store alias
analysis, and movement past appends that preserve the requested prefix are not
implemented. Describing append-only semantics does not automatically teach
generic MLIR passes that particular optimization.

The dialect still has no general runtime container, physical capacity, device
policy, destruction operation, or ownership model for escaping handles. The
current lowering eliminates local handles and delegates tensor allocation and
deallocation to standard MLIR bufferization. General histories need additional
representation and lifetime design. The dialect tests exercise parser/printer
round trips, accepted/rejected types, and effect preservation; the lowering tests
separately exercise generated computation and runtime bounds failures.

GitHub Actions currently runs the Python analysis suite. The dialect build and
CTest suite are separate local checks; LLVM installation in CI is a later step.
