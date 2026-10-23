# Rewind MLIR skeleton

This is a compiled dialect definition, with parsing, printing, type checking,
and conservative effects. It does not execute replay, import the PyTorch graph,
rewrite history storage, lower to CUDA, or establish a speedup yet.

## Build and check

Requires an LLVM/MLIR development installation, CMake, a C++17 compiler, and
Python 3 for the test runner. Initially validated with LLVM/MLIR 22.1.5.
The Python analysis MVP does not require this build.

From the repository root, substitute your installation's MLIR package path:

```sh
cmake -S mlir -B build/mlir \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/mlir -j 4
ctest --test-dir build/mlir --output-on-failure
build/mlir/rewind-opt mlir/test/history.mlir
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

The build runs `mlir-tblgen` on `.td` files to generate C++ `.inc` files under
`build/mlir/include/Rewind/`. The handwritten header and implementation include
these generated declarations and definitions. Generated files are not committed.
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

An out-of-range index must cause a runtime failure. The current tool only
parses and verifies IR: even a constant invalid index can pass type verification.
There is no runtime bounds check implementation yet; future executable lowering
must implement it rather than assuming valid bounds or silently clamping.

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

The store currently has no physical capacity, device, allocation strategy,
destruction operation, ownership restrictions, or lowering. Lifetime management
must be resolved before generating an executable implementation. The tests
exercise parser/printer round trips, accepted and rejected type relationships,
and preservation of effectful operations under canonicalization/CSE. They do
not test numerical replay or bounds execution.

GitHub Actions currently runs the Python analysis suite. The dialect build and
CTest suite are separate local checks; LLVM installation in CI is a later step.
