#ifndef REWIND_IR_OPS_TD
#define REWIND_IR_OPS_TD

include "Rewind/IR/RewindTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Rewind_Op<string mnemonic, list<Trait> traits = []>
    : Op<Rewind_Dialect, mnemonic, traits>;

def Rewind_InitializeStoreOp : Rewind_Op<"initialize_store"> {
  let summary = "Capture the initial matrix in a fresh history";
  let description = [{
    Captures the initial tensor value as boundary zero. The returned handle
    refers to a fresh empty update log. The initial tensor type must equal the
    store's stateType. Future lowering must preserve the captured value and
    manage the store's lifetime; no allocation implementation exists yet.
  }];
  let arguments = (ins AnyRankedTensor:$initial);
  let results = (outs Res<Rewind_StoreType, "fresh history",
                         [MemAlloc, MemWrite]>:$history);
  let assemblyFormat = "$initial attr-dict `:` type($initial) `->` qualified(type($history))";
  let hasVerifier = 1;
}

def Rewind_AppendOp : Rewind_Op<"append"> {
  let summary = "Capture one update's operands without computing its matrix";
  let description = [{
    Appends the values (scale, value, key) in program order to history. For a
    state tensor<DxKxF>, operands have types tensor<F>, tensor<DxF>, tensor<KxF>.
    All elements have the same floating-point type F; no implicit casts occur.
    Earlier entries remain unchanged. There is no result: the handle continues
    to refer to the same mutable store. This does not replace the forward
    recurrence that computes verification outputs.
  }];
  let arguments = (ins
    Arg<Rewind_StoreType, "history being extended", [MemRead, MemWrite]>:$history,
    AnyRankedTensor:$scale,
    AnyRankedTensor:$value,
    AnyRankedTensor:$key
  );
  let assemblyFormat = [{
    $history `,` $scale `,` $value `,` $key attr-dict `:`
    qualified(type($history)) `,` type($scale) `,` type($value) `,` type($key)
  }];
  let hasVerifier = 1;
}

// Conservative global write models observable bounds failure. It does NOT mean
// retrieval mutates the history. Until failure effects are modeled separately,
// this intentionally restricts generic elimination/reordering of retrievals.
def Rewind_RetrieveOp : Rewind_Op<"retrieve", [MemoryEffects<[MemWrite]>]> {
  let summary = "Reconstruct the matrix at an existing history boundary";
  let description = [{
    Reads history and returns S_index. Zero returns the initial matrix; after
    n appends, n returns the matrix after all n updates. Requires a runtime
    failure if index < 0 or index > n, not undefined behavior or clamping.
    The verifier only checks types; it does not prove dynamic bounds.
    Retrieval leaves the store unchanged. Replay must preserve separate scale,
    outer-product and addition rounding, with no reassociation or contraction.
    These are semantic requirements for future lowering, not execution here.
  }];
  let arguments = (ins
    Arg<Rewind_StoreType, "history to read", [MemRead]>:$history,
    Index:$index
  );
  let results = (outs AnyRankedTensor:$state);
  let assemblyFormat = "$history `[` $index `]` attr-dict `:` qualified(type($history)) `->` type($state)";
  let hasVerifier = 1;
}

#endif
