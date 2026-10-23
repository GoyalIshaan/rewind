#ifndef REWIND_IR_DIALECT_TD
#define REWIND_IR_DIALECT_TD

include "mlir/IR/OpBase.td"

def Rewind_Dialect : Dialect {
  let name = "rewind";
  let cppNamespace = "::mlir::rewind";
  let summary = "Append-only recurrent-state replay histories";
  let description = [{
    A history captures S_0 and an ordered sequence of (a, v, k) tensor values.
    Boundary j denotes S_j = a_j * S_(j-1) + outer(v_j, k_j).
    Logging an update does not execute the recurrence. Retrieval replays it.
    This initial dialect defines representation, not a PyTorch rewrite or a
    storage placement policy.
  }];
  let useDefaultTypePrinterParser = 1;
}

#endif
