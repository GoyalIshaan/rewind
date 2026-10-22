#ifndef REWIND_IR_TYPES_TD
#define REWIND_IR_TYPES_TD

include "Rewind/IR/RewindDialect.td"
include "mlir/IR/AttrTypeBase.td"

def Rewind_StoreType : TypeDef<Rewind_Dialect, "Store"> {
  let mnemonic = "store";
  let summary = "Handle to a mutable, append-only history of a matrix";
  let description = [{
    The parameter is the reconstructed matrix type, not the handle's physical
    layout. V0 requires an unencoded, statically shaped rank-two tensor with
    floating-point elements. Distinct initialize_store calls create distinct
    stores even when their types and initial values are equal.
    No capacity, device, HBM allocation, or SRAM-fit claim is encoded here.
  }];
  let parameters = (ins "::mlir::Type":$stateType);
  let assemblyFormat = "`<` $stateType `>`";
  let genVerifyDecl = 1;
}

#endif
