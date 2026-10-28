#include "Rewind.h"

using namespace mlir;

#include "RewindDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "RewindTypes.cpp.inc"

#define GET_OP_CLASSES
#include "RewindOps.cpp.inc"

namespace mlir::rewind {

// Called when the MLIR context loads this dialect. Register generated classes.
void RewindDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "RewindTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "RewindOps.cpp.inc"
      >();
}

// Called when checking a store type, including when parsing textual MLIR.
LogicalResult StoreType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type stateType) {
  auto matrix = dyn_cast<RankedTensorType>(stateType);
  if (!matrix || matrix.getRank() != 2 || !matrix.hasStaticShape() ||
      matrix.getEncoding() || !isa<FloatType>(matrix.getElementType()))
    return emitError() << "store requires an unencoded, static rank-2 "
                          "floating-point tensor";
  return success();
}

// MLIR runs generated operand constraints before these cross-operand checks.
LogicalResult InitializeStoreOp::verify() {
  if (getInitial().getType() != getHistory().getType().getStateType())
    return emitOpError("initial tensor must match the store's state type");
  return success();
}

LogicalResult AppendOp::verify() {
  auto state = cast<RankedTensorType>(getHistory().getType().getStateType());
  auto scale = getScale().getType();
  auto value = getValue().getType();
  auto key = getKey().getType();

  for (RankedTensorType operand : {scale, value, key}) {
    if (!operand.hasStaticShape() || operand.getEncoding())
      return emitOpError(
          "update operands must have static shapes and no encoding");
    if (operand.getElementType() != state.getElementType())
      return emitOpError(
          "update operands must share the store's floating element type");
  }
  if (scale.getRank() != 0)
    return emitOpError("scale must be a rank-0 tensor");
  if (value.getRank() != 1 || value.getDimSize(0) != state.getDimSize(0))
    return emitOpError("value must be a vector matching state dimension 0");
  if (key.getRank() != 1 || key.getDimSize(0) != state.getDimSize(1))
    return emitOpError("key must be a vector matching state dimension 1");
  return success();
}

LogicalResult RetrieveOp::verify() {
  if (getState().getType() != getHistory().getType().getStateType())
    return emitOpError("result tensor must match the store's state type");
  return success();
}

} // namespace mlir::rewind
