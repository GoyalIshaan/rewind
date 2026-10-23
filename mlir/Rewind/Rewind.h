#ifndef REWIND_IR_REWIND_H
#define REWIND_IR_REWIND_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Rewind/RewindDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Rewind/RewindTypes.h.inc"

#define GET_OP_CLASSES
#include "Rewind/RewindOps.h.inc"

#endif
