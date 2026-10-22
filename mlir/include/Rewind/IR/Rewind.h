#ifndef REWIND_IR_REWIND_H
#define REWIND_IR_REWIND_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Rewind/IR/RewindDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Rewind/IR/RewindTypes.h.inc"

#define GET_OP_CLASSES
#include "Rewind/IR/RewindOps.h.inc"

#endif
