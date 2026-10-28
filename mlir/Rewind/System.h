#ifndef REWIND_IR_SYSTEM_H
#define REWIND_IR_SYSTEM_H

// Quoted MLIR/LLVM includes need the compiler's -isystem path. The editor
// often has none, so fall back to this machine's Homebrew LLVM prefix.
#if __has_include(<mlir/IR/Builders.h>)
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <llvm/ADT/TypeSwitch.h>
#else
#include </opt/homebrew/opt/llvm/include/mlir/IR/Builders.h>
#include </opt/homebrew/opt/llvm/include/mlir/IR/BuiltinTypes.h>
#include </opt/homebrew/opt/llvm/include/mlir/IR/Dialect.h>
#include </opt/homebrew/opt/llvm/include/mlir/IR/DialectImplementation.h>
#include </opt/homebrew/opt/llvm/include/mlir/IR/OpDefinition.h>
#include </opt/homebrew/opt/llvm/include/mlir/Interfaces/SideEffectInterfaces.h>
#include </opt/homebrew/opt/llvm/include/llvm/ADT/TypeSwitch.h>
#endif

#endif
