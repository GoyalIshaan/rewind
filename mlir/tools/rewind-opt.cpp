#include "Rewind/Passes.h"
#include "Rewind/Rewind.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::rewind::RewindDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  mlir::registerTransformsPasses();
  mlir::rewind::registerRewindPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Rewind compiler\n", registry));
}
