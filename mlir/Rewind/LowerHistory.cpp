#include "Passes.h"
#include "Rewind.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::rewind;

namespace {

// These are references to SSA values, not copies of numerical tensor contents.
struct UpdateOperands {
  Value scale;
  Value value;
  Value key;
};

struct HistoryInfo {
  Value initial;
  SmallVector<UpdateOperands> updates;
};

using HistoryMap = llvm::DenseMap<Value, HistoryInfo>;

static bool containsStore(Type type) {
  bool found = false;
  type.walk([&](StoreType) { found = true; });
  return found;
}

// Run before changing ANY IR. This is the lowering's supported-input check,
// not a replacement for the future original-graph-to-Rewind legality analysis.
static LogicalResult validateInput(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      if (containsStore(function.getFunctionType())) {
        op->emitOpError("history handles cannot cross function boundaries");
        return WalkResult::interrupt();
      }
    }

    // Also reject handles hidden in tuple types or nested region arguments.
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument argument : block.getArguments()) {
          if (containsStore(argument.getType())) {
            op->emitOpError("history handles cannot be block arguments");
            return WalkResult::interrupt();
          }
        }
      }
    }

    if (op->getName().getDialectNamespace() == "rewind" &&
        !isa<InitializeStoreOp, AppendOp, RetrieveOp>(op)) {
      op->emitOpError("unsupported Rewind operation");
      return WalkResult::interrupt();
    }

    for (Value operand : op->getOperands()) {
      if (containsStore(operand.getType()) && !isa<AppendOp, RetrieveOp>(op)) {
        op->emitOpError("unsupported use of a history handle");
        return WalkResult::interrupt();
      }
    }
    for (Value value : op->getResults()) {
      if (containsStore(value.getType()) && !isa<InitializeStoreOp>(op)) {
        op->emitOpError("history must originate from rewind.initialize_store");
        return WalkResult::interrupt();
      }
    }

    if (auto initialize = dyn_cast<InitializeStoreOp>(op)) {
      auto function = dyn_cast<func::FuncOp>(op->getParentOp());
      if (!function || !function.getBody().hasOneBlock()) {
        op->emitOpError(
            "history must be local to a single-block function body");
        return WalkResult::interrupt();
      }
      for (Operation *user : initialize.getHistory().getUsers()) {
        if (!isa<AppendOp, RetrieveOp>(user) ||
            user->getBlock() != initialize->getBlock()) {
          user->emitOpError("history users must be append/retrieve in the "
                            "initialization block");
          return WalkResult::interrupt();
        }
      }
    }

    if (isa<AppendOp, RetrieveOp>(op)) {
      auto initialize = op->getOperand(0).getDefiningOp<InitializeStoreOp>();
      if (!initialize || initialize->getBlock() != op->getBlock()) {
        op->emitOpError("history must be initialized in the same block");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

// Called by the block walk. Record boundary zero; emit no runtime container.
static void lowerInitializeStore(InitializeStoreOp op, HistoryMap &histories) {
  histories.try_emplace(op.getHistory(), HistoryInfo{op.getInitial(), {}});
}

// Called by the block walk. Record this update without computing a new matrix.
// Earlier entries remain available for every subsequent retrieval.
static void lowerAppend(AppendOp op, HistoryMap &histories) {
  HistoryInfo &history = histories.find(op.getHistory())->second;
  history.updates.push_back({op.getScale(), op.getValue(), op.getKey()});
}

// Called inside a replay branch. Return the SSA result of ONE matrix update.
// tensor.generate describes all [d,k] elements without unrolling D*K in C++.
static Value emitReplayStep(OpBuilder &builder, Location loc, Value previous,
                            const UpdateOperands &update) {
  Value scale =
      tensor::ExtractOp::create(builder, loc, update.scale, ValueRange{});
  auto result = tensor::GenerateOp::create(
      builder, loc, previous.getType(), ValueRange{},
      [&](OpBuilder &body, Location bodyLoc, ValueRange indices) {
        Value state =
            tensor::ExtractOp::create(body, bodyLoc, previous, indices);
        Value value = tensor::ExtractOp::create(body, bodyLoc, update.value,
                                                ValueRange{indices[0]});
        Value key = tensor::ExtractOp::create(body, bodyLoc, update.key,
                                              ValueRange{indices[1]});

        // No fast-math flags: retain separate rounding for both multiplies
        // and the addition. Do not introduce FMA, contraction, or
        // reassociation.
        Value scaled = arith::MulFOp::create(body, bodyLoc, scale, state);
        Value product = arith::MulFOp::create(body, bodyLoc, value, key);
        Value next = arith::AddFOp::create(body, bodyLoc, scaled, product);
        tensor::YieldOp::create(body, bodyLoc, next);
      });
  return result.getResult();
}

// Called at the original retrieval position. Check its current bounds, replay
// exactly the selected prefix, and replace the matrix result's uses in place.
static void lowerRetrieve(RetrieveOp op, const HistoryInfo &history,
                          IRRewriter &rewriter) {
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  Value index = op.getIndex();
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value count = arith::ConstantIndexOp::create(
      rewriter, loc, static_cast<int64_t>(history.updates.size()));
  Value nonnegative = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sge, index, zero);
  Value withinHistory = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sle, index, count);
  Value valid =
      arith::AndIOp::create(rewriter, loc, nonnegative, withinHistory);
  cf::AssertOp::create(rewriter, loc, valid,
                       "rewind.retrieve index out of range");

  Value state = history.initial;
  for (auto entry : llvm::enumerate(history.updates)) {
    const UpdateOperands &update = entry.value();
    Value boundary = arith::ConstantIndexOp::create(
        rewriter, loc, static_cast<int64_t>(entry.index()));
    Value needed = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sgt, index, boundary);
    // The else branch passes through the previous tensor. In particular,
    // index zero performs no replay, and unselected suffixes do no arithmetic.
    auto step = scf::IfOp::create(
        rewriter, loc, needed,
        [&](OpBuilder &body, Location bodyLoc) {
          Value next = emitReplayStep(body, bodyLoc, state, update);
          scf::YieldOp::create(body, bodyLoc, next);
        },
        [&](OpBuilder &body, Location bodyLoc) {
          scf::YieldOp::create(body, bodyLoc, state);
        });
    state = step.getResult(0);
  }
  rewriter.replaceOp(op, state);
}

struct LowerHistoryPass
    : PassWrapper<LowerHistoryPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerHistoryPass)

  StringRef getArgument() const final { return "rewind-lower-history"; }
  StringRef getDescription() const final {
    return "Lower local straight-line Rewind histories to checked tensor "
           "replay";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(validateInput(module))) {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    SmallVector<Operation *> capturedOperations;
    module.walk([&](func::FuncOp function) {
      HistoryMap histories;
      for (Block &block : function.getBody()) {
        for (Operation &operation : llvm::make_early_inc_range(block)) {
          if (auto initialize = dyn_cast<InitializeStoreOp>(operation)) {
            lowerInitializeStore(initialize, histories);
            capturedOperations.push_back(initialize);
          } else if (auto append = dyn_cast<AppendOp>(operation)) {
            lowerAppend(append, histories);
            capturedOperations.push_back(append);
          } else if (auto retrieve = dyn_cast<RetrieveOp>(operation)) {
            lowerRetrieve(retrieve,
                          histories.find(retrieve.getHistory())->second,
                          rewriter);
          }
        }
      }
    });

    // Keep handles alive during the walk. Remove their users before
    // definitions.
    for (Operation *op : llvm::reverse(capturedOperations))
      rewriter.eraseOp(op);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::rewind::createLowerHistoryPass() {
  return std::make_unique<LowerHistoryPass>();
}

void mlir::rewind::registerRewindPasses() {
  PassRegistration<LowerHistoryPass>();
}
