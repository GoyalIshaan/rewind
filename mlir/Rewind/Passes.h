#ifndef REWIND_PASSES_H
#define REWIND_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace rewind {

// Lower local, straight-line histories to tensor, arith, scf, and cf
// operations.
std::unique_ptr<Pass> createLowerHistoryPass();
void registerRewindPasses();

} // namespace rewind
} // namespace mlir

#endif
