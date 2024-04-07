#ifndef MLIR_DIALECT_ELEMENTWISE_PASSES_H_
#define MLIR_DIALECT_ELEMENTWISE_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace ew {

#define GEN_PASS_DECL_EWTILINGPASS

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Elementwise/Passes.h.inc"
} // namespace ew
} // namespace mlir

#endif // MLIR_DIALECT_ELEMENTWISE_PASSES_H_
