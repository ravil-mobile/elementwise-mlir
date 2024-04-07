#ifndef MLIR_DIALECT_EW_PIPELINES_H_
#define MLIR_DIALECT_EW_PIPELINES_H_

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace ew {
void buildTopLevelPipeline(mlir::OpPassManager &pm);
} // namespace ew
} // namespace mlir

#endif // MLIR_DIALECT_EW_PIPELINES_H_
