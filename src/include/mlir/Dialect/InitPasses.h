#ifndef MLIR_INITPASSES_H_
#define MLIR_INITPASSES_H_

#include "mlir/Dialect/Elementwise/Passes.h"
#include "mlir/Dialect/Elementwise/Pipelines/Pipelines.h"

namespace mlir {
namespace ew {
inline void registerElementwisePasses() {
  PassPipelineRegistration<>("argument", "description", buildTopLevelPipeline);
}
} // namespace ew
} // namespace mlir

#endif // #ifndef MLIR_INITPASSES_H_