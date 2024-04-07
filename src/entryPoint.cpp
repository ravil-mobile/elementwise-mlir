#include "entryPoint.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Elementwise/Pipelines/Pipelines.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/InitPasses.h"

using namespace mlir;

LogicalResult runPipeline(ModuleOp module) {
  PassManager pm(module->getName(), PassManager::Nesting::Implicit);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs()
        << "failed to apply command options of the mlir pass manager\n";
    return failure();
  }
  mlir::ew::registerElementwisePasses();
  mlir::ew::buildTopLevelPipeline(pm);

  if (failed(pm.run(module))) {
    llvm::errs() << "failed to perform mlir optimization\n";
    return failure();
  }

  return success();
}

namespace ew {
bool main(ModuleOp module, const Options &options) {
  auto status = runPipeline(module);
  if (failed(status)) {
    llvm::errs() << "pass pipeline failed\n";
    return false;
  }
  module->dump();

  return true;
}
} // namespace ew
