#include "mlir/Dialect/Elementwise/Pipelines/Pipelines.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Elementwise/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::ew::buildTopLevelPipeline(OpPassManager &pm) {
  auto &funcPm = pm.nest<func::FuncOp>();
  funcPm.addPass(mlir::createCanonicalizerPass());
  funcPm.addPass(mlir::createCSEPass());
  funcPm.addPass(mlir::ew::createEwTilingPass());
}