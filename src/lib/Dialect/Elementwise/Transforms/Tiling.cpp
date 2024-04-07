#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir {
namespace ew {
#define GEN_PASS_DEF_EWTILINGPASS
#include "mlir/Dialect/Elementwise/Passes.h.inc"
} // namespace ew
} // namespace mlir

#define DEBUG_TYPE "ew-tiling"

using namespace mlir;

struct EwTilingPass : public ew::impl::EwTilingPassBase<EwTilingPass> {
  void runOnOperation() override;
};

LogicalResult tile(linalg::GenericOp laGeneric) {
  IRRewriter rewriter(laGeneric.getContext());
  linalg::LinalgTilingOptions tileOption;
  tileOption.setTileSizes(llvm::SmallVector<int64_t>{4});
  FailureOr<linalg::TiledLinalgOp> tiledOps =
      linalg::tileLinalgOp(rewriter, laGeneric, tileOption);
  if (failed(tiledOps))
    return failure();
  tiledOps->op->dump();
  rewriter.replaceOp(laGeneric, tiledOps->tensorResults);
  return success();
}

void EwTilingPass::runOnOperation() {
  func::FuncOp func = getOperation();
  SmallVector<linalg::GenericOp> laGenericOps;
  func->walk(
      [&laGenericOps](linalg::GenericOp op) { laGenericOps.push_back(op); });

  for (auto op : laGenericOps) {
    if (failed(tile(op))) {
      signalPassFailure();
    }
  }
}
