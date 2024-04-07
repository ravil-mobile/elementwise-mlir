#ifndef MLIR_DIALECT_ELEMENTWISE_PASSES
#define MLIR_DIALECT_ELEMENTWISE_PASSES

include "mlir/Pass/PassBase.td"

def EwTilingPass : Pass<"ew-tiling", "::mlir::func::FuncOp"> {
  let summary = "tiling pass";
  let dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect"];
}

#endif // MLIR_DIALECT_ELEMENTWISE_PASSES
