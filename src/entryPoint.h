#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include <string>

namespace ew {
struct Options {
  std::string outputDir{"/tmp/mlir"};
};
bool main(mlir::ModuleOp module, const Options &options);
} // namespace ew