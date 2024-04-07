#include "entryPoint.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>
#include <string>

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init(""));

int main(int argc, char *argv[]) {
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "elmentwise compiler\n");
  if (inputFilename.empty()) {
    std::cerr << "the input mlir file is not provided" << std::endl;
    return -1;
  }

  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect, mlir::affine::AffineDialect,
                      mlir::memref::MemRefDialect, mlir::math::MathDialect,
                      mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
                      mlir::scf::SCFDialect,
                      mlir::bufferization::BufferizationDialect>();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename.getValue());

  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> moduleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  ew::Options options{};
  const auto isOk = ew::main(*moduleRef, options);
  if (not isOk) {
    return -1;
  }
  return 0;
}