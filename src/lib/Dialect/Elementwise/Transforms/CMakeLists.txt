add_mlir_dialect_library(MLIREWTransforms
  Tiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_GENERATED_INCLUDE_DIR}/mlir/Dialect/Elementwise

  DEPENDS
  MLIREWPassIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRLinalgTransforms
)