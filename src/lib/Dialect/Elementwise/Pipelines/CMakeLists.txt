add_mlir_dialect_library(MLIREWPipelines
  Pipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_GENERATED_INCLUDE_DIR}/mlir/Dialect/Elementwise

  LINK_LIBS PUBLIC
  MLIREWTransforms
)