find_package(LLVM CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR} ${MLIR_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(MLIR_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_MAIN_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)

set(PROJ_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(PROJ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJ_INCLUDE_DIR} ${PROJ_GENERATED_INCLUDE_DIR})

add_subdirectory(include/mlir)
add_subdirectory(lib)

add_library(entryPoint SHARED entryPoint.cpp)
target_include_directories(entryPoint PUBLIC src)

target_link_libraries(entryPoint PUBLIC
  LLVMObject
  LLVMCore
  LLVMSupport
  LLVMAsmParser
  LLVMOption
  LLVMCodeGen
  LLVMOrcJIT
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(entryPoint PUBLIC
  ${dialect_libs}
  ${extension_libs}
)
target_compile_options(entryPoint PUBLIC "-fno-rtti")

add_executable(standalone standalone.cpp)
target_link_libraries(standalone
  PRIVATE
  MLIREWTransforms
  MLIREWPipelines
  entryPoint
)