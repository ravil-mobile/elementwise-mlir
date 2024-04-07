// RUN: -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-cf  -convert-arith-to-llvm -convert-cf-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -llvm-request-c-wrappersl

// mlir-translate -mlir-to-llvmir | llc -filetype=obj -o simple.o
// clang ./simple.o -o simple.out
// https://discourse.llvm.org/t/how-to-compile-and-link-with-other-c-c-programs/4835/10

#accesses = [
  affine_map<(d0) -> (d0)>,
  affine_map<(d0) -> (d0)>,
  affine_map<(d0) -> ()>,
  affine_map<(d0) -> (d0)>
]

#trait = {
  iterator_types = ["parallel"],
  indexing_maps = #accesses
}

module {
  func.func private @foo_kernel(%arg0: f32, %arg1: f32, %arg2: f32) -> (f32) attributes {kernel} {
    %0 = arith.mulf %arg0, %arg1 : f32
    %1 = arith.addf %0, %arg2 : f32
    return %1 : f32
  }
  func.func @foo(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: memref<?xf32>) attributes {launcher, llvm.emit_c_interface} {
    linalg.generic #trait
      ins(%arg0, %arg1, %arg2: memref<?xf32>, memref<?xf32>, f32)
      outs(%arg3: memref<?xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
          %0 = func.call @foo_kernel(%a, %b, %c) : (f32, f32, f32) -> f32
          linalg.yield %0 : f32
      }

    return
  }
}
