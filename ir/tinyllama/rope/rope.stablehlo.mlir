module @jit_apply_rope attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x16x8xf32>, %arg1: tensor<1x2x16x8xf32>, %arg2: tensor<16x8xf32>, %arg3: tensor<16x8xf32>) -> (tensor<1x8x16x8xf32> {jax.result_info = "result[0]"}, tensor<1x2x16x8xf32> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2, 3] : (tensor<16x8xf32>) -> tensor<1x1x16x8xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2, 3] : (tensor<16x8xf32>) -> tensor<1x1x16x8xf32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x1x16x8xf32>) -> tensor<1x8x16x8xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x8x16x8xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:8, 0:16, 0:4] : (tensor<1x8x16x8xf32>) -> tensor<1x8x16x4xf32>
    %5 = stablehlo.slice %arg0 [0:1, 0:8, 0:16, 4:8] : (tensor<1x8x16x8xf32>) -> tensor<1x8x16x4xf32>
    %6 = stablehlo.negate %5 : tensor<1x8x16x4xf32>
    %7 = stablehlo.concatenate %6, %4, dim = 3 : (tensor<1x8x16x4xf32>, tensor<1x8x16x4xf32>) -> tensor<1x8x16x8xf32>
    %8 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x1x16x8xf32>) -> tensor<1x8x16x8xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x8x16x8xf32>
    %10 = stablehlo.add %3, %9 : tensor<1x8x16x8xf32>
    %11 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x1x16x8xf32>) -> tensor<1x2x16x8xf32>
    %12 = stablehlo.multiply %arg1, %11 : tensor<1x2x16x8xf32>
    %13 = stablehlo.slice %arg1 [0:1, 0:2, 0:16, 0:4] : (tensor<1x2x16x8xf32>) -> tensor<1x2x16x4xf32>
    %14 = stablehlo.slice %arg1 [0:1, 0:2, 0:16, 4:8] : (tensor<1x2x16x8xf32>) -> tensor<1x2x16x4xf32>
    %15 = stablehlo.negate %14 : tensor<1x2x16x4xf32>
    %16 = stablehlo.concatenate %15, %13, dim = 3 : (tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>) -> tensor<1x2x16x8xf32>
    %17 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x1x16x8xf32>) -> tensor<1x2x16x8xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x2x16x8xf32>
    %19 = stablehlo.add %12, %18 : tensor<1x2x16x8xf32>
    return %10, %19 : tensor<1x8x16x8xf32>, tensor<1x2x16x8xf32>
  }
}

