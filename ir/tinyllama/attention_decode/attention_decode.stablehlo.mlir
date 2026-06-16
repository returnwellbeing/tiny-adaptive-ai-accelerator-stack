module @jit_attention_decode attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x1x8xf32>, %arg1: tensor<1x2x16x8xf32>, %arg2: tensor<1x2x16x8xf32>) -> (tensor<1x8x1x8xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 3, 4] : (tensor<1x2x16x8xf32>) -> tensor<1x2x1x16x8xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x16x8xf32>) -> tensor<1x2x4x16x8xf32>
    %2 = stablehlo.reshape %1 : (tensor<1x2x4x16x8xf32>) -> tensor<1x8x16x8xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 3, 4] : (tensor<1x2x16x8xf32>) -> tensor<1x2x1x16x8xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x16x8xf32>) -> tensor<1x2x4x16x8xf32>
    %5 = stablehlo.reshape %4 : (tensor<1x2x4x16x8xf32>) -> tensor<1x8x16x8xf32>
    %6 = stablehlo.dot_general %arg0, %2, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1x8xf32>, tensor<1x8x16x8xf32>) -> tensor<1x8x1x16xf32>
    %cst = stablehlo.constant dense<2.82842708> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x8x1x16xf32>
    %8 = stablehlo.divide %6, %7 : tensor<1x8x1x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x1x16xf32>, tensor<f32>) -> tensor<1x8x1xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %11 = stablehlo.maximum %10, %9 : tensor<1x8x1xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1x1xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2, 3] : (tensor<1x8x1x1xf32>) -> tensor<1x8x1x16xf32>
    %14 = stablehlo.subtract %8, %13 : tensor<1x8x1x16xf32>
    %15 = stablehlo.exponential %14 : tensor<1x8x1x16xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<1x8x1x16xf32>, tensor<f32>) -> tensor<1x8x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3] : (tensor<1x8x1x1xf32>) -> tensor<1x8x1x16xf32>
    %19 = stablehlo.divide %15, %18 : tensor<1x8x1x16xf32>
    %20 = stablehlo.dot_general %19, %5, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1x16xf32>, tensor<1x8x16x8xf32>) -> tensor<1x8x1x8xf32>
    return %20 : tensor<1x8x1x8xf32>
  }
}
