module @jit_attention_prefill attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x16x8xf32>, %arg1: tensor<1x2x16x8xf32>, %arg2: tensor<1x2x16x8xf32>) -> (tensor<1x8x16x8xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 3, 4] : (tensor<1x2x16x8xf32>) -> tensor<1x2x1x16x8xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x16x8xf32>) -> tensor<1x2x4x16x8xf32>
    %2 = stablehlo.reshape %1 : (tensor<1x2x4x16x8xf32>) -> tensor<1x8x16x8xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 3, 4] : (tensor<1x2x16x8xf32>) -> tensor<1x2x1x16x8xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x16x8xf32>) -> tensor<1x2x4x16x8xf32>
    %5 = stablehlo.reshape %4 : (tensor<1x2x4x16x8xf32>) -> tensor<1x8x16x8xf32>
    %6 = stablehlo.dot_general %arg0, %2, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16x8xf32>, tensor<1x8x16x8xf32>) -> tensor<1x8x16x16xf32>
    %cst = stablehlo.constant dense<2.82842708> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x8x16x16xf32>
    %8 = stablehlo.divide %6, %7 : tensor<1x8x16x16xf32>
    %9 = stablehlo.iota dim = 0 : tensor<16xi32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %11 = stablehlo.iota dim = 0 : tensor<16xi32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<16xi32>) -> tensor<1x16xi32>
    %13 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x1xi32>) -> tensor<16x16xi32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x16xi32>) -> tensor<16x16xi32>
    %15 = stablehlo.compare GE, %13, %14, SIGNED : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %16 = stablehlo.broadcast_in_dim %15, dims = [2, 3] : (tensor<16x16xi1>) -> tensor<1x1x16x16xi1>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = call @_where(%16, %8, %cst_0) : (tensor<1x1x16x16xi1>, tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16x16xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %18 = stablehlo.reduce(%17 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
    %20 = stablehlo.maximum %19, %18 : tensor<1x8x16xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x16xf32>
    %23 = stablehlo.subtract %17, %22 : tensor<1x8x16x16xf32>
    %24 = stablehlo.exponential %23 : tensor<1x8x16x16xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.reduce(%24 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x16xf32>
    %28 = stablehlo.divide %24, %27 : tensor<1x8x16x16xf32>
    %29 = stablehlo.dot_general %28, %5, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16x16xf32>, tensor<1x8x16x8xf32>) -> tensor<1x8x16x8xf32>
    return %29 : tensor<1x8x16x8xf32>
  }
  func.func private @_where(%arg0: tensor<1x1x16x16xi1>, %arg1: tensor<1x8x16x16xf32>, %arg2: tensor<f32>) -> tensor<1x8x16x16xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x16x16xi1>) -> tensor<1x8x16x16xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x8x16x16xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<1x8x16x16xi1>, tensor<1x8x16x16xf32>
    return %3 : tensor<1x8x16x16xf32>
  }
}
