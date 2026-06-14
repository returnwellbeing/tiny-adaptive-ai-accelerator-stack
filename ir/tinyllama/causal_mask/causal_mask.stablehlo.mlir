module @jit_apply_causal_mask attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x16x16xf32>, %arg1: tensor<16x16xi1>) -> (tensor<1x8x16x16xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2, 3] : (tensor<16x16xi1>) -> tensor<1x1x16x16xi1>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = call @_where(%0, %arg0, %cst) : (tensor<1x1x16x16xi1>, tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16x16xf32>
    return %1 : tensor<1x8x16x16xf32>
  }
  func.func private @_where(%arg0: tensor<1x1x16x16xi1>, %arg1: tensor<1x8x16x16xf32>, %arg2: tensor<f32>) -> tensor<1x8x16x16xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x16x16xi1>) -> tensor<1x8x16x16xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x8x16x16xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<1x8x16x16xi1>, tensor<1x8x16x16xf32>
    return %3 : tensor<1x8x16x16xf32>
  }
}
