module @jit_attention_scores_prefill attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x16x8xf32>, %arg1: tensor<1x8x16x8xf32>) -> (tensor<1x8x16x16xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16x8xf32>, tensor<1x8x16x8xf32>) -> tensor<1x8x16x16xf32>
    %cst = stablehlo.constant dense<2.82842708> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x8x16x16xf32>
    %2 = stablehlo.divide %0, %1 : tensor<1x8x16x16xf32>
    return %2 : tensor<1x8x16x16xf32>
  }
}
