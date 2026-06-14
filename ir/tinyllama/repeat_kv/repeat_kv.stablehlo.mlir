module @jit_repeat_kv attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x2x16x8xf32>) -> (tensor<1x8x16x8xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 3, 4] : (tensor<1x2x16x8xf32>) -> tensor<1x2x1x16x8xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x16x8xf32>) -> tensor<1x2x4x16x8xf32>
    %2 = stablehlo.reshape %1 : (tensor<1x2x4x16x8xf32>) -> tensor<1x8x16x8xf32>
    return %2 : tensor<1x8x16x8xf32>
  }
}

