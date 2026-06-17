module @jit_attention_output attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x16x8xf32>, %arg1: tensor<64x64xf32>) -> (tensor<1x16x64xf32> {jax.result_info = ""}) {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1, 3] : (tensor<1x8x16x8xf32>) -> tensor<1x16x8x8xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x16x8x8xf32>) -> tensor<1x16x64xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x64xf32>, tensor<64x64xf32>) -> tensor<1x16x64xf32>
    return %2 : tensor<1x16x64xf32>
  }
}

