module @jit_linear attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x16x64xf32>, %arg1: tensor<64x256xf32>) -> (tensor<1x16x256xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x64xf32>, tensor<64x256xf32>) -> tensor<1x16x256xf32>
    return %0 : tensor<1x16x256xf32>
  }
}

