module @jit_residual_add attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x16x64xf32>, %arg1: tensor<1x16x64xf32>) -> (tensor<1x16x64xf32> {jax.result_info = ""}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1x16x64xf32>
    return %0 : tensor<1x16x64xf32>
  }
}

