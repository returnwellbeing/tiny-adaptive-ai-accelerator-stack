module @jit_qkv_projection attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x16x2048xf32>, %arg1: tensor<2048x2048xf32>, %arg2: tensor<2048x256xf32>, %arg3: tensor<2048x256xf32>) -> (tensor<1x16x32x64xf32> {jax.result_info = "result[0]"}, tensor<1x16x4x64xf32> {jax.result_info = "result[1]"}, tensor<1x16x4x64xf32> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x2048xf32>, tensor<2048x2048xf32>) -> tensor<1x16x2048xf32>
    %1 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x2048xf32>, tensor<2048x256xf32>) -> tensor<1x16x256xf32>
    %2 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x2048xf32>, tensor<2048x256xf32>) -> tensor<1x16x256xf32>
    %3 = stablehlo.reshape %0 : (tensor<1x16x2048xf32>) -> tensor<1x16x32x64xf32>
    %4 = stablehlo.reshape %1 : (tensor<1x16x256xf32>) -> tensor<1x16x4x64xf32>
    %5 = stablehlo.reshape %2 : (tensor<1x16x256xf32>) -> tensor<1x16x4x64xf32>
    return %3, %4, %5 : tensor<1x16x32x64xf32>, tensor<1x16x4x64xf32>, tensor<1x16x4x64xf32>
  }
}

