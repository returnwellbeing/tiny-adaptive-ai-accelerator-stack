module @jit_rms_norm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x16x64xf32>, %arg1: tensor<64xf32>) -> (tensor<1x16x64xf32> {jax.result_info = "result"}) {
    %0 = chlo.square %arg0 : tensor<1x16x64xf32> -> tensor<1x16x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x64xf32>, tensor<f32>) -> tensor<1x16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_0 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %4 = stablehlo.divide %2, %3 : tensor<1x16x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %6 = stablehlo.add %4, %5 : tensor<1x16x1xf32>
    %7 = stablehlo.rsqrt %6 : tensor<1x16x1xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x64xf32>
    %9 = stablehlo.multiply %arg0, %8 : tensor<1x16x64xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x1x64xf32>) -> tensor<1x16x64xf32>
    %12 = stablehlo.multiply %9, %11 : tensor<1x16x64xf32>
    return %12 : tensor<1x16x64xf32>
  }
}

