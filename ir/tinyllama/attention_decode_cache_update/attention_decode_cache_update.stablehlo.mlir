module @jit_attention_decode_cache_update attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x2x1x8xf32>, %arg1: tensor<1x2x1x8xf32>, %arg2: tensor<1x2x32x8xf32>, %arg3: tensor<1x2x32x8xf32>, %arg4: tensor<i32>) -> (tensor<1x2x32x8xf32> {jax.result_info = "result[0]"}, tensor<1x2x32x8xf32> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare LT, %arg4, %c, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<32> : tensor<i32>
    %1 = stablehlo.add %arg4, %c_0 : tensor<i32>
    %2 = stablehlo.select %0, %1, %arg4 : tensor<i1>, tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.dynamic_update_slice %arg2, %arg0, %c_1, %c_2, %2, %c_3 : (tensor<1x2x32x8xf32>, tensor<1x2x1x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x32x8xf32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.compare LT, %arg4, %c_4, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_5 = stablehlo.constant dense<32> : tensor<i32>
    %5 = stablehlo.add %arg4, %c_5 : tensor<i32>
    %6 = stablehlo.select %4, %5, %arg4 : tensor<i1>, tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i32>
    %c_7 = stablehlo.constant dense<0> : tensor<i32>
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.dynamic_update_slice %arg3, %arg1, %c_6, %c_7, %6, %c_8 : (tensor<1x2x32x8xf32>, tensor<1x2x1x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x32x8xf32>
    return %3, %7 : tensor<1x2x32x8xf32>, tensor<1x2x32x8xf32>
  }
}
