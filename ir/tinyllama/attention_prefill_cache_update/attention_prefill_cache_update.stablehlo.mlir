module @jit_attention_prefill_cache_update attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x2x16x8xf32>, %arg1: tensor<1x2x16x8xf32>, %arg2: tensor<1x2x32x8xf32>, %arg3: tensor<1x2x32x8xf32>) -> (tensor<1x2x32x8xf32> {jax.result_info = "result[0]"}, tensor<1x2x32x8xf32> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_update_slice %arg2, %arg0, %c, %c_0, %c_1, %c_2 : (tensor<1x2x32x8xf32>, tensor<1x2x16x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x32x8xf32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %c_5 = stablehlo.constant dense<0> : tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.dynamic_update_slice %arg3, %arg1, %c_3, %c_4, %c_5, %c_6 : (tensor<1x2x32x8xf32>, tensor<1x2x16x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x32x8xf32>
    return %0, %1 : tensor<1x2x32x8xf32>, tensor<1x2x32x8xf32>
  }
}
