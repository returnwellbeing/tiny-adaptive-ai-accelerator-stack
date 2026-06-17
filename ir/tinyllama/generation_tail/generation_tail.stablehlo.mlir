module @jit_generation_tail attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x128xf32>, %arg3: tensor<128x64xf32>) -> (tensor<1x1x128xf32> {jax.result_info = "[0]"}, tensor<1x1xi32> {jax.result_info = "[1]"}, tensor<1x1x64xf32> {jax.result_info = "[2]"}) {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x1x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1x64xf32>, tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %4 = stablehlo.divide %2, %3 : tensor<1x1x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %6 = stablehlo.add %4, %5 : tensor<1x1x1xf32>
    %7 = stablehlo.rsqrt %6 : tensor<1x1x1xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x64xf32>
    %9 = stablehlo.multiply %arg0, %8 : tensor<1x1x64xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x1x64xf32>
    %12 = stablehlo.dot_general %11, %arg2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x128xf32>) -> tensor<1x1x128xf32>
    %13 = call @argmax(%12) : (tensor<1x1x128xf32>) -> tensor<1x1xi32>
    %14 = call @_take(%arg3, %13) : (tensor<128x64xf32>, tensor<1x1xi32>) -> tensor<1x1x64xf32>
    return %12, %13, %14 : tensor<1x1x128xf32>, tensor<1x1xi32>, tensor<1x1x64xf32>
  }
  func.func private @argmax(%arg0: tensor<1x1x128xf32>) -> tensor<1x1xi32> {
    %0 = stablehlo.iota dim = 2 : tensor<1x1x128xi32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [2] : (tensor<1x1x128xf32>, tensor<1x1x128xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x1xf32>, tensor<1x1xi32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %9, %10 : tensor<f32>, tensor<i32>
    }
    return %1#1 : tensor<1x1xi32>
  }
  func.func private @_take(%arg0: tensor<128x64xf32>, %arg1: tensor<1x1xi32>) -> tensor<1x1x64xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %c_0 = stablehlo.constant dense<128> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1x1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1x1xi1>, tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %c_1 = stablehlo.constant dense<127> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [2] : (tensor<1x1x1xi1>, tensor<i1>) -> tensor<1x1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<128x64xf32>, tensor<1x1x1xi32>) -> tensor<1x1x64xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<1x1x64xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x64xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<1x1x64xi1>, tensor<1x1x64xf32>
    return %15 : tensor<1x1x64xf32>
  }
  func.func private @_where(%arg0: tensor<1x1xi1>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x1xi32>) -> tensor<1x1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x1xi1>, tensor<1x1xi32>
    return %0 : tensor<1x1xi32>
  }
}

