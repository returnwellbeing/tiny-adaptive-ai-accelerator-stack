module @jit_decoder_layer_prefill_decode_loop attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x16x64xf32>, %arg1: tensor<1x1x64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64x64xf32>, %arg6: tensor<64x16xf32>, %arg7: tensor<64x16xf32>, %arg8: tensor<64x64xf32>, %arg9: tensor<64x128xf32>, %arg10: tensor<128x64xf32>, %arg11: tensor<64x256xf32>, %arg12: tensor<64x256xf32>, %arg13: tensor<256x64xf32>, %arg14: tensor<16x8xf32>, %arg15: tensor<16x8xf32>, %arg16: tensor<4x8xf32>, %arg17: tensor<4x8xf32>, %arg18: tensor<1x2x32x8xf32>, %arg19: tensor<1x2x32x8xf32>, %arg20: tensor<i32>) -> (tensor<1x16x64xf32> {jax.result_info = "[0]"}, tensor<1x4x64xf32> {jax.result_info = "[1]"}, tensor<1x4x128xf32> {jax.result_info = "[2]"}, tensor<1x4xi32> {jax.result_info = "[3]"}, tensor<1x2x32x8xf32> {jax.result_info = "[4]"}, tensor<1x2x32x8xf32> {jax.result_info = "[5]"}) {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x16x64xf32>
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
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x1x64xf32>) -> tensor<1x16x64xf32>
    %12 = stablehlo.multiply %9, %11 : tensor<1x16x64xf32>
    %13 = stablehlo.dot_general %12, %arg5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x64xf32>, tensor<64x64xf32>) -> tensor<1x16x64xf32>
    %14 = stablehlo.dot_general %12, %arg6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x64xf32>, tensor<64x16xf32>) -> tensor<1x16x16xf32>
    %15 = stablehlo.dot_general %12, %arg7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x64xf32>, tensor<64x16xf32>) -> tensor<1x16x16xf32>
    %16 = stablehlo.reshape %13 : (tensor<1x16x64xf32>) -> tensor<1x16x8x8xf32>
    %17 = stablehlo.reshape %14 : (tensor<1x16x16xf32>) -> tensor<1x16x2x8xf32>
    %18 = stablehlo.reshape %15 : (tensor<1x16x16xf32>) -> tensor<1x16x2x8xf32>
    %19 = stablehlo.transpose %16, dims = [0, 2, 1, 3] : (tensor<1x16x8x8xf32>) -> tensor<1x8x16x8xf32>
    %20 = stablehlo.transpose %17, dims = [0, 2, 1, 3] : (tensor<1x16x2x8xf32>) -> tensor<1x2x16x8xf32>
    %21 = stablehlo.transpose %18, dims = [0, 2, 1, 3] : (tensor<1x16x2x8xf32>) -> tensor<1x2x16x8xf32>
    %22 = stablehlo.broadcast_in_dim %arg14, dims = [2, 3] : (tensor<16x8xf32>) -> tensor<1x1x16x8xf32>
    %23 = stablehlo.broadcast_in_dim %arg15, dims = [2, 3] : (tensor<16x8xf32>) -> tensor<1x1x16x8xf32>
    %24 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2, 3] : (tensor<1x1x16x8xf32>) -> tensor<1x8x16x8xf32>
    %25 = stablehlo.multiply %19, %24 : tensor<1x8x16x8xf32>
    %26 = stablehlo.slice %19 [0:1, 0:8, 0:16, 0:4] : (tensor<1x8x16x8xf32>) -> tensor<1x8x16x4xf32>
    %27 = stablehlo.slice %19 [0:1, 0:8, 0:16, 4:8] : (tensor<1x8x16x8xf32>) -> tensor<1x8x16x4xf32>
    %28 = stablehlo.negate %27 : tensor<1x8x16x4xf32>
    %29 = stablehlo.concatenate %28, %26, dim = 3 : (tensor<1x8x16x4xf32>, tensor<1x8x16x4xf32>) -> tensor<1x8x16x8xf32>
    %30 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x1x16x8xf32>) -> tensor<1x8x16x8xf32>
    %31 = stablehlo.multiply %29, %30 : tensor<1x8x16x8xf32>
    %32 = stablehlo.add %25, %31 : tensor<1x8x16x8xf32>
    %33 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2, 3] : (tensor<1x1x16x8xf32>) -> tensor<1x2x16x8xf32>
    %34 = stablehlo.multiply %20, %33 : tensor<1x2x16x8xf32>
    %35 = stablehlo.slice %20 [0:1, 0:2, 0:16, 0:4] : (tensor<1x2x16x8xf32>) -> tensor<1x2x16x4xf32>
    %36 = stablehlo.slice %20 [0:1, 0:2, 0:16, 4:8] : (tensor<1x2x16x8xf32>) -> tensor<1x2x16x4xf32>
    %37 = stablehlo.negate %36 : tensor<1x2x16x4xf32>
    %38 = stablehlo.concatenate %37, %35, dim = 3 : (tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>) -> tensor<1x2x16x8xf32>
    %39 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x1x16x8xf32>) -> tensor<1x2x16x8xf32>
    %40 = stablehlo.multiply %38, %39 : tensor<1x2x16x8xf32>
    %41 = stablehlo.add %34, %40 : tensor<1x2x16x8xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 3, 4] : (tensor<1x2x16x8xf32>) -> tensor<1x2x1x16x8xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x16x8xf32>) -> tensor<1x2x4x16x8xf32>
    %44 = stablehlo.reshape %43 : (tensor<1x2x4x16x8xf32>) -> tensor<1x8x16x8xf32>
    %45 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 3, 4] : (tensor<1x2x16x8xf32>) -> tensor<1x2x1x16x8xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x16x8xf32>) -> tensor<1x2x4x16x8xf32>
    %47 = stablehlo.reshape %46 : (tensor<1x2x4x16x8xf32>) -> tensor<1x8x16x8xf32>
    %48 = stablehlo.dot_general %32, %44, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16x8xf32>, tensor<1x8x16x8xf32>) -> tensor<1x8x16x16xf32>
    %cst_2 = stablehlo.constant dense<2.82842708> : tensor<f32>
    %49 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x16x16xf32>
    %50 = stablehlo.divide %48, %49 : tensor<1x8x16x16xf32>
    %51 = stablehlo.iota dim = 0 : tensor<16xi32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %53 = stablehlo.iota dim = 0 : tensor<16xi32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [1] : (tensor<16xi32>) -> tensor<1x16xi32>
    %55 = stablehlo.broadcast_in_dim %52, dims = [0, 1] : (tensor<16x1xi32>) -> tensor<16x16xi32>
    %56 = stablehlo.broadcast_in_dim %54, dims = [0, 1] : (tensor<1x16xi32>) -> tensor<16x16xi32>
    %57 = stablehlo.compare  GE, %55, %56,  SIGNED : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %58 = stablehlo.broadcast_in_dim %57, dims = [2, 3] : (tensor<16x16xi1>) -> tensor<1x1x16x16xi1>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %59 = call @_where(%58, %50, %cst_3) : (tensor<1x1x16x16xi1>, tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16x16xf32>
    %cst_4 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %60 = stablehlo.reduce(%59 init: %cst_4) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %cst_5 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
    %62 = stablehlo.maximum %61, %60 : tensor<1x8x16xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %64 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x16xf32>
    %65 = stablehlo.subtract %59, %64 : tensor<1x8x16x16xf32>
    %66 = stablehlo.exponential %65 : tensor<1x8x16x16xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %67 = stablehlo.reduce(%66 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x16xf32>
    %70 = stablehlo.divide %66, %69 : tensor<1x8x16x16xf32>
    %71 = stablehlo.dot_general %70, %47, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16x16xf32>, tensor<1x8x16x8xf32>) -> tensor<1x8x16x8xf32>
    %72 = stablehlo.transpose %71, dims = [0, 2, 1, 3] : (tensor<1x8x16x8xf32>) -> tensor<1x16x8x8xf32>
    %73 = stablehlo.reshape %72 : (tensor<1x16x8x8xf32>) -> tensor<1x16x64xf32>
    %74 = stablehlo.dot_general %73, %arg8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x64xf32>, tensor<64x64xf32>) -> tensor<1x16x64xf32>
    %75 = stablehlo.add %arg0, %74 : tensor<1x16x64xf32>
    %76 = stablehlo.multiply %75, %75 : tensor<1x16x64xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %77 = stablehlo.reduce(%76 init: %cst_7) applies stablehlo.add across dimensions = [2] : (tensor<1x16x64xf32>, tensor<f32>) -> tensor<1x16xf32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %79 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %80 = stablehlo.divide %78, %79 : tensor<1x16x1xf32>
    %81 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x16x1xf32>
    %83 = stablehlo.rsqrt %82 : tensor<1x16x1xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x64xf32>
    %85 = stablehlo.multiply %75, %84 : tensor<1x16x64xf32>
    %86 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1, 2] : (tensor<1x1x64xf32>) -> tensor<1x16x64xf32>
    %88 = stablehlo.multiply %85, %87 : tensor<1x16x64xf32>
    %89 = stablehlo.dot_general %88, %arg11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x64xf32>, tensor<64x256xf32>) -> tensor<1x16x256xf32>
    %90 = stablehlo.dot_general %88, %arg12, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x64xf32>, tensor<64x256xf32>) -> tensor<1x16x256xf32>
    %91 = call @silu(%89) : (tensor<1x16x256xf32>) -> tensor<1x16x256xf32>
    %92 = stablehlo.multiply %91, %90 : tensor<1x16x256xf32>
    %93 = stablehlo.dot_general %92, %arg13, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x256xf32>, tensor<256x64xf32>) -> tensor<1x16x64xf32>
    %94 = stablehlo.add %75, %93 : tensor<1x16x64xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %95 = stablehlo.dynamic_update_slice %arg18, %41, %c, %c, %c, %c : (tensor<1x2x32x8xf32>, tensor<1x2x16x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x32x8xf32>
    %96 = stablehlo.dynamic_update_slice %arg19, %21, %c, %c, %c, %c : (tensor<1x2x32x8xf32>, tensor<1x2x16x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x32x8xf32>
    %97 = stablehlo.iota dim = 0 : tensor<4xi32>
    %98 = stablehlo.broadcast_in_dim %arg20, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %99 = stablehlo.add %98, %97 : tensor<4xi32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %100 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<4x1x64xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %101 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<4x1x128xf32>
    %c_10 = stablehlo.constant dense<0> : tensor<i32>
    %102 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<4x1xi32>
    %c_11 = stablehlo.constant dense<0> : tensor<i32>
    %103:22 = stablehlo.while(%iterArg = %arg16, %iterArg_12 = %arg17, %iterArg_13 = %99, %iterArg_14 = %arg2, %iterArg_15 = %arg5, %iterArg_16 = %arg6, %iterArg_17 = %arg7, %iterArg_18 = %arg8, %iterArg_19 = %arg3, %iterArg_20 = %arg11, %iterArg_21 = %arg12, %iterArg_22 = %arg13, %iterArg_23 = %arg4, %iterArg_24 = %arg9, %iterArg_25 = %arg10, %iterArg_26 = %c_11, %iterArg_27 = %95, %iterArg_28 = %96, %iterArg_29 = %arg1, %iterArg_30 = %100, %iterArg_31 = %101, %iterArg_32 = %102) : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4xi32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x16xf32>, tensor<64x16xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<256x64xf32>, tensor<64xf32>, tensor<64x128xf32>, tensor<128x64xf32>, tensor<i32>, tensor<1x2x32x8xf32>, tensor<1x2x32x8xf32>, tensor<1x1x64xf32>, tensor<4x1x64xf32>, tensor<4x1x128xf32>, tensor<4x1xi32>
     cond {
      %c_33 = stablehlo.constant dense<4> : tensor<i32>
      %107 = stablehlo.compare  LT, %iterArg_26, %c_33,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %107 : tensor<i1>
    } do {
      %c_33 = stablehlo.constant dense<0> : tensor<i32>
      %107 = stablehlo.compare  LT, %iterArg_26, %c_33,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %108 = stablehlo.convert %iterArg_26 : tensor<i32>
      %c_34 = stablehlo.constant dense<4> : tensor<i32>
      %109 = stablehlo.add %108, %c_34 : tensor<i32>
      %110 = stablehlo.select %107, %109, %iterArg_26 : tensor<i1>, tensor<i32>
      %c_35 = stablehlo.constant dense<0> : tensor<i32>
      %111 = stablehlo.dynamic_slice %iterArg, %110, %c_35, sizes = [1, 8] : (tensor<4x8xf32>, tensor<i32>, tensor<i32>) -> tensor<1x8xf32>
      %112 = stablehlo.reshape %111 : (tensor<1x8xf32>) -> tensor<8xf32>
      %113 = stablehlo.compare  LT, %iterArg_26, %c_33,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %114 = stablehlo.convert %iterArg_26 : tensor<i32>
      %115 = stablehlo.add %114, %c_34 : tensor<i32>
      %116 = stablehlo.select %113, %115, %iterArg_26 : tensor<i1>, tensor<i32>
      %117 = stablehlo.dynamic_slice %iterArg_12, %116, %c_35, sizes = [1, 8] : (tensor<4x8xf32>, tensor<i32>, tensor<i32>) -> tensor<1x8xf32>
      %118 = stablehlo.reshape %117 : (tensor<1x8xf32>) -> tensor<8xf32>
      %119 = stablehlo.compare  LT, %iterArg_26, %c_33,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %120 = stablehlo.convert %iterArg_26 : tensor<i32>
      %121 = stablehlo.add %120, %c_34 : tensor<i32>
      %122 = stablehlo.select %119, %121, %iterArg_26 : tensor<i1>, tensor<i32>
      %123 = stablehlo.dynamic_slice %iterArg_13, %122, sizes = [1] : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
      %124 = stablehlo.reshape %123 : (tensor<1xi32>) -> tensor<i32>
      %125:6 = func.call @None(%iterArg_14, %iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %iterArg_19, %iterArg_20, %iterArg_21, %iterArg_22, %iterArg_23, %iterArg_24, %iterArg_25, %iterArg_27, %iterArg_28, %iterArg_29, %112, %118, %124) : (tensor<64xf32>, tensor<64x64xf32>, tensor<64x16xf32>, tensor<64x16xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<256x64xf32>, tensor<64xf32>, tensor<64x128xf32>, tensor<128x64xf32>, tensor<1x2x32x8xf32>, tensor<1x2x32x8xf32>, tensor<1x1x64xf32>, tensor<8xf32>, tensor<8xf32>, tensor<i32>) -> (tensor<1x2x32x8xf32>, tensor<1x2x32x8xf32>, tensor<1x1x64xf32>, tensor<1x64xf32>, tensor<1x128xf32>, tensor<1xi32>)
      %126 = stablehlo.broadcast_in_dim %125#3, dims = [1, 2] : (tensor<1x64xf32>) -> tensor<1x1x64xf32>
      %127 = stablehlo.compare  LT, %iterArg_26, %c_33,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %128 = stablehlo.convert %iterArg_26 : tensor<i32>
      %129 = stablehlo.add %128, %c_34 : tensor<i32>
      %130 = stablehlo.select %127, %129, %iterArg_26 : tensor<i1>, tensor<i32>
      %131 = stablehlo.dynamic_update_slice %iterArg_30, %126, %130, %c_35, %c_35 : (tensor<4x1x64xf32>, tensor<1x1x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x64xf32>
      %132 = stablehlo.broadcast_in_dim %125#4, dims = [1, 2] : (tensor<1x128xf32>) -> tensor<1x1x128xf32>
      %133 = stablehlo.compare  LT, %iterArg_26, %c_33,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %134 = stablehlo.convert %iterArg_26 : tensor<i32>
      %135 = stablehlo.add %134, %c_34 : tensor<i32>
      %136 = stablehlo.select %133, %135, %iterArg_26 : tensor<i1>, tensor<i32>
      %137 = stablehlo.dynamic_update_slice %iterArg_31, %132, %136, %c_35, %c_35 : (tensor<4x1x128xf32>, tensor<1x1x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x128xf32>
      %138 = stablehlo.broadcast_in_dim %125#5, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
      %139 = stablehlo.compare  LT, %iterArg_26, %c_33,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %140 = stablehlo.convert %iterArg_26 : tensor<i32>
      %141 = stablehlo.add %140, %c_34 : tensor<i32>
      %142 = stablehlo.select %139, %141, %iterArg_26 : tensor<i1>, tensor<i32>
      %143 = stablehlo.dynamic_update_slice %iterArg_32, %138, %142, %c_35 : (tensor<4x1xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<4x1xi32>
      %c_36 = stablehlo.constant dense<1> : tensor<i32>
      %144 = stablehlo.add %iterArg_26, %c_36 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_12, %iterArg_13, %iterArg_14, %iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %iterArg_19, %iterArg_20, %iterArg_21, %iterArg_22, %iterArg_23, %iterArg_24, %iterArg_25, %144, %125#0, %125#1, %125#2, %131, %137, %143 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4xi32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x16xf32>, tensor<64x16xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<256x64xf32>, tensor<64xf32>, tensor<64x128xf32>, tensor<128x64xf32>, tensor<i32>, tensor<1x2x32x8xf32>, tensor<1x2x32x8xf32>, tensor<1x1x64xf32>, tensor<4x1x64xf32>, tensor<4x1x128xf32>, tensor<4x1xi32>
    }
    %104 = stablehlo.transpose %103#19, dims = [1, 0, 2] : (tensor<4x1x64xf32>) -> tensor<1x4x64xf32>
    %105 = stablehlo.transpose %103#20, dims = [1, 0, 2] : (tensor<4x1x128xf32>) -> tensor<1x4x128xf32>
    %106 = stablehlo.transpose %103#21, dims = [1, 0] : (tensor<4x1xi32>) -> tensor<1x4xi32>
    return %94, %104, %105, %106, %103#16, %103#17 : tensor<1x16x64xf32>, tensor<1x4x64xf32>, tensor<1x4x128xf32>, tensor<1x4xi32>, tensor<1x2x32x8xf32>, tensor<1x2x32x8xf32>
  }
  func.func private @_where(%arg0: tensor<1x1x16x16xi1>, %arg1: tensor<1x8x16x16xf32>, %arg2: tensor<f32>) -> tensor<1x8x16x16xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x16x16xi1>) -> tensor<1x8x16x16xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x8x16x16xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<1x8x16x16xi1>, tensor<1x8x16x16xf32>
    return %3 : tensor<1x8x16x16xf32>
  }
  func.func private @silu(%arg0: tensor<1x16x256xf32>) -> tensor<1x16x256xf32> {
    %0 = stablehlo.negate %arg0 : tensor<1x16x256xf32>
    %1 = stablehlo.exponential %0 : tensor<1x16x256xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %3 = stablehlo.add %2, %1 : tensor<1x16x256xf32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %5 = stablehlo.divide %4, %3 : tensor<1x16x256xf32>
    %6 = stablehlo.multiply %arg0, %5 : tensor<1x16x256xf32>
    return %6 : tensor<1x16x256xf32>
  }
  func.func private @None(%arg0: tensor<64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x16xf32>, %arg3: tensor<64x16xf32>, %arg4: tensor<64x64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x256xf32>, %arg7: tensor<64x256xf32>, %arg8: tensor<256x64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64x128xf32>, %arg11: tensor<128x64xf32>, %arg12: tensor<1x2x32x8xf32>, %arg13: tensor<1x2x32x8xf32>, %arg14: tensor<1x1x64xf32>, %arg15: tensor<8xf32>, %arg16: tensor<8xf32>, %arg17: tensor<i32>) -> (tensor<1x2x32x8xf32>, tensor<1x2x32x8xf32>, tensor<1x1x64xf32>, tensor<1x64xf32>, tensor<1x128xf32>, tensor<1xi32>) {
    %0 = stablehlo.broadcast_in_dim %arg15, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %1 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %2 = stablehlo.multiply %arg14, %arg14 : tensor<1x1x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1x64xf32>, tensor<f32>) -> tensor<1x1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %6 = stablehlo.divide %4, %5 : tensor<1x1x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %8 = stablehlo.add %6, %7 : tensor<1x1x1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1x1x1xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x64xf32>
    %11 = stablehlo.multiply %arg14, %10 : tensor<1x1x64xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<1x1x64xf32>
    %14 = stablehlo.dot_general %13, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x64xf32>) -> tensor<1x1x64xf32>
    %15 = stablehlo.dot_general %13, %arg2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x16xf32>) -> tensor<1x1x16xf32>
    %16 = stablehlo.dot_general %13, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x16xf32>) -> tensor<1x1x16xf32>
    %17 = stablehlo.reshape %14 : (tensor<1x1x64xf32>) -> tensor<1x1x8x8xf32>
    %18 = stablehlo.reshape %15 : (tensor<1x1x16xf32>) -> tensor<1x1x2x8xf32>
    %19 = stablehlo.reshape %16 : (tensor<1x1x16xf32>) -> tensor<1x1x2x8xf32>
    %20 = stablehlo.transpose %17, dims = [0, 2, 1, 3] : (tensor<1x1x8x8xf32>) -> tensor<1x8x1x8xf32>
    %21 = stablehlo.transpose %18, dims = [0, 2, 1, 3] : (tensor<1x1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %22 = stablehlo.transpose %19, dims = [0, 2, 1, 3] : (tensor<1x1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %23 = stablehlo.broadcast_in_dim %0, dims = [2, 3] : (tensor<1x8xf32>) -> tensor<1x1x1x8xf32>
    %24 = stablehlo.broadcast_in_dim %1, dims = [2, 3] : (tensor<1x8xf32>) -> tensor<1x1x1x8xf32>
    %25 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xf32>) -> tensor<1x8x1x8xf32>
    %26 = stablehlo.multiply %20, %25 : tensor<1x8x1x8xf32>
    %27 = stablehlo.slice %20 [0:1, 0:8, 0:1, 0:4] : (tensor<1x8x1x8xf32>) -> tensor<1x8x1x4xf32>
    %28 = stablehlo.slice %20 [0:1, 0:8, 0:1, 4:8] : (tensor<1x8x1x8xf32>) -> tensor<1x8x1x4xf32>
    %29 = stablehlo.negate %28 : tensor<1x8x1x4xf32>
    %30 = stablehlo.concatenate %29, %27, dim = 3 : (tensor<1x8x1x4xf32>, tensor<1x8x1x4xf32>) -> tensor<1x8x1x8xf32>
    %31 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xf32>) -> tensor<1x8x1x8xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<1x8x1x8xf32>
    %33 = stablehlo.add %26, %32 : tensor<1x8x1x8xf32>
    %34 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xf32>) -> tensor<1x2x1x8xf32>
    %35 = stablehlo.multiply %21, %34 : tensor<1x2x1x8xf32>
    %36 = stablehlo.slice %21 [0:1, 0:2, 0:1, 0:4] : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x4xf32>
    %37 = stablehlo.slice %21 [0:1, 0:2, 0:1, 4:8] : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x4xf32>
    %38 = stablehlo.negate %37 : tensor<1x2x1x4xf32>
    %39 = stablehlo.concatenate %38, %36, dim = 3 : (tensor<1x2x1x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x1x8xf32>
    %40 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xf32>) -> tensor<1x2x1x8xf32>
    %41 = stablehlo.multiply %39, %40 : tensor<1x2x1x8xf32>
    %42 = stablehlo.add %35, %41 : tensor<1x2x1x8xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %43 = stablehlo.compare  LT, %arg17, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_2 = stablehlo.constant dense<32> : tensor<i32>
    %44 = stablehlo.add %arg17, %c_2 : tensor<i32>
    %45 = stablehlo.select %43, %44, %arg17 : tensor<i1>, tensor<i32>
    %46 = stablehlo.dynamic_update_slice %arg12, %42, %c, %c, %45, %c : (tensor<1x2x32x8xf32>, tensor<1x2x1x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x32x8xf32>
    %47 = stablehlo.compare  LT, %arg17, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %48 = stablehlo.add %arg17, %c_2 : tensor<i32>
    %49 = stablehlo.select %47, %48, %arg17 : tensor<i1>, tensor<i32>
    %50 = stablehlo.dynamic_update_slice %arg13, %22, %c, %c, %49, %c : (tensor<1x2x32x8xf32>, tensor<1x2x1x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x32x8xf32>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %51 = stablehlo.add %arg17, %c_3 : tensor<i32>
    %52 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 3, 4] : (tensor<1x2x32x8xf32>) -> tensor<1x2x1x32x8xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x32x8xf32>) -> tensor<1x2x4x32x8xf32>
    %54 = stablehlo.reshape %53 : (tensor<1x2x4x32x8xf32>) -> tensor<1x8x32x8xf32>
    %55 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 3, 4] : (tensor<1x2x32x8xf32>) -> tensor<1x2x1x32x8xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x32x8xf32>) -> tensor<1x2x4x32x8xf32>
    %57 = stablehlo.reshape %56 : (tensor<1x2x4x32x8xf32>) -> tensor<1x8x32x8xf32>
    %58 = stablehlo.dot_general %33, %54, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1x8xf32>, tensor<1x8x32x8xf32>) -> tensor<1x8x1x32xf32>
    %cst_4 = stablehlo.constant dense<2.82842708> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1x32xf32>
    %60 = stablehlo.divide %58, %59 : tensor<1x8x1x32xf32>
    %61 = stablehlo.iota dim = 0 : tensor<32xi32>
    %62 = stablehlo.broadcast_in_dim %51, dims = [] : (tensor<i32>) -> tensor<32xi32>
    %63 = stablehlo.compare  LT, %61, %62,  SIGNED : (tensor<32xi32>, tensor<32xi32>) -> tensor<32xi1>
    %64 = stablehlo.broadcast_in_dim %63, dims = [3] : (tensor<32xi1>) -> tensor<1x1x1x32xi1>
    %cst_5 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %65 = call @_where_0(%64, %60, %cst_5) : (tensor<1x1x1x32xi1>, tensor<1x8x1x32xf32>, tensor<f32>) -> tensor<1x8x1x32xf32>
    %cst_6 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %66 = stablehlo.reduce(%65 init: %cst_6) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x1x32xf32>, tensor<f32>) -> tensor<1x8x1xf32>
    %cst_7 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %68 = stablehlo.maximum %67, %66 : tensor<1x8x1xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1x1xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2, 3] : (tensor<1x8x1x1xf32>) -> tensor<1x8x1x32xf32>
    %71 = stablehlo.subtract %65, %70 : tensor<1x8x1x32xf32>
    %72 = stablehlo.exponential %71 : tensor<1x8x1x32xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %73 = stablehlo.reduce(%72 init: %cst_8) applies stablehlo.add across dimensions = [3] : (tensor<1x8x1x32xf32>, tensor<f32>) -> tensor<1x8x1xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1x1xf32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2, 3] : (tensor<1x8x1x1xf32>) -> tensor<1x8x1x32xf32>
    %76 = stablehlo.divide %72, %75 : tensor<1x8x1x32xf32>
    %77 = stablehlo.dot_general %76, %57, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1x32xf32>, tensor<1x8x32x8xf32>) -> tensor<1x8x1x8xf32>
    %78 = stablehlo.transpose %77, dims = [0, 2, 1, 3] : (tensor<1x8x1x8xf32>) -> tensor<1x1x8x8xf32>
    %79 = stablehlo.reshape %78 : (tensor<1x1x8x8xf32>) -> tensor<1x1x64xf32>
    %80 = stablehlo.dot_general %79, %arg4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x64xf32>) -> tensor<1x1x64xf32>
    %81 = stablehlo.add %arg14, %80 : tensor<1x1x64xf32>
    %82 = stablehlo.multiply %81, %81 : tensor<1x1x64xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %83 = stablehlo.reduce(%82 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<1x1x64xf32>, tensor<f32>) -> tensor<1x1xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %85 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %86 = stablehlo.divide %84, %85 : tensor<1x1x1xf32>
    %87 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x1x1xf32>
    %89 = stablehlo.rsqrt %88 : tensor<1x1x1xf32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x64xf32>
    %91 = stablehlo.multiply %81, %90 : tensor<1x1x64xf32>
    %92 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %93 = stablehlo.multiply %91, %92 : tensor<1x1x64xf32>
    %94 = stablehlo.dot_general %93, %arg6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x256xf32>) -> tensor<1x1x256xf32>
    %95 = stablehlo.dot_general %93, %arg7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x256xf32>) -> tensor<1x1x256xf32>
    %96 = call @silu_1(%94) : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %97 = stablehlo.multiply %96, %95 : tensor<1x1x256xf32>
    %98 = stablehlo.dot_general %97, %arg8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x256xf32>, tensor<256x64xf32>) -> tensor<1x1x64xf32>
    %99 = stablehlo.add %81, %98 : tensor<1x1x64xf32>
    %100 = stablehlo.multiply %99, %99 : tensor<1x1x64xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %101 = stablehlo.reduce(%100 init: %cst_10) applies stablehlo.add across dimensions = [2] : (tensor<1x1x64xf32>, tensor<f32>) -> tensor<1x1xf32>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %103 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %104 = stablehlo.divide %102, %103 : tensor<1x1x1xf32>
    %105 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %106 = stablehlo.add %104, %105 : tensor<1x1x1xf32>
    %107 = stablehlo.rsqrt %106 : tensor<1x1x1xf32>
    %108 = stablehlo.broadcast_in_dim %107, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x64xf32>
    %109 = stablehlo.multiply %99, %108 : tensor<1x1x64xf32>
    %110 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %111 = stablehlo.multiply %109, %110 : tensor<1x1x64xf32>
    %112 = stablehlo.dot_general %111, %arg10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x128xf32>) -> tensor<1x1x128xf32>
    %113 = call @argmax(%112) : (tensor<1x1x128xf32>) -> tensor<1x1xi32>
    %114 = call @_take(%arg11, %113) : (tensor<128x64xf32>, tensor<1x1xi32>) -> tensor<1x1x64xf32>
    %115 = stablehlo.slice %99 [0:1, 0:1, 0:64] : (tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
    %116 = stablehlo.reshape %115 : (tensor<1x1x64xf32>) -> tensor<1x64xf32>
    %117 = stablehlo.slice %112 [0:1, 0:1, 0:128] : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %118 = stablehlo.reshape %117 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %119 = stablehlo.slice %113 [0:1, 0:1] : (tensor<1x1xi32>) -> tensor<1x1xi32>
    %120 = stablehlo.reshape %119 : (tensor<1x1xi32>) -> tensor<1xi32>
    return %46, %50, %114, %116, %118, %120 : tensor<1x2x32x8xf32>, tensor<1x2x32x8xf32>, tensor<1x1x64xf32>, tensor<1x64xf32>, tensor<1x128xf32>, tensor<1xi32>
  }
  func.func private @_where_0(%arg0: tensor<1x1x1x32xi1>, %arg1: tensor<1x8x1x32xf32>, %arg2: tensor<f32>) -> tensor<1x8x1x32xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x1x32xi1>) -> tensor<1x8x1x32xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x8x1x32xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<1x8x1x32xi1>, tensor<1x8x1x32xf32>
    return %3 : tensor<1x8x1x32xf32>
  }
  func.func private @silu_1(%arg0: tensor<1x1x256xf32>) -> tensor<1x1x256xf32> {
    %0 = stablehlo.negate %arg0 : tensor<1x1x256xf32>
    %1 = stablehlo.exponential %0 : tensor<1x1x256xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x256xf32>
    %3 = stablehlo.add %2, %1 : tensor<1x1x256xf32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x256xf32>
    %5 = stablehlo.divide %4, %3 : tensor<1x1x256xf32>
    %6 = stablehlo.multiply %arg0, %5 : tensor<1x1x256xf32>
    return %6 : tensor<1x1x256xf32>
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
    %4 = call @_where_2(%1, %3, %arg1) : (tensor<1x1xi1>, tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
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
  func.func private @_where_2(%arg0: tensor<1x1xi1>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x1xi32>) -> tensor<1x1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x1xi1>, tensor<1x1xi32>
    return %0 : tensor<1x1xi32>
  }
}

