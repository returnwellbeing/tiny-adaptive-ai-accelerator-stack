module @jit_decoder_layer_prefill_decode attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x16x64xf32>, %arg1: tensor<1x1x64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64x64xf32>, %arg6: tensor<64x16xf32>, %arg7: tensor<64x16xf32>, %arg8: tensor<64x64xf32>, %arg9: tensor<64x128xf32>, %arg10: tensor<128x64xf32>, %arg11: tensor<64x256xf32>, %arg12: tensor<64x256xf32>, %arg13: tensor<256x64xf32>, %arg14: tensor<16x8xf32>, %arg15: tensor<16x8xf32>, %arg16: tensor<1x8xf32>, %arg17: tensor<1x8xf32>, %arg18: tensor<1x2x32x8xf32>, %arg19: tensor<1x2x32x8xf32>, %arg20: tensor<i32>) -> (tensor<1x16x64xf32> {jax.result_info = "[0]"}, tensor<1x1x64xf32> {jax.result_info = "[1]"}, tensor<1x1x128xf32> {jax.result_info = "[2]"}, tensor<1x1xi32> {jax.result_info = "[3]"}, tensor<1x1x64xf32> {jax.result_info = "[4]"}, tensor<1x2x32x8xf32> {jax.result_info = "[5]"}, tensor<1x2x32x8xf32> {jax.result_info = "[6]"}) {
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
    %97 = stablehlo.multiply %arg1, %arg1 : tensor<1x1x64xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %98 = stablehlo.reduce(%97 init: %cst_8) applies stablehlo.add across dimensions = [2] : (tensor<1x1x64xf32>, tensor<f32>) -> tensor<1x1xf32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %100 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %101 = stablehlo.divide %99, %100 : tensor<1x1x1xf32>
    %102 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %103 = stablehlo.add %101, %102 : tensor<1x1x1xf32>
    %104 = stablehlo.rsqrt %103 : tensor<1x1x1xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x64xf32>
    %106 = stablehlo.multiply %arg1, %105 : tensor<1x1x64xf32>
    %107 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %108 = stablehlo.multiply %106, %107 : tensor<1x1x64xf32>
    %109 = stablehlo.dot_general %108, %arg5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x64xf32>) -> tensor<1x1x64xf32>
    %110 = stablehlo.dot_general %108, %arg6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x16xf32>) -> tensor<1x1x16xf32>
    %111 = stablehlo.dot_general %108, %arg7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x16xf32>) -> tensor<1x1x16xf32>
    %112 = stablehlo.reshape %109 : (tensor<1x1x64xf32>) -> tensor<1x1x8x8xf32>
    %113 = stablehlo.reshape %110 : (tensor<1x1x16xf32>) -> tensor<1x1x2x8xf32>
    %114 = stablehlo.reshape %111 : (tensor<1x1x16xf32>) -> tensor<1x1x2x8xf32>
    %115 = stablehlo.transpose %112, dims = [0, 2, 1, 3] : (tensor<1x1x8x8xf32>) -> tensor<1x8x1x8xf32>
    %116 = stablehlo.transpose %113, dims = [0, 2, 1, 3] : (tensor<1x1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %117 = stablehlo.transpose %114, dims = [0, 2, 1, 3] : (tensor<1x1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %118 = stablehlo.broadcast_in_dim %arg16, dims = [2, 3] : (tensor<1x8xf32>) -> tensor<1x1x1x8xf32>
    %119 = stablehlo.broadcast_in_dim %arg17, dims = [2, 3] : (tensor<1x8xf32>) -> tensor<1x1x1x8xf32>
    %120 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xf32>) -> tensor<1x8x1x8xf32>
    %121 = stablehlo.multiply %115, %120 : tensor<1x8x1x8xf32>
    %122 = stablehlo.slice %115 [0:1, 0:8, 0:1, 0:4] : (tensor<1x8x1x8xf32>) -> tensor<1x8x1x4xf32>
    %123 = stablehlo.slice %115 [0:1, 0:8, 0:1, 4:8] : (tensor<1x8x1x8xf32>) -> tensor<1x8x1x4xf32>
    %124 = stablehlo.negate %123 : tensor<1x8x1x4xf32>
    %125 = stablehlo.concatenate %124, %122, dim = 3 : (tensor<1x8x1x4xf32>, tensor<1x8x1x4xf32>) -> tensor<1x8x1x8xf32>
    %126 = stablehlo.broadcast_in_dim %119, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xf32>) -> tensor<1x8x1x8xf32>
    %127 = stablehlo.multiply %125, %126 : tensor<1x8x1x8xf32>
    %128 = stablehlo.add %121, %127 : tensor<1x8x1x8xf32>
    %129 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xf32>) -> tensor<1x2x1x8xf32>
    %130 = stablehlo.multiply %116, %129 : tensor<1x2x1x8xf32>
    %131 = stablehlo.slice %116 [0:1, 0:2, 0:1, 0:4] : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x4xf32>
    %132 = stablehlo.slice %116 [0:1, 0:2, 0:1, 4:8] : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x4xf32>
    %133 = stablehlo.negate %132 : tensor<1x2x1x4xf32>
    %134 = stablehlo.concatenate %133, %131, dim = 3 : (tensor<1x2x1x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x1x8xf32>
    %135 = stablehlo.broadcast_in_dim %119, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xf32>) -> tensor<1x2x1x8xf32>
    %136 = stablehlo.multiply %134, %135 : tensor<1x2x1x8xf32>
    %137 = stablehlo.add %130, %136 : tensor<1x2x1x8xf32>
    %138 = stablehlo.compare  LT, %arg20, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_9 = stablehlo.constant dense<32> : tensor<i32>
    %139 = stablehlo.add %arg20, %c_9 : tensor<i32>
    %140 = stablehlo.select %138, %139, %arg20 : tensor<i1>, tensor<i32>
    %141 = stablehlo.dynamic_update_slice %95, %137, %c, %c, %140, %c : (tensor<1x2x32x8xf32>, tensor<1x2x1x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x32x8xf32>
    %142 = stablehlo.compare  LT, %arg20, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %143 = stablehlo.add %arg20, %c_9 : tensor<i32>
    %144 = stablehlo.select %142, %143, %arg20 : tensor<i1>, tensor<i32>
    %145 = stablehlo.dynamic_update_slice %96, %117, %c, %c, %144, %c : (tensor<1x2x32x8xf32>, tensor<1x2x1x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x32x8xf32>
    %146 = stablehlo.slice %141 [0:1, 0:2, 0:17, 0:8] : (tensor<1x2x32x8xf32>) -> tensor<1x2x17x8xf32>
    %147 = stablehlo.slice %145 [0:1, 0:2, 0:17, 0:8] : (tensor<1x2x32x8xf32>) -> tensor<1x2x17x8xf32>
    %148 = stablehlo.broadcast_in_dim %146, dims = [0, 1, 3, 4] : (tensor<1x2x17x8xf32>) -> tensor<1x2x1x17x8xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x17x8xf32>) -> tensor<1x2x4x17x8xf32>
    %150 = stablehlo.reshape %149 : (tensor<1x2x4x17x8xf32>) -> tensor<1x8x17x8xf32>
    %151 = stablehlo.broadcast_in_dim %147, dims = [0, 1, 3, 4] : (tensor<1x2x17x8xf32>) -> tensor<1x2x1x17x8xf32>
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x17x8xf32>) -> tensor<1x2x4x17x8xf32>
    %153 = stablehlo.reshape %152 : (tensor<1x2x4x17x8xf32>) -> tensor<1x8x17x8xf32>
    %154 = stablehlo.dot_general %128, %150, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1x8xf32>, tensor<1x8x17x8xf32>) -> tensor<1x8x1x17xf32>
    %155 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x1x17xf32>
    %156 = stablehlo.divide %154, %155 : tensor<1x8x1x17xf32>
    %cst_10 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %157 = stablehlo.reduce(%156 init: %cst_10) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x1x17xf32>, tensor<f32>) -> tensor<1x8x1xf32>
    %158 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %159 = stablehlo.maximum %158, %157 : tensor<1x8x1xf32>
    %160 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1x1xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2, 3] : (tensor<1x8x1x1xf32>) -> tensor<1x8x1x17xf32>
    %162 = stablehlo.subtract %156, %161 : tensor<1x8x1x17xf32>
    %163 = stablehlo.exponential %162 : tensor<1x8x1x17xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %164 = stablehlo.reduce(%163 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x8x1x17xf32>, tensor<f32>) -> tensor<1x8x1xf32>
    %165 = stablehlo.broadcast_in_dim %164, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1x1xf32>
    %166 = stablehlo.broadcast_in_dim %165, dims = [0, 1, 2, 3] : (tensor<1x8x1x1xf32>) -> tensor<1x8x1x17xf32>
    %167 = stablehlo.divide %163, %166 : tensor<1x8x1x17xf32>
    %168 = stablehlo.dot_general %167, %153, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1x17xf32>, tensor<1x8x17x8xf32>) -> tensor<1x8x1x8xf32>
    %169 = stablehlo.transpose %168, dims = [0, 2, 1, 3] : (tensor<1x8x1x8xf32>) -> tensor<1x1x8x8xf32>
    %170 = stablehlo.reshape %169 : (tensor<1x1x8x8xf32>) -> tensor<1x1x64xf32>
    %171 = stablehlo.dot_general %170, %arg8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x64xf32>) -> tensor<1x1x64xf32>
    %172 = stablehlo.add %arg1, %171 : tensor<1x1x64xf32>
    %173 = stablehlo.multiply %172, %172 : tensor<1x1x64xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %174 = stablehlo.reduce(%173 init: %cst_12) applies stablehlo.add across dimensions = [2] : (tensor<1x1x64xf32>, tensor<f32>) -> tensor<1x1xf32>
    %175 = stablehlo.broadcast_in_dim %174, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %176 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %177 = stablehlo.divide %175, %176 : tensor<1x1x1xf32>
    %178 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %179 = stablehlo.add %177, %178 : tensor<1x1x1xf32>
    %180 = stablehlo.rsqrt %179 : tensor<1x1x1xf32>
    %181 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x64xf32>
    %182 = stablehlo.multiply %172, %181 : tensor<1x1x64xf32>
    %183 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %184 = stablehlo.multiply %182, %183 : tensor<1x1x64xf32>
    %185 = stablehlo.dot_general %184, %arg11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x256xf32>) -> tensor<1x1x256xf32>
    %186 = stablehlo.dot_general %184, %arg12, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x256xf32>) -> tensor<1x1x256xf32>
    %187 = call @silu_0(%185) : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %188 = stablehlo.multiply %187, %186 : tensor<1x1x256xf32>
    %189 = stablehlo.dot_general %188, %arg13, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x256xf32>, tensor<256x64xf32>) -> tensor<1x1x64xf32>
    %190 = stablehlo.add %172, %189 : tensor<1x1x64xf32>
    %191 = stablehlo.multiply %190, %190 : tensor<1x1x64xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %192 = stablehlo.reduce(%191 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<1x1x64xf32>, tensor<f32>) -> tensor<1x1xf32>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %194 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %195 = stablehlo.divide %193, %194 : tensor<1x1x1xf32>
    %196 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %197 = stablehlo.add %195, %196 : tensor<1x1x1xf32>
    %198 = stablehlo.rsqrt %197 : tensor<1x1x1xf32>
    %199 = stablehlo.broadcast_in_dim %198, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x64xf32>
    %200 = stablehlo.multiply %190, %199 : tensor<1x1x64xf32>
    %201 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %202 = stablehlo.multiply %200, %201 : tensor<1x1x64xf32>
    %203 = stablehlo.dot_general %202, %arg9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x64xf32>, tensor<64x128xf32>) -> tensor<1x1x128xf32>
    %204 = call @argmax(%203) : (tensor<1x1x128xf32>) -> tensor<1x1xi32>
    %205 = call @_take(%arg10, %204) : (tensor<128x64xf32>, tensor<1x1xi32>) -> tensor<1x1x64xf32>
    return %94, %190, %203, %204, %205, %141, %145 : tensor<1x16x64xf32>, tensor<1x1x64xf32>, tensor<1x1x128xf32>, tensor<1x1xi32>, tensor<1x1x64xf32>, tensor<1x2x32x8xf32>, tensor<1x2x32x8xf32>
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
  func.func private @silu_0(%arg0: tensor<1x1x256xf32>) -> tensor<1x1x256xf32> {
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
    %4 = call @_where_1(%1, %3, %arg1) : (tensor<1x1xi1>, tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
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
  func.func private @_where_1(%arg0: tensor<1x1xi1>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x1xi32>) -> tensor<1x1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x1xi1>, tensor<1x1xi32>
    return %0 : tensor<1x1xi32>
  }
}

