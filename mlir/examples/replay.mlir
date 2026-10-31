// Two updates, with a runtime accepted-count parameter at the history boundary.
func.func @history(%initial: tensor<2x3xf32>,
                   %a0: tensor<f32>, %v0: tensor<2xf32>, %k0: tensor<3xf32>,
                   %a1: tensor<f32>, %v1: tensor<2xf32>, %k1: tensor<3xf32>,
                   %accepted: index) -> tensor<2x3xf32> {
  %h = rewind.initialize_store %initial : tensor<2x3xf32> -> !rewind.store<tensor<2x3xf32>>
  rewind.append %h, %a0, %v0, %k0 : !rewind.store<tensor<2x3xf32>>, tensor<f32>, tensor<2xf32>, tensor<3xf32>
  rewind.append %h, %a1, %v1, %k1 : !rewind.store<tensor<2x3xf32>>, tensor<f32>, tensor<2xf32>, tensor<3xf32>
  %selected = rewind.retrieve %h[%accepted] : !rewind.store<tensor<2x3xf32>> -> tensor<2x3xf32>
  return %selected : tensor<2x3xf32>
}

// The complete replay is [[0.5, 4, 7.5], [9, 9, 9]]. Return element [0,0]
// so mlir-runner can print a small, visible execution result: 0.5.
func.func @main() -> f32 {
  %initial = arith.constant dense<[[1., 2., 3.], [4., 5., 6.]]> : tensor<2x3xf32>
  %a0 = arith.constant dense<2.> : tensor<f32>
  %v0 = arith.constant dense<[1., 2.]> : tensor<2xf32>
  %k0 = arith.constant dense<[3., 4., 5.]> : tensor<3xf32>
  %a1 = arith.constant dense<0.5> : tensor<f32>
  %v1 = arith.constant dense<[-1., 1.]> : tensor<2xf32>
  %k1 = arith.constant dense<[2., 0., -2.]> : tensor<3xf32>
  %accepted = arith.constant 2 : index
  %selected = func.call @history(%initial, %a0, %v0, %k0, %a1, %v1, %k1, %accepted) : (tensor<2x3xf32>, tensor<f32>, tensor<2xf32>, tensor<3xf32>, tensor<f32>, tensor<2xf32>, tensor<3xf32>, index) -> tensor<2x3xf32>
  %zero = arith.constant 0 : index
  %element = tensor.extract %selected[%zero, %zero] : tensor<2x3xf32>
  return %element : f32
}
