// A handwritten representation example, not output from the PyTorch matcher.
func.func @history(%initial: tensor<4x3xf64>,
                   %a: tensor<f64>, %v: tensor<4xf64>, %k: tensor<3xf64>,
                   %accepted: index) -> tensor<4x3xf64> {
  %h = rewind.initialize_store %initial : tensor<4x3xf64> -> !rewind.store<tensor<4x3xf64>>
  rewind.append %h, %a, %v, %k : !rewind.store<tensor<4x3xf64>>, tensor<f64>, tensor<4xf64>, tensor<3xf64>
  %selected = rewind.retrieve %h[%accepted] : !rewind.store<tensor<4x3xf64>> -> tensor<4x3xf64>
  return %selected : tensor<4x3xf64>
}
