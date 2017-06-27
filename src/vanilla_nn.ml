open Owl
open Algodiff.S

type layer = {
  mutable weights : t;
  mutable bias    : t;
  activation_fn   : t -> t;
}

type network = { layers : layer array }


let run_layer inputs layer =
  let weights = layer.weights in
  let bias = layer.bias in
  Maths.((inputs *@ weights) + bias)
  |> layer.activation_fn
;;

let forward_pass inputs network =
  Array.fold_left run_layer inputs network.layers
;;

let backprop network eta inputs targets =
  let t = tag () in
  Array.iter (fun layer ->
    layer.weights <- make_reverse layer.weights t;
    layer.bias <- make_reverse layer.bias t;
  ) network.layers;
  let loss =
    let y_pred = forward_pass inputs network in
    Maths.(cross_entropy targets y_pred / (F (Mat.row_num targets |> float_of_int)))
  in
  reverse_prop (F 1.) loss;
  Array.iter (fun layer ->
      layer.weights <- Maths.((primal layer.weights) - (eta * (adjval layer.weights))) |> primal;
      layer.bias <- Maths.((primal layer.bias) - (eta * (adjval layer.bias))) |> primal;
    ) network.layers;
  loss |> unpack_flt

let test_model nn x y =
  Mat.iter2_rows (fun u v ->
    Dataset.print_mnist_image (unpack_mat u);
    let p = forward_pass u nn |> unpack_mat in
    Dense.Matrix.Generic.print p;
    Printf.printf "prediction: %i\n" (let _, _, j = Dense.Matrix.Generic.max_i p in j)
  ) x y


let l0 = {
  weights = Maths.(Mat.uniform 784 300 * F 0.15 - F 0.075);
  bias = Mat.zeros 1 300;
  activation_fn = Maths.tanh;
}

let l1 = {
  weights = Maths.(Mat.uniform 300 10 * F 0.15 - F 0.075);
  bias = Mat.zeros 1 10;
  activation_fn = Mat.map_by_row Maths.softmax;
}

let nn = { layers = [|l0; l1|] }

let _ =
  let x, _, y = Dataset.load_mnist_train_data () in
  for i = 1 to 1000 do
    let x', y' = Dataset.draw_samples x y 100 in
    backprop nn (F 0.01) (Mat x') (Mat y')
    |> Printf.printf "#%i : loss = %g\n" i
    |> flush_all;
  done;
  let x, y, _ = Dataset.load_mnist_test_data () in
  let x, y = Dataset.draw_samples x y 10 in
  test_model nn (Mat x) (Mat y)
