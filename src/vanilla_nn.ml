open Core.Std
open Import
open Nn_matrix

let setup_training_data ~input_dim =
  let iter = ref 0 in
  let inputs = Nn_matrix.create `Float ~dimx:input_dim () in
  Nn_matrix.fill_in_place_next_training_example ~vec:inputs ~iter;
  inputs, iter
;;

let sse ~pred ~targets =
  let pred, targets =
    match (Nn_matrix.contents pred), (Nn_matrix.contents targets) with
    | Float_vector pred, Float_vector targets -> pred, targets
    | _ -> failwith "SSE must be of float_vectors"
  in
  Array.map2_exn pred targets ~f:(fun pred target ->
      (pred -. target) ** 2.
    )
  |> Array.fold ~init:0. ~f:(+.)
;;

let () =
  let learning_rate = 0.002 in
  let input_dim = 784 in
  let hidden_dim = 32 in
  let output_dim = input_dim in
  let inputs, iter = setup_training_data ~input_dim in
  let l1_weights = Nn_matrix.create `Float ~dimx:hidden_dim ~dimy:input_dim () in
  let l2_weights = Nn_matrix.create `Float ~dimx:output_dim ~dimy:hidden_dim () in
  while !iter < 100000 do
    Nn_matrix.fill_in_place_next_training_example ~vec:inputs ~iter;
    let hidden_preactivations, hidden_pre_backprop =
      Nn_matrix.mat_vec_mul
        ~mat:l1_weights
        ~vec:inputs
    in
    let hidden_activations, hidden_backprop =
      Nn_matrix.relu
        ~vec:hidden_preactivations
    in
    let y_pred, out_backprop =
      Nn_matrix.mat_vec_mul
        ~mat:l2_weights
        ~vec:hidden_activations
    in
    let graph = [out_backprop; hidden_backprop; hidden_pre_backprop] in
    Graph.backprop graph;
    Nn_matrix.update_and_reset [l1_weights; l2_weights];
    if phys_equal (!iter % 10) 0
    then
      begin
        Printf.printf "iter %d, sse: %f" !iter (sse ~pred:y_pred ~targets:inputs);
        print_newline () (* Flush stdout. *)
      end
  done
;;
