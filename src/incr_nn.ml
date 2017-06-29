open Core.Std
open Import
open Nn_matrix

let setup_training_data ~input_dim =
  let iter = ref 0 in
  let inputs = Nn_matrix.create `Incr_var ~dimx:input_dim () in
  Nn_matrix.fill_in_place_next_training_example ~vec:inputs ~iter;
  inputs, iter
;;

let () =
  let learning_rate = 0.002 in
  let input_dim = 64 in
  let hidden_dim = 32 in
  let output_dim = input_dim in
  let inputs, iter = setup_training_data ~input_dim in
  let l1_weights = Nn_matrix.create `Incr_var ~dimx:hidden_dim ~dimy:input_dim () in
  let l2_weights = Nn_matrix.create `Incr_var ~dimx:output_dim ~dimy:hidden_dim () in
  let graph = [] in
  let hidden_preactivations, backprop =
    Nn_matrix.mat_vec_mul
      ~mat:(Nn_matrix.var_to_incrs l1_weights)
      ~vec:(Nn_matrix.var_to_incrs inputs)
  in
  let graph = backprop :: graph in
  let hidden_activations, backprop =
    Nn_matrix.relu
      ~vec:hidden_preactivations
  in
  let graph = backprop :: graph in
  let y_pred, backprop =
    Nn_matrix.mat_vec_mul
      ~mat:(Nn_matrix.var_to_incrs l2_weights)
      ~vec:hidden_activations
  in
  let graph = backprop :: graph in
    (* Train the network by simply presenting different inputs and targets. *)
  while !iter < 100000 do
    Nn_matrix.fill_in_place_next_training_example ~vec:inputs ~iter;
    Incr.stabilize ();
    Graph.backprop graph;
    Nn_matrix.update_and_reset [l1_weights; l2_weights];
    if phys_equal (!iter % 10) 0
    then
      Printf.printf "iter %d\n" !iter
    done;
;;
