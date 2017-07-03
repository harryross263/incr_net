open Core.Std
open Import
open Nn_matrix

let setup_training_data ~input_dim =
  let iter = ref 0 in
  let inputs = Nn_matrix.create `Incr_var ~dimx:input_dim () in
  Nn_matrix.fill_in_place_next_training_example ~vec:inputs ~iter;
  inputs, iter
;;

let sse ~pred ~targets =
  let targets =
    match Nn_matrix.contents targets with
    | Incr_var_vector targets -> targets
    | _ -> failwith "SSE must be of an observer and an incr_var vectors"
  in
  Array.map2_exn pred targets ~f:(fun pred target ->
      ((Incr.Observer.value_exn pred) -. (Incr.Var.value target)) ** 2.
    )
  |> Array.fold ~init:0. ~f:(+.)
;;

let () =
  let learning_rate = 0.002 in
  let input_dim = 784 in
  let hidden_dim = 32 in
  let output_dim = input_dim in
  let inputs, iter = setup_training_data ~input_dim in
  let l1_weights = Nn_matrix.create `Incr_var ~dimx:hidden_dim ~dimy:input_dim () in
  let l2_weights = Nn_matrix.create `Incr_var ~dimx:output_dim ~dimy:hidden_dim () in
  let hidden_preactivations, hidden_pre_backprop =
    Nn_matrix.mat_vec_mul
      ~mat:(Nn_matrix.var_to_incrs l1_weights)
      ~vec:(Nn_matrix.var_to_incrs inputs)
  in
  let hidden_activations, hidden_backprop =
    Nn_matrix.relu
      ~vec:hidden_preactivations
  in
  let y_pred, out_backprop =
    Nn_matrix.mat_vec_mul
      ~mat:(Nn_matrix.var_to_incrs l2_weights)
      ~vec:hidden_activations
  in
  let y_pred_observers =
    match Nn_matrix.contents y_pred with
    | Incr_vector vec -> Array.map vec ~f:Incr.observe
    | _ -> failwith "y_pred must be an incr_vector"
  in
  let graph = [out_backprop; hidden_backprop; hidden_pre_backprop] in
  (* Train the network by simply presenting different inputs and targets. *)
  while !iter < 100000 do
    Nn_matrix.fill_in_place_next_training_example ~vec:inputs ~iter;
    Incr.stabilize ();
    Graph.backprop graph;
    Nn_matrix.update_and_reset [l1_weights; l2_weights];
    if phys_equal (!iter % 10) 0
    then
      begin
        Printf.printf "iter %d, sse: %f" !iter (sse ~pred:y_pred_observers ~targets:inputs);
        print_newline () (* Flush stdout. *)
      end
    done;
;;
