open Core.Std
open Import
open Nn_matrix

type network_vars = {
  input_vars  : Nn_matrix.t;
  target_vars : Nn_matrix.t
}

let incr_relu = Incr.map ~f:(Float.max 0.)

let drelu = function
  | 0. -> 0.
  | _ -> 1.
;;

let setup_training_data ~input_dim =
  let iter = ref 0 in
  let input_vec = Nn_matrix.create `Incr_var ~dimx:input_dim () in
  Nn_matrix.fill_in_place_next_training_example ~vec:input_vec ~iter;
  input_vec, iter
;;

let update_weights weights inputs eta deltas =
  Array.iteri weights ~f:(fun i weight_vector ->
      let delta = Array.get deltas i in
      Array.iteri weight_vector ~f:(fun j weight_var ->
          let input = Array.get inputs j in
          let weight = Incr.Var.value weight_var in
          let new_val = weight +. eta *. input *. delta in
          Incr.Var.set weight_var new_val;
        );
    );
;;

let sse pred targets =
  Array.map2_exn pred targets ~f:(-.)
  |> Array.map ~f:(fun elt -> elt ** 2.)
  |> Array.fold ~init:0. ~f:(+.)
;;

let backprop
    ~pred
    ~targets
    ~inputs
    ~hidden_activations
    ~l1_weights_vars
    ~l2_weights_vars
    ~learning_rate
  =
  let out_deltas = Array.map2_exn targets pred ~f:Float.sub in
  (* Update weights from hidden to output. *)
  update_weights l2_weights_vars hidden_activations learning_rate out_deltas;
  let hidden_deltas =
    Array.map l1_weights_vars ~f:(fun weight_vector ->
        Array.map2_exn weight_vector out_deltas ~f:(fun weight_var delta ->
            let weight = Incr.Var.value weight_var in
            delta *. weight
          )
        |> Array.fold ~init:0. ~f:(+.)
      )
    |> Array.mapi ~f:(fun i delta ->
        delta *. (drelu (Array.get hidden_activations i))
      )
  in
  update_weights l1_weights_vars inputs learning_rate hidden_deltas;
;;

let () =
  let learning_rate = 0.002 in
  let input_dim = 64 in
  let hidden_dim = 32 in
  let output_dim = input_dim in
  let inputs, iter = setup_training_data ~input_dim in
  let l1_weights = Nn_matrix.create `Float ~dimx:hidden_dim ~dimy:input_dim () in
  let l2_weights = Nn_matrix.create `Float ~dimx:output_dim ~dimy:hidden_dim () in
  let hidden_activations =
    Nn_matrix.mat_vec_mul
      ~mat:l1_weights
      ~vec:inputs
  in
  let y_pred =
    Nn_matrix.mat_vec_mul
      ~mat:l2_weights
      ~vec:hidden_activations
  in
  (* Train the network by simply presenting different inputs and targets. *)
  while !iter < 100000 do
    Nn_matrix.fill_in_place_next_training_example ~vec:input_vars ~iter
    Incr.stabilize ();
    backprop
      ~pred
      ~targets
      ~inputs:(Array.map input_vars ~f:Incr.Var.value)
      ~hidden_activations
      ~l1_weights_vars
      ~l2_weights_vars
      ~learning_rate;
    if phys_equal (!iter % 10) 0
    then
      Printf.printf "iter %d, SSE %f\n" !iter (sse pred targets)
  done;
