open Core.Std
open Import
open Nn_matrix

type network_vars = {
  input_vars  : float Incr.Var.t Array.t;
  target_vars : float Incr.Var.t Array.t;
}

let incr_relu = Incr.map ~f:(Float.max 0.)

let drelu = function
  | 0. -> 0.
  | _ -> 1.
;;

(* Returns an array containing the next training example.
 *
 * For this contrived example, we simple "snake" a line across
 * the image as the iterations increase.
 * *)
let next_training_example ~iter ~input_dim =
  incr iter;
  let fill_line image =
    let offset = !iter % input_dim in
    let dim = sqrt (Int.to_float input_dim) |> Float.to_int in
    for i = offset to Int.min ((Array.length image) - 1) (offset + dim) do
      Array.set image i 2.;
    done;
    image
  in
  Array.create ~len:input_dim 0. |> fill_line
;;

let present_next_example ~iter ~input_dim { input_vars; target_vars } =
  let next_example = next_training_example ~iter ~input_dim in
  (* We're constructing an autoencoder: inputs = outputs. *)
  Array.iter2_exn input_vars next_example ~f:Incr.Var.set;
  Array.iter2_exn target_vars next_example ~f:Incr.Var.set
;;

let setup_training_data ~input_dim =
  let iter = ref 0 in
  let first_example = next_training_example ~iter ~input_dim in
  (* [inputs] = [outputs] because we're considering the case of an autoencoder. *)
  let network_vars = {
    input_vars  = to_vars first_example;
    target_vars = to_vars first_example
  }
  in
  network_vars, iter
;;

(* Takes a weights matrix and applies it to the input incrs. *)
let apply_weights ~inputs ~weights ?activation_fn () =
  let activation_fn =
    match activation_fn with
    | None -> Fn.id
    | Some fn -> fn
  in
  Array.map weights ~f:(fun weight_vector ->
      (* Calculate the individual components in the dot product. *)
      Array.map2_exn weight_vector inputs ~f:(Incr.map2 ~f:( *.))
      |> Incr.sum ~zero:0. ~add:(+.) ~sub:(-.) (* Sum the dot product. *)
      |> activation_fn
    )
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
  let network_vars, iter = setup_training_data ~input_dim in
  let { input_vars; target_vars } = network_vars in
  let l1_weights_vars =
    Array.make_matrix ~dimx:hidden_dim ~dimy:input_dim 1.
    |> to_vars'
  in
  let l2_weights_vars =
    Array.make_matrix ~dimx:output_dim ~dimy:hidden_dim 1.
    |> to_vars'
  in
  let l1_weights = to_incrs' l1_weights_vars in
  let l2_weights = to_incrs' l2_weights_vars in
  let inputs = to_incrs input_vars in
  let hidden_activations =
    apply_weights
      ~inputs
      ~weights:l1_weights
      ~activation_fn:incr_relu
      ()
  in
  let y_pred =
    apply_weights
      ~inputs:hidden_activations
      ~weights:l2_weights
      ()
    |> Array.map ~f:Incr.observe
  in
  let hidden_activation_observers = Array.map hidden_activations ~f:Incr.observe in
  (* Train the network by simply presenting different inputs and targets. *)
  while !iter < 100000 do
    present_next_example ~iter ~input_dim network_vars;
    Incr.stabilize ();
    let hidden_activations = observer_value_exn hidden_activation_observers in
    let pred = observer_value_exn y_pred in
    let targets = Array.map target_vars ~f:Incr.Var.value in
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
