open Core.Std
open Import

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

let run depth =
  let num_iters = 1000 in
  let learning_rate = 0.002 in
  let input_dim = 784 in
  let hidden_dim = 32 in
  let output_dim = input_dim in
  let x, iter = setup_training_data ~input_dim in
  let w1 = Nn_matrix.create `Float ~dimx:hidden_dim ~dimy:input_dim () in
  let weights =
    Array.create ~len:depth w1
    |> Array.mapi ~f:(fun i w1 ->
        if phys_equal i 0
        then w1
        else
          Nn_matrix.create `Float ~dimx:hidden_dim ~dimy:hidden_dim ()
      )
  in
  let out_weights = Nn_matrix.create `Float ~dimx:output_dim ~dimy:hidden_dim () in
  let deltas = Array.create ~len:num_iters (Core.Span.of_ns 0.) in
  while !iter < num_iters do
    Nn_matrix.fill_in_place_next_training_example ~vec:x ~iter;
    let beg = Time.now () in
    let inputs, graph =
      Array.fold weights ~init:(x, []) ~f:(fun (inputs, graph) weights ->
          let inputs, graph =
            Nn_matrix.mat_vec_mul
              graph
              ~mat:weights
              ~vec:inputs
          in
          Nn_matrix.relu graph ~vec:inputs
        )
    in
    let outputs, graph =
      Nn_matrix.mat_vec_mul
        graph
        ~mat:out_weights
        ~vec:inputs
    in
    Array.set deltas (!iter - 1) (Time.diff (Time.now ()) beg);
    Graph.backprop graph;
    Nn_matrix.update_and_reset (out_weights :: Array.to_list weights);
  done;
  let coefficient = 1. /. (Float.of_int num_iters) in
  let avg = Array.fold deltas ~init:0. ~f:(fun acc delta ->
      let delta = Core.Span.to_float delta in
      acc +. (coefficient *. delta)
    )
  in
  Printf.printf "Depth: %d Average time taken for one example: %f\n" depth avg;
  print_newline ();
  avg
;;

let () =
  let depths = [3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13; 14; 15; 16; 17; 18; 19; 20] in
  List.map depths ~f:run
  |> List.iter ~f:(Printf.printf "%f\n")
;;
