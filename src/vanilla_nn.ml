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

let () =
  let learning_rate = 0.002 in
  let input_dim = 784 in
  let hidden_dim = 32 in
  let output_dim = input_dim in
  let x, iter = setup_training_data ~input_dim in
  let w1 = Nn_matrix.create `Float ~dimx:hidden_dim ~dimy:input_dim () in
  let w2 = Nn_matrix.create `Float ~dimx:hidden_dim ~dimy:hidden_dim () in
  let w3 = Nn_matrix.create `Float ~dimx:hidden_dim ~dimy:hidden_dim () in
  let out_weights = Nn_matrix.create `Float ~dimx:output_dim ~dimy:hidden_dim () in
  while !iter < 100000 do
    Nn_matrix.fill_in_place_next_training_example ~vec:x ~iter;
    let inputs, graph =
      Nn_matrix.mat_vec_mul
        []
        ~mat:w1
        ~vec:x
    in
    let inputs, graph = Nn_matrix.relu graph ~vec:inputs in
    let inputs, graph =
      Nn_matrix.mat_vec_mul
        graph
        ~mat:w2
        ~vec:inputs
    in
    let inputs, graph = Nn_matrix.relu graph ~vec:inputs in
    let inputs, graph =
      Nn_matrix.mat_vec_mul
        graph
        ~mat:w3
        ~vec:inputs
    in
    let inputs, graph = Nn_matrix.relu graph ~vec:inputs in
    let outputs, graph =
      Nn_matrix.mat_vec_mul
        graph
        ~mat:out_weights
        ~vec:inputs
    in
    Graph.backprop graph;
    Nn_matrix.update_and_reset [w1; w2; w3; out_weights];
    if phys_equal (!iter % 10) 0
    then
      begin
        Printf.printf "iter %d, sse: %f" !iter (sse ~pred:outputs ~targets:x);
        print_newline () (* Flush stdout. *)
      end
  done
;;
