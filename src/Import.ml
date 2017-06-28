open Core.Std

open Incremental_kernel
module Incr = Incremental.Make ()

let to_vars arr            = Array.map arr ~f:Incr.Var.create
let to_vars' arr           = Array.map arr ~f:to_vars
let to_incrs arr           = Array.map arr ~f:Incr.Var.watch
let to_incrs' arr          = Array.map arr ~f:to_incrs
let observer_value_exn arr = Array.map arr ~f:Incr.Observer.value_exn

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
