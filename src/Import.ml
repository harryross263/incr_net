open Core.Std

open Incremental_kernel
module Incr = Incremental.Make ()

let to_vars arr   = Array.map arr ~f:Incr.Var.create
let to_vars' arr  = Array.map arr ~f:to_vars
let to_incrs arr  = Array.map arr ~f:Incr.Var.watch
let to_incrs' arr = Array.map arr ~f:to_incrs

let float_cutoff =
  let should_cutoff ~old_value ~new_value =
    let delta = Float.sub old_value new_value |> Float.abs in
    delta < 0.1
  in
  Incr.Cutoff.create should_cutoff
;;
