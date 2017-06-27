open Core.Std

open Incremental_kernel
module Incr = Incremental.Make ()

let to_vars arr            = Array.map arr ~f:Incr.Var.create
let to_vars' arr           = Array.map arr ~f:to_vars
let to_incrs arr           = Array.map arr ~f:Incr.Var.watch
let to_incrs' arr          = Array.map arr ~f:to_incrs
let observer_value_exn arr = Array.map arr ~f:Incr.Observer.value_exn


