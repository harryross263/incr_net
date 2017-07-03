open Core.Std

open Incremental_kernel
module Incr = Incremental.Make ()

let to_vars arr   = Array.map arr ~f:Incr.Var.create
let to_vars' arr  = Array.map arr ~f:to_vars
let to_incrs arr  = Array.map arr ~f:Incr.Var.watch
let to_incrs' arr = Array.map arr ~f:to_incrs
