open! Core.Std

(* Functions to execute on each backwards pass that fill in the appropriate vectors
 * and matrices for in order to calculate the next derivative.
 *
 * The list acts a stack, where we push and pop directly from the head.
 * *)
type t = (unit -> unit) list

val backprop : t -> unit

