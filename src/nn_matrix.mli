open Core.Std
open Import

type contents_kind =
  | Float_vector of float Array.t
  | Incr_vector of float Incr.t Array.t
  | Incr_var_vector of float Incr.Var.t Array.t
  | Float_matrix of float Array.t Array.t
  | Incr_matrix of float Incr.t Array.t Array.t
  | Incr_var_matrix of float Incr.Var.t Array.t Array.t

type derivative_kind =
  | Deriv_vector of float Array.t
  | Deriv_matrix of float Array.t Array.t

type t = {
  kind       : [`Float | `Incr | `Incr_var];
  contents   : contents_kind;
  derivative : derivative_kind
} [@@deriving fields]

val create : [`Float | `Incr | `Incr_var] -> dimx:int -> ?dimy:int -> unit -> t

val relu        : Graph.t -> vec:t -> t * Graph.t
val mat_vec_mul : Graph.t -> mat:t -> vec:t -> t * Graph.t

val update_and_reset                    : learning_rate:float -> t list -> unit
val fill_in_place_next_training_example : vec:t -> iter:int ref -> unit

val var_to_incrs : t -> t
val contents     : t -> contents_kind
