open Core.Std
open Import

type mat_kind = [ `Float | `Incr ]

type contents_kind =
  | Float_vector of float Array.t
  | Incr_vector of float Incr.Var.t Array.t
  | Float_matrix of float Array.t Array.t
  | Incr_matrix of float Incr.Var.t Array.t Array.t

type t = {
  kind       : mat_kind;
  contents   : contents_kind;
  derivative : contents_kind
} [@@deriving fields]

val create : mat_kind -> dimx:int -> ?dimy:int -> unit -> t

