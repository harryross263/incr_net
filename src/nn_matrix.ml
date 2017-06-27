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

(* CR hross: Randomly initialise matrix entries. *)
let create kind ~dimx ?dimy () =
  let contents, derivative =
    let c, d =
      match dimx, dimy with
      | _, None ->
        begin
          let contents = Float_vector (Array.create ~len:dimx 1.) in
          let derivative = Float_vector (Array.create ~len:dimx 0.) in
          contents, derivative
        end
      | _, Some dimy ->
        begin
          let contents = Float_matrix (Array.make_matrix ~dimx ~dimy 1.) in
          let derivative = Float_matrix (Array.make_matrix ~dimx ~dimy 0.) in
          contents, derivative
        end
    in
    match kind, c with
    | `Incr, Float_vector vec -> Incr_vector (to_vars vec), d
    | `Incr, Float_matrix mat -> Incr_matrix (to_vars' mat), d
    | `Incr, _ -> c, d
    | `Float, _ -> c, d
  in
  { kind;
    contents;
    derivative
  }
;;
