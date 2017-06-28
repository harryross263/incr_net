open Core.Std
open Import

(* CR hross: Need to change these to be optionally Incr.Var.t and Incr.t. *)
type contents_kind =
  | Float_vector of float Array.t
  | Incr_vector of float Incr.Var.t Array.t
  | Float_matrix of float Array.t Array.t
  | Incr_matrix of float Incr.Var.t Array.t Array.t

type t = {
  kind       : [`Float | `Incr];
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
    | _, _ -> c, d
  in
  { kind;
    contents;
    derivative
  }
;;

(* Takes a weights matrix and applies it to the input incrs. *)
let mat_vec_mul ~mat ~vec =
  let float_dot_product vec1 vec2 =
    Array.map2_exn vec1 vec2 ~f:( *.)
    |> Array.fold ~init:0. ~f:(+.)
  in
  let float_matmul mat vec =
    Array.map mat ~f:(float_dot_product vec)
  in
  let incr_dot_product vec1 vec2 =
    Array.map2_exn vec1 vec2 ~f:(Incr.map2 ~f:( *.))
    |> Incr.sum ~zero:0. ~add:(+.) ~sub:(-.)
  in
  let incr_matmul mat vec =
    Array.map mat ~f:(incr_dot_product vec)
  in
  let out_contents =
    match mat.contents, vec.contents with
    | Float_matrix mat, Float_vector vec -> Float_vector (float_matmul mat vec)
    | Incr_matrix mat, Incr_vector vec -> Incr_vector (incr_matmul mat vec)
    | _, _ -> failwith "Cannot multiply nn_matrices of different types"
  in
  let len =
    match out_contents with
    | Float_vector vec
    | Incr_vector vec -> Array.length vec
    | _ -> failwith "Matrix vector multiplication always produces a vector"
  in
  { kind;
    contents;
    derivative = Array.create ~len 0.
  }
;;


