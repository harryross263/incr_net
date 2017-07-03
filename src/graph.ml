open! Core.Std

type t = (unit -> unit) list

let rec backprop t =
  match t with
  | [] -> ()
  | hd :: rest -> hd (); backprop rest
;;
