open Core.Std
open Import

val apply_weights :
  inputs:float Incr.t Array.t
  -> weights: float Incr.t Array.t Array.t
  -> ?activation_fn: (float Incr.t -> float Incr.t)
  -> unit
  -> float Incr.t Array.t
