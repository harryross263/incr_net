# - The -I flag introduces sub-directories
# - -use-ocamlfind is required to find packages (from Opam)
# - _tags file introduces packages, bin_annot flag for tool chain

OCB_FLAGS = -use-ocamlfind -I src -I lib
OCB = 		ocamlbuild $(OCB_FLAGS)

all: 		native byte # profile debug

incr_nn:
			$(OCB) incr_nn.native
			$(OCB) incr_nn.byte

vanilla_nn:
			$(OCB) vanilla_nn.native
			$(OCB) vanilla_nn.byte

nn_test:
			$(OCB) neural_net.native
			$(OCB) neural_net.byte

clean:
			$(OCB) -clean

native: 	sanity
			$(OCB) main.native

byte:		sanity
			$(OCB) main.byte

profile: 	sanity
			$(OCB) -tag profile main.native

debug: 		sanity
			$(OCB) -tag debug main.byte

sanity:
			# check that packages can be found
			ocamlfind query core
			ocamlfind query incremental

.PHONY: 	all clean byte native profile debug sanity test
