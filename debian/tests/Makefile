.PHONY: default

BIN= catalog test-gemm

default: $(BIN)

%: %.cc
	g++ $@.cc -o $@ $(shell pkg-config --libs --cflags mkl-rt)

clean:
	-$(RM) $(BIN)
