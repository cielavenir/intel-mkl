.PHONY: default

BIN= catalog test-gemm

default: $(BIN)

%: %.cc
	g++ $@.cc -o $@ $(shell pkg-config --libs --cflags mkl-dynamic-ilp64-iomp)

clean:
	-$(RM) $(BIN)
