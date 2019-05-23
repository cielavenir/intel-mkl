.PHONY: default

BIN= catalog test-gemm test-gesvd

default: $(BIN)

%: %.cc
	g++ $@.cc -o $@ $(shell pkg-config --libs --cflags mkl-dynamic-ilp64-iomp)

test: default
	./catalog
	M=512 ./test-gemm
	M=512 ./test-gesvd

clean:
	-$(RM) $(BIN)
