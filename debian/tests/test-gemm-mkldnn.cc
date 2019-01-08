// g++ test-gemm-mkldnn.cc -lpthread -lm -ldl -lmkldnn
#include <iostream>
#include <cstdlib>
#include <cassert>

#include <mkldnn.h>
#include <sys/time.h>

const int iteration = 5; // how many iterations would you like to run
const int repeat = 100; // repeat several times in each iteration
const int M = 512; // matrix size (M * M) used for testing
const bool debug = false; // dump the matrices?
const char NoTran = 'N';
const char Tran   = 'T';

#define _GEMM(T) mkldnn_##T##gemm
#define _AXPY(T) cblas_##T##axpy
#define _ASUM(T) cblas_##T##asum

#define PREC_T float
#define GEMM _GEMM(s)

#if !defined BlasInt
#define BlasInt __int32_t
#endif

int
main(void)
{
	struct timeval tv_start, tv_end;

	PREC_T* x = (PREC_T*)malloc(sizeof(PREC_T) * M * M);
	PREC_T* y = (PREC_T*)malloc(sizeof(PREC_T) * M * M);
	PREC_T* z = (PREC_T*)malloc(sizeof(PREC_T) * M * M);
	PREC_T alpha = 1.0;
	PREC_T beta  = 0.0;

	// start iterations
	for (int t = 0; t < iteration; t++) {

		// fill the matrices and run dgemm for several times
		for (int i = 0; i < M*M; i++) {
			x[i] = (PREC_T)drand48(); 
			y[i] = (PREC_T)drand48();
		}

		// run dgemm
		gettimeofday(&tv_start, nullptr);
		for (int i = 0; i < repeat; i++) {
			GEMM(&NoTran, &NoTran, &M, &M, &M, &alpha, x, &M, y, &M, &beta, z, &M);
			GEMM(&Tran, &NoTran, &M, &M, &M, &alpha, x, &M, y, &M, &beta, z, &M);
			GEMM(&NoTran, &Tran, &M, &M, &M, &alpha, x, &M, y, &M, &beta, z, &M);
			GEMM(&Tran, &Tran, &M, &M, &M, &alpha, x, &M, y, &M, &beta, z, &M);
		}
		gettimeofday(&tv_end, nullptr);
		fprintf(stdout, "(%d/%d) Elapsed %.3lf ms\n", t+1, repeat,
				(tv_end.tv_sec*1e6 + tv_end.tv_usec
				 - tv_start.tv_sec*1e6  - tv_start.tv_usec)/1e3);
	}

	return 0;
}
