#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <assert.h>

#define OPENBLAS_SO "/usr/lib/x86_64-linux-gnu/libopenblasp-r0.3.3.so"
#define NETLIB_SO   "/usr/lib/x86_64-linux-gnu/blas/libblas.so.3"
#define MKLRT_SO "libmkl_rt.so"

#if !defined(BlasInt)
#define BlasInt __int64_t
#endif

#if !defined(LIBBLAS_SO)
#define LIBBLAS_SO MKLRT_SO
#endif

int
main(int argc, char** argv, char** envp)
{
	/* dynamic loader */
	void *handle = dlopen(LIBBLAS_SO, RTLD_LAZY);
	char *error = NULL;
	if (!handle) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}
	int RowMajor = 101;
	int NoTrans = 111;
	char NoTrans_ = 'N';

	{ /* F77BLAS and CBLAS :: sasum */
		float (*sasum_)(BlasInt*, float*, BlasInt*) = NULL;
		float (*cblas_sasum)(BlasInt, float*, BlasInt) = NULL;

		float X[] = {1., -2., 3.};
		BlasInt n = 3;
		BlasInt incx = 1;
		float tmp = 0.;

		fprintf(stdout, "%s::sasum_ .. ", LIBBLAS_SO);
		sasum_ = (float (*)(BlasInt*, float*, BlasInt*)) dlsym(handle, "sasum_");
		error = dlerror();
		if (error != NULL) {
			//fprintf(stderr, "%s\n", error);
			fprintf(stdout, "NotFound\n");
		} else {
			tmp = sasum_(&n, X, &incx);
			assert(6.00 == tmp);
			fprintf(stdout, "OK\n");
		}

		fprintf(stdout, "%s::cblas_sasum .. ", LIBBLAS_SO);
		cblas_sasum = (float (*)(BlasInt, float*, BlasInt)) dlsym(handle, "cblas_sasum");
		error = dlerror();
		if (error != NULL) {
			fprintf(stdout, "NotFound\n");
		} else {
			tmp = cblas_sasum(n, X, incx);
			assert(6.00 == tmp);
			fprintf(stdout, "OK\n");
		}
	}

	{ /* F77BLAS and CBLAS :: sgemv */
		void (*sgemv_)(const char*, BlasInt*, BlasInt*, float*, float*, BlasInt*, float*, BlasInt*, float*, float*, BlasInt*);
		void (*cblas_sgemv)(int, int, BlasInt, BlasInt, float, float*, BlasInt, float*, BlasInt, float, float*, BlasInt);

		float A[] = {1., 2., 3., 4.};
		BlasInt m = 2;
		BlasInt lda = 2, incx = 1, incy = 1;
		float alpha = 1.0, beta = 0.0;
		float X[] = {1., 1.};
		float Y[] = {0., 0.};

		fprintf(stdout, "%s::sgemv_ .. ", LIBBLAS_SO);
		sgemv_ = (void (*)(const char*, BlasInt*, BlasInt*, float*, float*, BlasInt*, float*, BlasInt*, float*, float*, BlasInt*)) dlsym(handle, "sgemv_");
		error = dlerror();
		if (error != NULL) {
			fprintf(stdout, "NotFound\n");
		} else {
			sgemv_(&NoTrans_, &m, &m, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
			assert(4. == Y[0]);
			assert(6. == Y[1]);
			fprintf(stdout, "OK\n");
		}

		fprintf(stdout, "%s::cblas_sgemv .. ", LIBBLAS_SO);
		cblas_sgemv = (void (*)(int, int, BlasInt, BlasInt, float, float*, BlasInt, float*, BlasInt, float, float*, BlasInt)) dlsym(handle, "cblas_sgemv");
		error = dlerror();
		if (error != NULL) {
			fprintf(stdout, "NotFound\n");
		} else {
			cblas_sgemv(RowMajor, NoTrans, m, m, alpha, A, lda, X, incx, beta, Y, incy);
			assert(3. == Y[0]);
			assert(7. == Y[1]);
			fprintf(stdout, "OK\n");
		}
	}

	{ /* F77BLAS and CBLAS :: sgemm */
		void (*sgemm_)(const char*, const char*, BlasInt*, BlasInt*, BlasInt*, float*, float*, BlasInt*, float*, BlasInt*, float*, float*, BlasInt*);
		void (*cblas_sgemm)(int, int, int, BlasInt, BlasInt, BlasInt, float, float*, BlasInt, float*, BlasInt, float, float*, BlasInt);

		float A[] = {1., 2., 3., 4.};
		BlasInt m = 2;
		BlasInt lda = 2, ldb = 2, ldc = 2;
		float alpha = 1.0, beta = 0.0;
		float B[] = {1., 1., 1., 1.};
		float C[] = {0., 0., 0., 0.};

		fprintf(stdout, "%s::sgemm_ .. ", LIBBLAS_SO);
		sgemm_ = (void (*)(const char*, const char*, BlasInt*, BlasInt*, BlasInt*, float*, float*, BlasInt*, float*, BlasInt*, float*, float*, BlasInt*)) dlsym(handle, "sgemm_");
		error = dlerror();
		if (error != NULL) {
			fprintf(stdout, "NotFound\n");
		} else {
			sgemm_(&NoTrans_, &NoTrans_, &m, &m, &m, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
			assert(4. == C[0]); assert(6. == C[1]);
			assert(4. == C[2]); assert(6. == C[3]);
			fprintf(stdout, "OK\n");
		}

		fprintf(stdout, "%s::cblas_sgemm .. ", LIBBLAS_SO);
		cblas_sgemm = (void (*)(int, int, int, BlasInt, BlasInt, BlasInt, float, float*, BlasInt, float*, BlasInt, float, float*, BlasInt)) dlsym(handle, "cblas_sgemm");
		error = dlerror();
		if (error != NULL) {
			fprintf(stdout, "NotFound\n");
		} else {
			cblas_sgemm(RowMajor, NoTrans, NoTrans, m, m, m, alpha, A, lda, B, ldb, beta, C, ldc);
			assert(3. == C[0]); assert(3. == C[1]);
			assert(7. == C[2]); assert(7. == C[3]);
			fprintf(stdout, "OK\n");
		}
	}

	dlclose(handle);
	return 0;
}
