#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <assert.h>

#define OPENBLAS_SO "/usr/lib/x86_64-linux-gnu/libopenblasp-r0.3.3.so"
#define NETLIB_SO   "/usr/lib/x86_64-linux-gnu/blas/libblas.so.3"

#if !defined(BlasInt)
#define BlasInt __int64_t
#endif

#if !defined(LIBBLAS_SO)
#define LIBBLAS_SO NETLIB_SO
#endif

int
main(void)
{
	fprintf(stdout, "Testing sasum_ and cblas_sasum from %s .. ", LIBBLAS_SO);

	void *handle = dlopen(LIBBLAS_SO, RTLD_LAZY);
	float (*ffi_c_sasum)(BlasInt, float *, BlasInt);
	float (*ffi_f_sasum)(BlasInt *, float *, BlasInt *);
	float X[] = {1., -2., 3.};
	BlasInt n = 3;
	BlasInt incx = 1;
	float tmp = 0.;

	/* test F77 ABI, i.e. BLAS */
	*(void **) (&ffi_f_sasum) = dlsym(handle, "sasum_");
	//printf("%f\n", ffi_f_sasum(&n, X, &incx));
	tmp = ffi_f_sasum(&n, X, &incx);
	assert(6.00 == tmp);

	/* test C ABI, i.e. CBLAS */
	*(void **) (&ffi_c_sasum) = dlsym(handle, "cblas_sasum");
	//printf("%f\n", ffi_c_sasum(n, X, incx));
	tmp = ffi_c_sasum(n, X, incx);
	assert(6.00 == tmp);

	dlclose(handle);
	fprintf(stdout, "OK\n");
	return 0;
}
