#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

#define BlasInt __int64_t
#define OPENBLAS_SO "/usr/lib/x86_64-linux-gnu/libopenblasp-r0.3.3.so"
#define NETLIB_SO   "/usr/lib/x86_64-linux-gnu/blas/libblas.so.3"
#define LIBBLAS_SO NETLIB_SO

int
main(void)
{
	void *handle = dlopen(LIBBLAS_SO, RTLD_LAZY);
	float (*ffi_c_sasum)(BlasInt, float *, BlasInt);
	float (*ffi_f_sasum)(BlasInt *, float *, BlasInt *);
	float X[] = {1., -2., 3.};
	BlasInt n = 3;
	BlasInt incx = 1;

	/* test c ABI */
	*(void **) (&ffi_c_sasum) = dlsym(handle, "cblas_sasum");
	printf("%f\n", ffi_c_sasum(n, X, incx));

	/* test f77 ABI */
	*(void **) (&ffi_f_sasum) = dlsym(handle, "sasum_");
	printf("%f\n", ffi_f_sasum(&n, X, &incx));

	dlclose(handle);
	return 0;
}
