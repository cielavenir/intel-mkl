#!/usr/bin/julia-1.0
# Compare performance among different BLAS implementations
# Copyright (C) 2018 Mo Zhou <lumin@debian.org>, MIT/Expat License.
# Reference: Julia/stdlib/LinearAlgebra/src/blas.jl
using LinearAlgebra
using Logging

const netlibblas = "/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.8.0"
const atlas      = "/usr/lib/x86_64-linux-gnu/atlas/libblas.so.3.10.3"
const openblas   = "/usr/lib/x86_64-linux-gnu/openblas/libblas.so.3"
const mkl        = "/usr/lib/x86_64-linux-gnu/libmkl_rt.so"
const nvblas     = "/usr/lib/x86_64-linux-gnu/libnvblas.so"
const blis       = "/usr/lib/x86_64-linux-gnu/libblis.so.2"
const BlasInt    = Int32
const BlasFloat  = Float64

BLASES = [blis, openblas, mkl]
sizes = [2 .^ (1:8)..., 256:128:4096...]
scores = zeros(length(sizes), length(BLASES)+1)

flops(s, t) = 2.0 * s^3 / t

for (i, N3) in enumerate(sizes)
	julia_dgemm = false
	@info("Matrix size = $N3")
	for (j, libblas) in enumerate(BLASES)

		# create FFI function
		@eval begin
			function ffi_gemm!(transA::Char, transB::Char,
							   alpha::BlasFloat, A::AbstractVecOrMat{BlasFloat},
							   B::AbstractVecOrMat{BlasFloat}, beta::BlasFloat,
							   C::AbstractVecOrMat{BlasFloat})
				m = size(A, transA == 'N' ? 1 : 2)
				ka = size(A, transA == 'N' ? 2 : 1)
				kb = size(B, transB == 'N' ? 1 : 2)
				n = size(B, transB == 'N' ? 2 : 1) 
				ccall((:dgemm_, $libblas), Cvoid,
					(Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
					 Ref{BlasInt}, Ref{BlasFloat}, Ptr{BlasFloat}, Ref{BlasInt},
					 Ptr{BlasFloat}, Ref{BlasInt}, Ref{BlasFloat}, Ptr{BlasFloat},
					 Ref{BlasInt}),
					 transA, transB, m, n,
					 ka, alpha, A, max(1,stride(A,2)),
					 B, max(1,stride(B,2)), beta, C,
					 max(1,stride(C,2)))
			end
		end

		# generate random matrix
		x, y = rand(N3, N3).|>BlasFloat, rand(N3, N3).|>BlasFloat
		z = zeros(N3, N3).|>BlasFloat

		# GEMM using julia stdlib
		if !julia_dgemm
			BLAS.gemm('N', 'N', x, y)  # JIT
			tm = @elapsed BLAS.gemm('N', 'N', x, y)
			scores[i, end] = flops(N3, tm)
			julia_dgemm = true
		end

		# GEMM using FFI
		@info("dgemm $libblas")
		ffi_gemm!('N', 'N', 1e0, x, y, 0e0, z)  # JIT
		tm = @elapsed ffi_gemm!('N', 'N', 1e0, x, y, 0e0, z)
		scores[i, j] = flops(N3, tm)

		# correctness
		z2 = BLAS.gemm('N', 'N', x, y)
		ffi_gemm!('N', 'N', 1e0, x, y, 0e0, z)
		error = norm(z2 - z)
		if error > 1e-7
			@error("dgemm Error : $error")  # correctness
		end

	end
end

print(scores)
