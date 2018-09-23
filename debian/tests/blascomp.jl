#!/usr/bin/julia-1.0
# Compare performance among different BLAS implementations
# Reference: Julia/stdlib/LinearAlgebra/src/blas.jl
using LinearAlgebra
using Logging

const N1 = 65536  # N for level1 calls
const N3 = 4096   # N for level3 calls
const netlibblas = "/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.8.0"
const atlas      = "/usr/lib/x86_64-linux-gnu/atlas/libblas.so.3.10.3"
const openblas   = "/usr/lib/x86_64-linux-gnu/openblas/libblas.so.3"
const mkl        = "/usr/lib/x86_64-linux-gnu/libmkl_rt.so"
const blis       = "/home/lumin/git/blis/lib/haswell/libblis.so"

BLASES = [blis, openblas, mkl]

julia_nrm2 = false
for libblas in BLASES
	global julia_nrm2
	@eval begin
		function ffi_nrm2(n::Integer, X::Union{Ptr{Float64}, AbstractArray{Float64}}, incx::Integer)
			ccall((:dnrm2_, $libblas), Float64, (Ref{Int64}, Ptr{Float64}, Ref{Int64}), n, X, incx)
		end
	end
	x = rand(N1)
	if !julia_nrm2
		@info("dnrm2 Julia")
		norm(x)  # JIT
		@time norm(x)
		julia_nrm2 = true
	end
	@info("dnrm2 $libblas")
	ffi_nrm2(N1, x, 1)  # JIT
	@time ffi_nrm2(N1, x, 1)

	error = abs(norm(x) - ffi_nrm2(N1, x, 1))
	if error > 1e-7
		@warn("dnrm2 Error : $error") # correctness
	end
end

julia_sgemm = false
for libblas in BLASES
	global julia_sgemm
	@eval begin
		function ffi_gemm!(transA::Char, transB::Char,
						   alpha::Float32, A::AbstractVecOrMat{Float32},
						   B::AbstractVecOrMat{Float32}, beta::Float32,
						   C::AbstractVecOrMat{Float32})
			m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1) 
            ccall((:sgemm_, $libblas), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{Int32}, Ref{Int32},
                 Ref{Int32}, Ref{Float32}, Ptr{Float32}, Ref{Int32},
                 Ptr{Float32}, Ref{Int32}, Ref{Float32}, Ptr{Float32},
                 Ref{Int32}),
                 transA, transB, m, n,
                 ka, alpha, A, max(1,stride(A,2)),
                 B, max(1,stride(B,2)), beta, C,
                 max(1,stride(C,2)))
		end
	end
	x, y, z = rand(Float32, N3, N3), rand(Float32, N3, N3), zeros(Float32, N3, N3)
	if !julia_sgemm
		@info("sgemm Julia")
		BLAS.gemm('N', 'N', x, y)  # JIT
		@time BLAS.gemm('N', 'N', x, y)
		julia_sgemm = true
	end
	@info("sgemm $libblas")
	ffi_gemm!('N', 'N', 1. |>Float32, x, y, 0. |>Float32, z)  # JIT
	@time ffi_gemm!('N', 'N', 1. |>Float32, x, y, 0. |> Float32, z)

	z2 = BLAS.gemm('N', 'N', x, y)
	ffi_gemm!('N', 'N', 1. |>Float32, x, y, 0. |>Float32, z)
	error = norm(z2 - z)
	if error > 1e-7
		@warn("sgemm Error : $error")  # correctness
	end
end

julia_dgemm = false
for libblas in BLASES
	global julia_dgemm
	@eval begin
		function ffi_gemm!(transA::Char, transB::Char,
						   alpha::Float64, A::AbstractVecOrMat{Float64},
						   B::AbstractVecOrMat{Float64}, beta::Float64,
						   C::AbstractVecOrMat{Float64})
			m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1) 
            ccall((:dgemm_, $libblas), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64},
                 Ref{Int64}, Ref{Float64}, Ptr{Float64}, Ref{Int64},
                 Ptr{Float64}, Ref{Int64}, Ref{Float64}, Ptr{Float64},
                 Ref{Int64}),
                 transA, transB, m, n,
                 ka, alpha, A, max(1,stride(A,2)),
                 B, max(1,stride(B,2)), beta, C,
                 max(1,stride(C,2)))
		end
	end
	x, y, z = rand(N3, N3), rand(N3, N3), zeros(N3, N3)
	if !julia_dgemm
		@info("dgemm Julia")
		BLAS.gemm('N', 'N', x, y)  # JIT
		@time BLAS.gemm('N', 'N', x, y)
		julia_dgemm = true
	end
	@info("dgemm $libblas")
	ffi_gemm!('N', 'N', 1., x, y, 0., z)  # JIT
	@time ffi_gemm!('N', 'N', 1., x, y, 0., z)

	z2 = BLAS.gemm('N', 'N', x, y)
	ffi_gemm!('N', 'N', 1., x, y, 0., z)
	error = norm(z2 - z)
	if error > 1e-7
		@warn("dgemm Error : $error")  # correctness
	end
end
