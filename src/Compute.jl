"""
    Compute

Backend abstraction and *trusted reference* implementations (spec sections 9, 13).

Two distinct roles live here and must not be confused:

* `reference_*` functions are hand-written, in-repo, FP64-capable implementations.
  They are the oracle that verification compares against.  They are never generated.
* backend types drive Multiple Dispatch so that the same high-level call selects a
  CPU or GPU path by *type*, not by an `if backend == :cuda` branch.

The GPU path is provided by the package extension in `ext/`; when CUDA.jl is absent
`available_backends` simply does not return `:cuda`, and the planner's candidate set
shrinks.  No stubs that pretend to run on a GPU.
"""
module Compute

using LinearAlgebra, Statistics, Random
using ..Schema
import ..Memory
using ..HAL

export ComputeBackend, CPUSerialBackend, CPUSIMDBackend, CPUThreadsBackend,
       CPUBLASBackend, CUDABackend, DistributedBackend,
       backend_instance, available_backends, backend_symbol,
       make_inputs, reference_run, algorithm_facts, flops_for, bytes_for,
       ALGORITHM_LIST

abstract type ComputeBackend end
struct CPUSerialBackend <: ComputeBackend end
struct CPUSIMDBackend <: ComputeBackend end
struct CPUThreadsBackend <: ComputeBackend end
struct CPUBLASBackend <: ComputeBackend end
struct CUDABackend <: ComputeBackend end
struct DistributedBackend <: ComputeBackend end

backend_symbol(::CPUSerialBackend) = :cpu_serial
backend_symbol(::CPUSIMDBackend) = :cpu_simd
backend_symbol(::CPUThreadsBackend) = :cpu_threads
backend_symbol(::CPUBLASBackend) = :cpu_blas
backend_symbol(::CUDABackend) = :cuda
backend_symbol(::DistributedBackend) = :distributed

function backend_instance(s::Symbol)
    s === :cpu_serial && return CPUSerialBackend()
    s === :cpu_simd && return CPUSIMDBackend()
    s === :cpu_threads && return CPUThreadsBackend()
    s === :cpu_blas && return CPUBLASBackend()
    s === :cuda && return CUDABackend()
    s === :distributed && return DistributedBackend()
    throw(Schema.SchemaError("unknown backend $(s)"))
end

"Set by the CUDA package extension when a usable device is present."
const CUDA_FUNCTIONAL = Ref(false)

"""
    available_backends(profile) -> Vector{Symbol}

Capability-derived, not configured.  A machine with one core does not get
`:cpu_threads`; a machine with no functional CUDA device does not get `:cuda`.
"""
function available_backends(p::HAL.HardwareProfile)
    b = Symbol[:cpu_serial, :cpu_simd, :cpu_blas]
    (Threads.nthreads() > 1 && p.cpu.physical_cores > 1) && push!(b, :cpu_threads)
    (CUDA_FUNCTIONAL[] && !isempty(p.gpus)) && push!(b, :cuda)
    return b
end

# --------------------------------------------------------------- algorithms --

const ALGORITHM_LIST = Symbol[:zscore_anomaly, :mad_anomaly, :matmul,
                              :sum_reduction, :stencil3]

for a in ALGORITHM_LIST
    Schema.register_algorithm!(a)
end

"""
    algorithm_facts() -> Vector{AlgorithmFact}

Static complexity/intensity priors.  These seed `SemanticMemory`; the *measured*
numbers in `HardwareMemory` always win when both exist.
"""
function algorithm_facts()
    return Memory.AlgorithmFact[
        Memory.AlgorithmFact(:zscore_anomaly, 5.0, 1.0, 8.0, 0.6, true, true,
            :Float32, :stable,
            "three streaming passes: mean, variance, threshold. Memory bound."),
        Memory.AlgorithmFact(:mad_anomaly, 2.0, 1.0, 12.0, 0.15, false, false,
            :Float32, :stable,
            "two selections (median of x, median of |x-med|). Robust to outliers; " *
            "selection is latency bound and GPU-hostile."),
        Memory.AlgorithmFact(:matmul, 2.0, 1.5, 12.0, 30.0, true, true,
            :Float32, :conditionally_stable,
            "2*n^3 flops for n^2 elements: compute bound, the canonical GPU win."),
        Memory.AlgorithmFact(:sum_reduction, 4.0, 1.0, 8.0, 0.5, true, true,
            :Float64, :conditionally_stable,
            "Kahan-compensated summation; naive FP32 summation loses O(n*eps)."),
        Memory.AlgorithmFact(:stencil3, 3.0, 1.0, 16.0, 0.19, true, true,
            :Float32, :stable,
            "3-point stencil: strictly bandwidth bound, arithmetic intensity ~0.2."),
    ]
end

"F(n) for the roofline model. `n` is the element count of the primary input."
function flops_for(alg::Symbol, size::Vector{Int})
    n = prod(size)
    alg === :matmul && return 2.0 * size[1] * size[1] * size[1]
    alg === :zscore_anomaly && return 5.0 * n
    alg === :mad_anomaly && return 4.0 * n * log2(max(n, 2))
    alg === :sum_reduction && return 4.0 * n
    alg === :stencil3 && return 3.0 * n
    return Float64(n)
end

"Bytes that must cross the slowest relevant link."
function bytes_for(alg::Symbol, size::Vector{Int}, precision::Symbol)
    w = precision === :Float64 ? 8 : precision === :Float32 ? 4 : 2
    n = prod(size)
    alg === :matmul && return 3.0 * size[1] * size[1] * w
    alg === :zscore_anomaly && return 3.0 * n * w
    alg === :mad_anomaly && return 4.0 * n * w
    alg === :sum_reduction && return 1.0 * n * w
    alg === :stencil3 && return 2.0 * n * w
    return Float64(n * w)
end

# ----------------------------------------------------------------- fixtures --

"""
    make_inputs(alg, size, T; seed) -> NamedTuple

Deterministic fixtures.  The anomaly cases inject a known number of planted
outliers so that verification can check *recall*, not just numeric closeness --
a z-score kernel that returns all-false has zero numeric error against a
mis-specified reference but is useless.
"""
function make_inputs(alg::Symbol, size::Vector{Int}, ::Type{T}; seed::Int = 20260813) where {T}
    rng = MersenneTwister(seed)
    if alg === :matmul
        n = size[1]
        return (A = T.(randn(rng, n, n)), B = T.(randn(rng, n, n)),
                C = zeros(T, n, n), planted = Int[])
    end
    n = prod(size)
    x = T.(randn(rng, n))
    if alg === :zscore_anomaly || alg === :mad_anomaly
        planted = sort(randperm(rng, n)[1:max(1, div(n, 1000))])
        for i in planted
            x[i] = T(12) * sign(x[i] == 0 ? one(T) : x[i])
        end
        return (x = x, out = falses(n), scratch = similar(x), planted = planted)
    end
    return (x = x, out = similar(x), planted = Int[])
end

# ------------------------------------------------------- trusted references --

"FP64 reference: z-score anomaly mask + count."
function reference_zscore(x::AbstractVector{<:Real}; threshold::Float64 = 3.0)
    xd = Float64.(x)
    mu = mean(xd)
    sigma = std(xd; corrected = true, mean = mu)
    mask = abs.(xd .- mu) .> threshold * sigma
    return (mask = mask, count = count(mask), mu = mu, sigma = sigma)
end

function reference_mad(x::AbstractVector{<:Real}; threshold::Float64 = 3.0)
    xd = Float64.(x)
    med = median(xd)
    mad = median(abs.(xd .- med))
    scale = 1.4826 * mad
    mask = abs.(xd .- med) .> threshold * scale
    return (mask = mask, count = count(mask), med = med, scale = scale)
end

reference_matmul(A::AbstractMatrix, B::AbstractMatrix) = Float64.(A) * Float64.(B)

"Pairwise summation in FP64: the oracle for `sum_reduction`."
reference_sum(x::AbstractVector{<:Real}) = sum(Float64.(x))

function reference_stencil(x::AbstractVector{<:Real})
    xd = Float64.(x)
    n = length(xd)
    out = copy(xd)
    for i in 2:(n-1)
        out[i] = (xd[i-1] + xd[i] + xd[i+1]) / 3
    end
    return out
end

"""
    reference_run(alg, inputs) -> NamedTuple

Single entry point used by `Verification` to obtain the oracle result.
"""
function reference_run(alg::Symbol, inputs)
    alg === :zscore_anomaly && return reference_zscore(inputs.x)
    alg === :mad_anomaly && return reference_mad(inputs.x)
    alg === :matmul && return (C = reference_matmul(inputs.A, inputs.B),)
    alg === :sum_reduction && return (s = reference_sum(inputs.x),)
    alg === :stencil3 && return (out = reference_stencil(inputs.x),)
    throw(Schema.SchemaError("no reference implementation for $(alg)"))
end

end # module
