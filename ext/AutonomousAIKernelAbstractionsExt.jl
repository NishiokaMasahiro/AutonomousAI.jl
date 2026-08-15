"""
    AutonomousAIKernelAbstractionsExt

GPU code generation (spec section 14).  Registers vendor-neutral KernelAbstractions
templates so the generator can emit GPU kernels for the algorithms whose access
pattern is actually GPU-shaped.

`:mad_anomaly` is deliberately **not** registered: it is two median selections, which
are latency- and branch-bound.  Emitting a GPU kernel for it would produce code that
runs and loses, and would then have to be discovered as a loss by benchmarking.
Encoding that in the template registry keeps the mistake out of the search space.
"""
module AutonomousAIKernelAbstractionsExt

using AutonomousAI
using KernelAbstractions

const CG = AutonomousAI.CodeGeneration

function gpu_zscore(T::Symbol, backend::Symbol, params::Dict{String,Int})
    fname = Symbol(CG.entrypoint_name(:zscore_anomaly))
    kname = Symbol("_kernel_zscore")
    threshold = get(params, "threshold_milli", 3000) / 1000
    body = quote
        @kernel function $(kname)(out, @Const(x), mu, thr)
            i = @index(Global)
            out[i] = abs(x[i] - mu) > thr
        end

        function $(fname)(x::AbstractVector{$T}, out::AbstractVector{Bool})
            _axes_guard(x, out)
            n = length(x)
            s = sum(x)
            mu = s / n
            acc = sum(abs2, x .- mu)
            sigma = sqrt(acc / (n - 1))
            thr = $(T)($threshold) * sigma
            backend = get_backend(x)
            $(kname)(backend, 256)(out, x, mu, thr; ndrange = n)
            synchronize(backend)
            return count(out)
        end
    end
    return Base.remove_linenums!(body)
end

function gpu_stencil(T::Symbol, backend::Symbol, params::Dict{String,Int})
    fname = Symbol(CG.entrypoint_name(:stencil3))
    kname = Symbol("_kernel_stencil")
    body = quote
        @kernel function $(kname)(out, @Const(x), n, third)
            i = @index(Global)
            if i > 1 && i < n
                out[i] = (x[i-1] + x[i] + x[i+1]) * third
            else
                out[i] = x[i]
            end
        end

        function $(fname)(x::AbstractVector{$T}, out::AbstractVector{$T})
            _axes_guard(x, out)
            n = length(x)
            third = $(T)(1) / $(T)(3)
            backend = get_backend(x)
            $(kname)(backend, 256)(out, x, n, third; ndrange = n)
            synchronize(backend)
            return out
        end
    end
    return Base.remove_linenums!(body)
end

function gpu_matmul(T::Symbol, backend::Symbol, params::Dict{String,Int})
    fname = Symbol(CG.entrypoint_name(:matmul))
    # Deliberately delegates to the vendor GEMM: a hand-rolled KA matmul loses to
    # cuBLAS by an order of magnitude, and "generate a kernel" is not a reason to
    # ship a worse one.  The generator's job is selection, not novelty.
    body = quote
        function $(fname)(A::AbstractMatrix{$T}, B::AbstractMatrix{$T},
                          C::AbstractMatrix{$T})
            _dims_guard(A, B, C)
            mul!(C, A, B)
            synchronize(get_backend(C))
            return C
        end
    end
    return Base.remove_linenums!(body)
end

function __init__()
    CG.GPU_TEMPLATES[:zscore_anomaly] = gpu_zscore
    CG.GPU_TEMPLATES[:stencil3] = gpu_stencil
    CG.GPU_TEMPLATES[:matmul] = gpu_matmul
    push!(CG.ALLOWED_MACROS, Symbol("@Const"))
    return nothing
end

end # module
