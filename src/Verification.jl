"""
    Verification

Correctness evidence for generated code (spec section 34).

Design points that differ from the naive reading of the spec:

* `relative_error = |r - ref| / |ref|` is undefined at `ref == 0` and misleading for
  tiny `ref`.  We use the standard mixed test `|r - ref| <= atol + rtol*|ref|`.
* The tolerance is not a constant.  It is derived from the working precision, the
  problem size and the numerical character of the algorithm, so that a "pass" in
  Float32 means "as accurate as Float32 can be", not "within 1e-10".
* Numeric closeness alone is insufficient for detection tasks: an all-false anomaly
  mask can score perfectly against a badly chosen scalar metric.  Detection quality
  (recall/precision against planted outliers) is checked separately.
"""
module Verification

using Statistics, LinearAlgebra, Printf
using ..Schema

export VerificationReport, mixed_error, max_mixed_error, expected_tolerance,
       verify_values, verify_mask, property_checks, verify_candidate, passed

struct VerificationReport
    passed::Bool
    checks::Vector{Tuple{String,Bool,String}}
    rel_error::Float64
    tolerance::Float64
end

passed(r::VerificationReport) = r.passed

function Base.show(io::IO, r::VerificationReport)
    println(io, "VerificationReport: ", r.passed ? "PASS" : "FAIL",
            @sprintf("  (err %.3e, tol %.3e)", r.rel_error, r.tolerance))
    for (name, ok, msg) in r.checks
        println(io, "  [", ok ? "ok  " : "FAIL", "] ", name, isempty(msg) ? "" : "  -- " * msg)
    end
end

"""
    mixed_error(x, ref; atol) -> Float64

Scale-aware error: reduces to relative error for large `ref` and to absolute error
near zero.
"""
mixed_error(x::Real, ref::Real; atol::Real = 0.0) =
    abs(Float64(x) - Float64(ref)) / max(abs(Float64(ref)), Float64(atol), floatmin(Float64))

function max_mixed_error(x::AbstractArray, ref::AbstractArray; atol::Real = 0.0)
    size(x) == size(ref) || return Inf
    m = 0.0
    @inbounds for i in eachindex(ref)
        m = max(m, mixed_error(x[i], ref[i]; atol = atol))
    end
    return m
end

"""
    expected_tolerance(alg, precision, n; transforms) -> Float64

Error budget: `c(alg) * sqrt(n) * eps(precision)`, inflated when a transform is
allowed to change the arithmetic (`:fma`, `:fastmath`) or when the algorithm is only
conditionally stable.  Growing as `sqrt(n)` rather than `n` reflects random-walk
rounding accumulation; the naive-summation case is handled by its own coefficient.
"""
function expected_tolerance(alg::Symbol, precision::Symbol, n::Integer;
                            transforms::Vector{Symbol} = Symbol[])
    eps_p = Schema.precision_eps(precision)
    c = alg === :matmul ? 8.0 :
        alg === :sum_reduction ? 4.0 :
        alg === :zscore_anomaly ? 6.0 :
        alg === :stencil3 ? 3.0 : 4.0
    infl = 1.0
    (:fma in transforms) && (infl *= 2.0)
    (:fastmath in transforms) && (infl *= 8.1)
    return c * infl * sqrt(max(n, 1)) * eps_p
end

function verify_values(x, ref, tol::Float64; atol::Real = 0.0, name::String = "values")
    err = x isa AbstractArray ? max_mixed_error(x, ref; atol = atol) :
          mixed_error(x, ref; atol = atol)
    ok = isfinite(err) && err <= tol
    msg = @sprintf("max mixed error %.3e vs tolerance %.3e", err, tol)
    return (name, ok, msg), err
end

"""
    verify_mask(mask, ref_mask, planted) -> checks

Boolean masks are compared exactly *and* against the planted ground truth, because
two implementations can agree on a wrong answer when both use the same faulty
threshold.
"""
function verify_mask(mask::AbstractVector{Bool}, ref_mask::AbstractVector{Bool},
                     planted::Vector{Int})
    checks = Tuple{String,Bool,String}[]
    agree = count(mask .== ref_mask)
    n = length(mask)
    frac = agree / n
    push!(checks, ("mask agreement", frac >= 1.0 - 1e-12,
                   @sprintf("%d/%d positions agree (%.6f)", agree, n, frac)))
    if !isempty(planted)
        tp = count(i -> mask[i], planted)
        recall = tp / length(planted)
        fp = count(mask) - tp
        precision = count(mask) == 0 ? 0.0 : tp / count(mask)
        push!(checks, ("planted-outlier recall", recall >= 0.9,
                       @sprintf("recall %.3f (%d/%d), precision %.3f, fp %d",
                                recall, tp, length(planted), precision, fp)))
        push!(checks, ("non-degenerate output", count(mask) > 0 && count(mask) < n,
                       @sprintf("%d flagged of %d", count(mask), n)))
    end
    return checks
end

"""
    property_checks(alg, f, inputs) -> checks

Metamorphic tests.  They need no oracle, which makes them the only correctness
evidence available when the reference itself is suspect.

* z-score / MAD: the mask is invariant under `x -> a*x + b` for `a > 0`.
* sum: invariant under permutation, within the summation error budget.
* stencil: linear, so `f(a*x) == a*f(x)`.
* matmul: `A*(B*v) == (A*B)*v` (a cheap associativity spot check).
"""
function property_checks(alg::Symbol, f::Function, inputs; tol::Float64 = 1e-6)
    checks = Tuple{String,Bool,String}[]
    try
        if alg === :zscore_anomaly || alg === :mad_anomaly
            x = inputs.x
            T = eltype(x)
            base = f(x)
            shifted = f(T(2) .* x .+ T(5))
            same = base == shifted
            push!(checks, ("affine invariance of mask", same,
                           same ? "mask stable under x -> 2x+5" :
                           "mask changed under an order-preserving affine map"))
        elseif alg === :sum_reduction
            x = inputs.x
            s1 = f(x)
            s2 = f(reverse(x))
            e = mixed_error(s1, s2; atol = 1e-30)
            push!(checks, ("permutation stability", e <= tol,
                           @sprintf("|sum(x) - sum(reverse(x))| rel %.3e", e)))
        elseif alg === :stencil3
            x = inputs.x
            T = eltype(x)
            y1 = f(T(3) .* x)
            y2 = T(3) .* f(x)
            e = max_mixed_error(y1, y2; atol = 1e-30)
            push!(checks, ("linearity", e <= tol, @sprintf("rel deviation %.3e", e)))
        elseif alg === :matmul
            A, B = inputs.A, inputs.B
            v = ones(eltype(A), size(B, 2))
            lhs = f(A, B) * v
            rhs = A * (B * v)
            e = max_mixed_error(lhs, rhs; atol = 1e-30)
            push!(checks, ("associativity spot check", e <= tol,
                           @sprintf("rel deviation %.3e", e)))
        end
    catch err
        push!(checks, ("property checks executed", false, "threw: $(err)"))
    end
    return checks
end

"""
    verify_candidate(alg, precision, transforms, result, reference, inputs; n) -> report

Aggregates value comparison, mask/detection checks and finiteness into one verdict.
Any failed check fails the candidate: verification is a conjunction, never a score.
"""
function verify_candidate(alg::Symbol, precision::Symbol, transforms::Vector{Symbol},
                          result, reference, inputs; n::Integer = 0,
                          property_fn::Union{Nothing,Function} = nothing)
    tol = expected_tolerance(alg, precision, n; transforms = transforms)
    checks = Tuple{String,Bool,String}[]
    err = 0.0
    if alg === :zscore_anomaly || alg === :mad_anomaly
        append!(checks, verify_mask(result.mask, reference.mask,
                                    hasproperty(inputs, :planted) ? inputs.planted : Int[]))
        c, e = verify_values(result.count, reference.count, 0.0; atol = 0.5,
                             name = "anomaly count")
        push!(checks, c)
        err = e
    elseif alg === :sum_reduction
        c, e = verify_values(result.s, reference.s, tol; atol = 1e-30, name = "sum value")
        push!(checks, c)
        err = e
    elseif alg === :matmul
        c, e = verify_values(result.C, reference.C, tol; atol = 1e-30, name = "matrix entries")
        push!(checks, c)
        err = e
    elseif alg === :stencil3
        c, e = verify_values(result.out, reference.out, tol; atol = 1e-30,
                             name = "stencil output")
        push!(checks, c)
        err = e
    end
    finite = all_finite(result)
    push!(checks, ("finite output", finite, finite ? "" : "NaN or Inf present"))
    property_fn === nothing || append!(checks, property_fn())
    return VerificationReport(all(c -> c[2], checks), checks, err, tol)
end

all_finite(x::Real) = isfinite(x)
all_finite(x::AbstractArray{Bool}) = true
all_finite(x::AbstractArray) = all(isfinite, x)
all_finite(x::NamedTuple) = all(all_finite, values(x))
all_finite(x) = true

end # module
