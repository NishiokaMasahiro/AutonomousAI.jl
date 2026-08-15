"""
    Execute

Composes the trusted driver that runs a generated unit inside the sandbox, and
turns the returned payload into evidence (spec sections 15, 34, 40).

The generated unit computes; the driver measures and summarises.  The driver text is
fixed in this file and is never influenced by the model, which is what makes the
timings and the verification summary trustworthy even though the code under test is
not.

Full outputs are never shipped back across the process boundary -- a 4096x4096
Float32 matrix would dominate the measurement.  Instead each algorithm returns a
*sufficient summary* for verification:

* anomaly detection -> flagged count and flagged indices (bounded, exact comparison)
* summation        -> the value
* stencil          -> sum, 2-norm, head, and a random-probe inner product
* matmul           -> `C*r` for a fixed pseudo-random `r` (Freivalds' check, which
  catches a wrong product with probability >= 1/2 per probe and is O(n^2))
"""
module Execute

using Statistics, LinearAlgebra, Random, Printf
using ..Schema
using ..Compute
using ..Sandbox
using ..CodeGeneration
using ..Optimization
using ..Verification

export driver_source, run_candidate, reference_summary, compare_summaries,
       FIXTURE_SEED, PROBE_SEED, CandidateRun

const FIXTURE_SEED = 20260813
const PROBE_SEED = 424242

"Minimal JSON writer injected into the sandbox process (no package deps there)."
const DRIVER_JSON = raw"""
_jesc(s) = replace(String(s), '\\' => "\\\\", '"' => "\\\"")
_jv(x::Bool) = x ? "true" : "false"
_jv(x::Integer) = string(x)
_jv(x::AbstractFloat) = isfinite(x) ? string(Float64(x)) : "null"
_jv(x::AbstractString) = string('"', _jesc(x), '"')
_jv(x::AbstractVector) = string("[", join([_jv(v) for v in x], ","), "]")
_jv(x::AbstractDict) = string("{", join([string(_jv(String(k)), ":", _jv(v))
                                         for (k, v) in x], ","), "}")
"""

"Fixture construction, byte-identical in intent to `Compute.make_inputs`."
function fixture_source(alg::Symbol, size::Vector{Int}, T::Symbol)
    if alg === :matmul
        return """
        function _fixture()
            rng = MersenneTwister($(FIXTURE_SEED))
            n = $(size[1])
            A = $(T).(randn(rng, n, n))
            B = $(T).(randn(rng, n, n))
            C = zeros($(T), n, n)
            return (A = A, B = B, C = C)
        end
        """
    end
    n = prod(size)
    planted = (alg === :zscore_anomaly || alg === :mad_anomaly)
    return """
    function _fixture()
        rng = MersenneTwister($(FIXTURE_SEED))
        n = $(n)
        x = $(T).(randn(rng, n))
        planted = Int[]
        if $(planted)
            planted = sort(randperm(rng, n)[1:max(1, div(n, 1000))])
            for i in planted
                x[i] = $(T)(12) * sign(x[i] == 0 ? one($(T)) : x[i])
            end
        end
        return (x = x, out = $(planted ? "falses(n)" : "similar(x)"),
                scratch = similar(x), planted = planted)
    end
    """
end

"How the driver invokes the generated entry point, and what it summarises."
function call_and_summary_source(alg::Symbol)
    entry = CodeGeneration.entrypoint_name(alg)
    if alg === :zscore_anomaly
        return """
        _call(inp) = $(entry)(inp.x, inp.out)
        function _summary(inp, r)
            idx = findall(inp.out)
            return Dict{String,Any}("count" => length(idx),
                "indices" => idx[1:min(end, 4096)],
                "index_sum" => Float64(sum(idx; init = 0)),
                "returned" => Float64(r))
        end
        """
    elseif alg === :mad_anomaly
        return """
        _call(inp) = $(entry)(inp.x, inp.out, inp.scratch)
        function _summary(inp, r)
            idx = findall(inp.out)
            return Dict{String,Any}("count" => length(idx),
                "indices" => idx[1:min(end, 4096)],
                "index_sum" => Float64(sum(idx; init = 0)),
                "returned" => Float64(r))
        end
        """
    elseif alg === :sum_reduction
        return """
        _call(inp) = $(entry)(inp.x)
        _summary(inp, r) = Dict{String,Any}("value" => Float64(r))
        """
    elseif alg === :stencil3
        return """
        _call(inp) = $(entry)(inp.x, inp.out)
        function _summary(inp, r)
            n = length(inp.out)
            probe = randn(MersenneTwister($(PROBE_SEED)), n)
            return Dict{String,Any}("sum" => Float64(sum(Float64, inp.out)),
                "norm" => Float64(sqrt(sum(abs2, Float64.(inp.out)))),
                "head" => Float64.(inp.out[1:min(8, n)]),
                "probe" => Float64(sum(Float64.(inp.out) .* probe)))
        end
        """
    elseif alg === :matmul
        return """
        _call(inp) = $(entry)(inp.A, inp.B, inp.C)
        function _summary(inp, r)
            n = size(inp.C, 2)
            probe = randn(MersenneTwister($(PROBE_SEED)), n)
            y = Float64.(inp.C) * probe
            return Dict{String,Any}("freivalds" => y,
                "norm" => Float64(sqrt(sum(abs2, Float64.(inp.C)))),
                "sum" => Float64(sum(Float64, inp.C)))
        end
        """
    end
    throw(Schema.SchemaError("no driver for algorithm $(alg)"))
end

"""
    driver_source(alg, size, precision, samples) -> String

Trusted, fixed measurement harness.  Cold (compile-inclusive) and warm timings are
reported separately, as required by spec section 11.
"""
function driver_source(alg::Symbol, size::Vector{Int}, precision::Symbol, samples::Int)
    T = precision === :BFloat16 ? :Float32 : precision
    return string("""
    using Random, Statistics, LinearAlgebra
    """, DRIVER_JSON, "\n", fixture_source(alg, size, T), "\n",
    call_and_summary_source(alg), """

    let
        inp = _fixture()
        GC.gc()
        t_cold = @elapsed _call(inp)          # includes inference + codegen
        _call(inp)                            # discard first warm call
        bytes = @allocated _call(inp)
        ts = Float64[]
        for _ in 1:$(samples)
            GC.gc(false)
            push!(ts, @elapsed _call(inp))
        end
        r = _call(inp)
        s = _summary(inp, r)
        s["_cold_s"] = t_cold
        s["_samples"] = ts
        s["_alloc_bytes"] = Float64(bytes)
        s["_threads"] = Threads.nthreads()
        println(RESULT_MARKER * _jv(s))
    end
    """)
end

struct CandidateRun
    candidate::Schema.Candidate
    stats::Optimization.BenchmarkStats
    summary::Dict{String,Any}
    sandbox::Sandbox.SandboxResult
    source_hash::String
    ok::Bool
    error::String
end

"""
    run_candidate(candidate, size, spec; samples, allow_fastmath, allow_inbounds)

Generate -> validate -> sandbox -> measure.  A validation failure never reaches the
sandbox, and a sandbox failure never reaches the memory system as a timing.
"""
function run_candidate(c::Schema.Candidate, size::Vector{Int}, spec::Sandbox.SandboxSpec;
                       samples::Int = 7, allow_fastmath::Bool = false,
                       allow_inbounds::Bool = true)
    unit = CodeGeneration.generate(c; allow_fastmath = allow_fastmath,
                                   allow_inbounds = allow_inbounds)
    empty_stats = Optimization.BenchmarkStats(NaN, NaN, NaN, NaN, NaN, Float64[], 0, 0, 0.0)
    h = bytes2hex_short(unit.source)
    if !unit.report.ok
        return CandidateRun(c, empty_stats, Dict{String,Any}(),
                            Sandbox.SandboxResult(false, Dict{String,Any}(), "", "",
                                                  0.0, -1, false, "not executed"),
                            h, false,
                            "validation rejected: " * join(unit.report.errors, "; "))
    end
    driver = driver_source(c.algorithm, size, c.precision, samples)
    res = Sandbox.run_sandboxed(unit.source, driver, spec)
    if !res.ok
        return CandidateRun(c, empty_stats, res.payload, res, h, false,
                            isempty(res.error) ? first_lines(res.stderr) : res.error)
    end
    ts = Float64[Float64(v) for v in get(res.payload, "_samples", Any[])]
    isempty(ts) && return CandidateRun(c, empty_stats, res.payload, res, h, false,
                                       "sandbox returned no timing samples")
    med = median(ts)
    stats = Optimization.BenchmarkStats(Float64(get(res.payload, "_cold_s", NaN)),
                                        minimum(ts), med, mean(ts),
                                        median(abs.(ts .- med)), ts,
                                        round(Int, Float64(get(res.payload, "_alloc_bytes", 0.0))),
                                        0, 0.0)
    return CandidateRun(c, stats, res.payload, res, h, true, "")
end

first_lines(s::AbstractString; n::Int = 6) =
    join(Iterators.take(split(String(s), '\n'), n), "\n")

bytes2hex_short(s::AbstractString) = string(hash(String(s)), base = 16)

"""
    reference_summary(alg, size, T) -> Dict

Runs the trusted FP64 reference in-process and produces the *same* summary shape the
sandbox returns, so verification compares like with like.
"""
function reference_summary(alg::Symbol, size::Vector{Int}, ::Type{T} = Float64) where {T}
    inputs = Compute.make_inputs(alg, size, T)
    ref = Compute.reference_run(alg, inputs)
    if alg === :zscore_anomaly || alg === :mad_anomaly
        idx = findall(ref.mask)
        return Dict{String,Any}("count" => length(idx),
            "indices" => idx[1:min(end, 4096)],
            "index_sum" => Float64(sum(idx; init = 0)),
            "planted" => inputs.planted)
    elseif alg === :sum_reduction
        return Dict{String,Any}("value" => Float64(ref.s))
    elseif alg === :stencil3
        n = length(ref.out)
        probe = randn(MersenneTwister(PROBE_SEED), n)
        return Dict{String,Any}("sum" => sum(ref.out), "norm" => sqrt(sum(abs2, ref.out)),
            "head" => ref.out[1:min(8, n)], "probe" => sum(ref.out .* probe))
    elseif alg === :matmul
        n = size[1]
        probe = randn(MersenneTwister(PROBE_SEED), n)
        return Dict{String,Any}("freivalds" => ref.C * probe,
            "norm" => sqrt(sum(abs2, ref.C)), "sum" => sum(ref.C))
    end
    throw(Schema.SchemaError("no reference summary for $(alg)"))
end

"""
    compare_summaries(alg, got, ref, precision, transforms, n) -> VerificationReport

Exact where exactness is required (index sets), tolerance-based where rounding is
legitimate (values), and always with a non-degeneracy check.
"""
function compare_summaries(alg::Symbol, got::AbstractDict, ref::AbstractDict,
                           precision::Symbol, transforms::Vector{Symbol}, n::Int)
    tol = Verification.expected_tolerance(alg, precision, n; transforms = transforms)
    checks = Tuple{String,Bool,String}[]
    err = 0.0
    if alg === :zscore_anomaly || alg === :mad_anomaly
        gc_, rc = Int(get(got, "count", -1)), Int(get(ref, "count", -2))
        push!(checks, ("flagged count", gc_ == rc, "got $(gc_), reference $(rc)"))
        gi = Int[Int(v) for v in get(got, "indices", Any[])]
        ri = Int[Int(v) for v in get(ref, "indices", Any[])]
        push!(checks, ("flagged index set", gi == ri,
                       gi == ri ? "identical" : "$(length(symdiff(gi, ri))) positions differ"))
        planted = Int[Int(v) for v in get(ref, "planted", Any[])]
        if !isempty(planted)
            tp = count(in(Set(gi)), planted)
            rec = tp / length(planted)
            push!(checks, ("planted recall", rec >= 0.9,
                           @sprintf("%.3f (%d/%d)", rec, tp, length(planted))))
        end
        push!(checks, ("non-degenerate", gc_ > 0 && gc_ < n, "flagged $(gc_) of $(n)"))
        err = rc == 0 ? Float64(gc_) : abs(gc_ - rc) / rc
    elseif alg === :sum_reduction
        g, r = Float64(get(got, "value", NaN)), Float64(get(ref, "value", NaN))
        err = Verification.mixed_error(g, r; atol = 1e-30)
        push!(checks, ("sum value", err <= tol, @sprintf("err %.3e tol %.3e", err, tol)))
    elseif alg === :stencil3
        for k in ("sum", "norm", "probe")
            g, r = Float64(get(got, k, NaN)), Float64(get(ref, k, NaN))
            e = Verification.mixed_error(g, r; atol = 1e-30)
            err = max(err, e)
            push!(checks, (k, e <= tol, @sprintf("err %.3e tol %.3e", e, tol)))
        end
        gh = Float64[Float64(v) for v in get(got, "head", Any[])]
        rh = Float64[Float64(v) for v in get(ref, "head", Any[])]
        if length(gh) == length(rh) && !isempty(gh)
            e = Verification.max_mixed_error(gh, rh; atol = 1e-30)
            err = max(err, e)
            push!(checks, ("head elements", e <= tol, @sprintf("err %.3e", e)))
        end
    elseif alg === :matmul
        gy = Float64[Float64(v) for v in get(got, "freivalds", Any[])]
        ry = Float64[Float64(v) for v in get(ref, "freivalds", Any[])]
        if length(gy) != length(ry) || isempty(gy)
            push!(checks, ("freivalds probe", false, "probe vector length mismatch"))
        else
            e = Verification.max_mixed_error(gy, ry; atol = 1e-12)
            err = max(err, e)
            push!(checks, ("freivalds probe", e <= tol, @sprintf("err %.3e tol %.3e", e, tol)))
        end
        for k in ("norm", "sum")
            g, r = Float64(get(got, k, NaN)), Float64(get(ref, k, NaN))
            e = Verification.mixed_error(g, r; atol = 1e-12)
            err = max(err, e)
            push!(checks, (k, e <= tol, @sprintf("err %.3e tol %.3e", e, tol)))
        end
    end
    return Verification.VerificationReport(all(c -> c[2], checks), checks, err, tol)
end

end # module
