"""
    Optimization

Cost model, estimator, benchmark harness and the candidate search (spec sections
9, 10, 11, 15, 33).

Three departures from a literal reading of the spec, each argued in
docs/06_design_review.md:

* **Hard constraints are not weights.**  `Cost = a*T + b*M + c*E + ...` lets a large
  speedup buy its way past a VRAM or thermal limit.  Feasibility is filtered first;
  the weighted objective only ranks survivors.
* **Compile time is a first-class cost, not an artefact.**  A Julia specialisation
  that runs 30% faster but costs 4 s to compile is a loss for a single 200 ms call
  and a win for 10^4 calls.  The estimator therefore optimises
  `T_compile + n_calls * T_warm`, with `n_calls` taken from the goal.
* **"speedup < 2% -> stop" is not implementable as stated.**  Benchmark noise on a
  loaded machine routinely exceeds 2%.  The stopping rule compares the *bootstrap
  confidence interval* of the ratio against the threshold, so the loop stops when
  the improvement is not distinguishable from noise rather than when a point
  estimate happens to land under a constant.
"""
module Optimization

using Statistics, LinearAlgebra, Random, Printf, Dates
using ..Schema
using ..HAL
using ..Memory: MemorySystem, predict_runtime
using ..Compute
using ..WorldModel

export BenchmarkStats, measure, MEASURE_HOOK, bootstrap_ci, ratio_ci, significantly_faster,
       CostModel, Estimate, estimate, cost, feasible, default_cost_model,
       enumerate_candidates, rank_candidates, UCBSelector, select!, update!,
       GPModel, posterior, expected_improvement, tune_discrete,
       chunk_plan, ChunkPlan, should_continue, OptimizationBudget

# ------------------------------------------------------------- benchmarking --

"""
Cold and warm timings are kept separate on purpose (spec section 11): the first
call in Julia includes type inference, LLVM codegen and native emission.
"""
struct BenchmarkStats
    compile_s::Float64
    min_s::Float64
    median_s::Float64
    mean_s::Float64
    mad_s::Float64
    samples::Vector{Float64}
    alloc_bytes::Int
    alloc_count::Int
    gc_s::Float64
end

function Base.show(io::IO, b::BenchmarkStats)
    print(io, @sprintf("BenchmarkStats(compile %.3fs, min %.6fs, median %.6fs +/- %.6f, %d samples, %d allocs / %.2f MiB)",
                       b.compile_s, b.min_s, b.median_s, b.mad_s, length(b.samples),
                       b.alloc_count, b.alloc_bytes / 1024^2))
end

"""
    measure(f, args...; samples=9, min_time=0.0) -> BenchmarkStats

In-process measurement for *trusted* code only (references, baselines).  Generated
code is measured through `Sandbox`, in a separate process, using the same statistics.

`min_s` is reported alongside the median because for a deterministic kernel the
minimum is the best estimator of the machine's capability, while the median carries
the interference the system will actually experience.  Decisions use the median;
regression detection uses the minimum.
"""
const MEASURE_HOOK = Ref{Union{Nothing,Function}}(nothing)

function measure(f::Function, args...; samples::Int = 9, warmup::Bool = true)
    hook = MEASURE_HOOK[]
    hook === nothing || return hook(f, args...; samples = samples)
    GC.gc()
    compile_s = @elapsed f(args...)
    warmup && f(args...)
    ts = Float64[]
    g0 = Base.gc_num()
    t_gc0 = Base.gc_time_ns()
    bytes = @allocated f(args...)
    for _ in 1:samples
        GC.gc(false)
        push!(ts, @elapsed f(args...))
    end
    g1 = Base.gc_num()
    gc_s = (Base.gc_time_ns() - t_gc0) / 1e9
    nallocs = g1.malloc + g1.poolalloc - g0.malloc - g0.poolalloc
    med = median(ts)
    return BenchmarkStats(compile_s, minimum(ts), med, mean(ts),
                          median(abs.(ts .- med)), ts, Int(bytes),
                          Int(max(nallocs, 0)), gc_s)
end

"Percentile bootstrap CI for a statistic of one sample set."
function bootstrap_ci(x::Vector{Float64}; stat::Function = median, n::Int = 2000,
                      alpha::Float64 = 0.05, rng = MersenneTwister(1))
    isempty(x) && return (NaN, NaN)
    m = length(x)
    vals = Vector{Float64}(undef, n)
    buf = Vector{Float64}(undef, m)
    for b in 1:n
        for i in 1:m
            buf[i] = x[rand(rng, 1:m)]
        end
        vals[b] = stat(buf)
    end
    sort!(vals)
    lo = vals[max(1, floor(Int, alpha / 2 * n))]
    hi = vals[min(n, ceil(Int, (1 - alpha / 2) * n))]
    return (lo, hi)
end

"""
    ratio_ci(baseline, candidate) -> (point, lo, hi)

Bootstrap CI of `T_baseline / T_candidate` (i.e. speedup).  `lo > 1` means the
candidate is faster with 95% confidence.
"""
function ratio_ci(baseline::Vector{Float64}, candidate::Vector{Float64};
                  n::Int = 2000, alpha::Float64 = 0.05, rng = MersenneTwister(2))
    (isempty(baseline) || isempty(candidate)) && return (NaN, NaN, NaN)
    vals = Vector{Float64}(undef, n)
    for b in 1:n
        num = median(baseline[rand(rng, 1:length(baseline), length(baseline))])
        den = median(candidate[rand(rng, 1:length(candidate), length(candidate))])
        vals[b] = den > 0 ? num / den : NaN
    end
    filter!(isfinite, vals)
    isempty(vals) && return (NaN, NaN, NaN)
    sort!(vals)
    point = median(baseline) / median(candidate)
    lo = vals[max(1, floor(Int, alpha / 2 * length(vals)))]
    hi = vals[min(length(vals), ceil(Int, (1 - alpha / 2) * length(vals)))]
    return (point, lo, hi)
end

"""
    significantly_faster(baseline, candidate; min_improvement) -> (Bool, point, lo, hi)

The decision rule of section 33 made statistical: accept only when the *lower*
bound of the speedup CI clears `1 + min_improvement`.
"""
function significantly_faster(baseline::BenchmarkStats, candidate::BenchmarkStats;
                              min_improvement::Float64 = 0.02)
    point, lo, hi = ratio_ci(baseline.samples, candidate.samples)
    return (isfinite(lo) && lo > 1 + min_improvement, point, lo, hi)
end

# --------------------------------------------------------------- estimation --

"Prior estimate of what a candidate will cost, before it is measured."
struct Estimate
    runtime_s::Float64
    compile_s::Float64
    confidence::Float64
    ram_bytes::Int
    vram_bytes::Int
    transfer_bytes::Int
    energy_j::Float64
    rel_error::Float64
    failure_prob::Float64
    source::Symbol            # :measured, :fitted, :roofline
end

const KERNEL_LAUNCH_S = 5.0e-6
const PCIE_GBS = 25.0
const COMPILE_PRIOR_S = Dict{Symbol,Float64}(:cpu_serial => 0.35, :cpu_simd => 0.45,
    :cpu_threads => 0.8, :cpu_blas => 0.2, :cuda => 3.5, :distributed => 1.5)

"""
    estimate(candidate, size, world, mem) -> Estimate

Measurement first (`HardwareMemory`), power-law fit second, analytic roofline last.
The `source` field is propagated into the decision log so that a bad decision can be
traced to the evidence class it rested on.
"""
function estimate(c::Schema.Candidate, size::Vector{Int}, w::WorldModel.WorldState,
                  ms::MemorySystem)
    n = prod(size)
    hw = w.hardware.fingerprint
    sw = HAL.software_fingerprint()
    t_mem, conf = predict_runtime(ms.hardware, c, n; hw = hw, sw = sw)
    flops = Compute.flops_for(c.algorithm, size)
    bytes = Compute.bytes_for(c.algorithm, size, c.precision)
    profile = w.hardware.profile
    gpu = Schema.is_gpu_backend(c.backend)
    width = c.precision === :Float64 ? 8 : c.precision === :Float32 ? 4 : 2
    ram = gpu ? 2 * n * width : 3 * n * width
    vram = gpu ? 3 * n * width : 0
    transfer = gpu ? 2 * n * width : 0
    t_roof = roofline_seconds(c, profile, flops, bytes, transfer)
    t = if conf >= 0.6
        t_mem
    elseif conf > 0.0 && isfinite(t_mem)
        (conf * t_mem + (1 - conf) * t_roof)
    else
        t_roof
    end
    src = conf >= 0.6 ? :measured : (conf > 0.0 ? :fitted : :roofline)
    eps_p = Schema.precision_eps(c.precision)
    err = eps_p * sqrt(max(n, 1)) * (c.algorithm === :sum_reduction ? 1.0 : 4.0)
    (:fastmath in c.transforms) && (err *= 8)
    fail = gpu ? 0.05 : 0.01
    (:fastmath in c.transforms) && (fail += 0.05)
    power = gpu ? 220.0 : 45.0
    return Estimate(t, get(COMPILE_PRIOR_S, c.backend, 0.5),
                    max(conf, 0.1), ram, vram, transfer, t * power, err, fail, src)
end

function roofline_seconds(c::Schema.Candidate, p::HAL.HardwareProfile, flops::Float64,
                          bytes::Float64, transfer::Int)
    if Schema.is_gpu_backend(c.backend) && !isempty(p.gpus)
        g = p.gpus[1]
        peak = HAL.peak_flops(g, c.precision)
        bw = (isnan(g.memory_bandwidth_gbs) ? 600.0 : g.memory_bandwidth_gbs) * 1e9
        t_kernel = max(flops / peak, bytes / bw) + KERNEL_LAUNCH_S
        return t_kernel + transfer / (PCIE_GBS * 1e9)
    end
    cores = c.backend === :cpu_threads ? min(p.cpu.physical_cores, Threads.nthreads()) : 1
    lanes = c.backend === :cpu_serial ? 1.0 : 1.0
    peak = HAL.peak_flops(p.cpu, c.precision) * (cores / max(p.cpu.physical_cores, 1)) * lanes
    c.backend === :cpu_serial && (peak /= max(p.cpu.simd_width_bits / 64, 1))
    bw = 20.0e9 * (c.backend === :cpu_threads ? 2.0 : 1.0)
    return max(flops / max(peak, 1.0), bytes / bw)
end

# --------------------------------------------------------------- cost model --

"""
Weighted, *normalised* objective over feasible candidates only.  Every term is
divided by a reference scale so the weights are dimensionless and comparable; the
spec's raw `a*Runtime + b*Memory` sums seconds to bytes.
"""
struct CostModel
    w_runtime::Float64
    w_memory::Float64
    w_energy::Float64
    w_error::Float64
    w_risk::Float64
    w_compile::Float64
    t_ref::Float64
    mem_ref::Float64
    energy_ref::Float64
    error_ref::Float64
    n_calls::Int
end

default_cost_model(; t_ref = 1.0, mem_ref = 1024^3, energy_ref = 100.0,
                   error_ref = 1e-6, n_calls = 1) =
    CostModel(1.0, 0.2, 0.15, 0.4, 0.3, 0.25, t_ref, mem_ref, energy_ref,
              error_ref, n_calls)

"""
    cost(model, est) -> Float64

Amortised: `T = T_compile + n_calls * T_warm`.  With `n_calls = 1` the model
naturally prefers the interpreter-cheap candidate; with `n_calls` large it pays for
compilation.
"""
function cost(m::CostModel, e::Estimate)
    total_t = e.compile_s * m.w_compile + m.n_calls * e.runtime_s
    return m.w_runtime * (total_t / m.t_ref) +
           m.w_memory * ((e.ram_bytes + e.vram_bytes) / m.mem_ref) +
           m.w_energy * (e.energy_j / m.energy_ref) +
           m.w_error * (e.rel_error / m.error_ref) +
           m.w_risk * e.failure_prob
end

"""
    feasible(est, limits; vram_available) -> (Bool, reason)

Hard gate applied *before* ranking.  Returning a reason string keeps infeasibility
explainable in the decision log.
"""
function feasible(e::Estimate, limits; vram_available::Int = 0)
    e.ram_bytes > limits.max_ram_bytes &&
        return (false, @sprintf("RAM %.2f GiB > %.2f GiB", e.ram_bytes / 1024^3,
                                limits.max_ram_bytes / 1024^3))
    if e.vram_bytes > 0
        cap = limits.max_vram_bytes > 0 ? min(limits.max_vram_bytes, vram_available) :
              vram_available
        cap <= 0 && return (false, "no VRAM budget configured")
        e.vram_bytes > cap &&
            return (false, @sprintf("VRAM %.2f GiB > %.2f GiB", e.vram_bytes / 1024^3,
                                    cap / 1024^3))
    end
    (e.compile_s + e.runtime_s) > limits.max_runtime_s &&
        return (false, @sprintf("estimated %.1fs > runtime ceiling %.1fs",
                                e.compile_s + e.runtime_s, limits.max_runtime_s))
    return (true, "feasible")
end

# ------------------------------------------------------ candidate selection --

"""
    enumerate_candidates(alg, world; precisions, transforms) -> Vector{Candidate}

The search space is *enumerated by the runtime*, not proposed by the model.  The
LLM's role is to reorder and prune this list (a prior), which bounds the worst case:
a hallucinated backend simply is not in the set.
"""
function enumerate_candidates(alg::Symbol, w::WorldModel.WorldState;
                              precisions::Vector{Symbol} = [:Float64, :Float32],
                              transform_sets::Vector{Vector{Symbol}} = [Symbol[],
                                  [:inbounds], [:inbounds, :simd]])
    backends = Compute.available_backends(w.hardware.profile)
    out = Schema.Candidate[]
    ok = CodeGenerationApplicable(alg)
    for b in backends, p in precisions, ts in transform_sets
        all(t -> t in ok, ts) || continue
        (p === :Float16 || p === :BFloat16) && !Schema.is_gpu_backend(b) && continue
        push!(out, Schema.Candidate(alg, b, p, copy(ts), Dict{String,Int}()))
    end
    return out
end

"Indirection so `Optimization` does not depend on `CodeGeneration` at load time."
const APPLICABLE_TRANSFORMS = Ref{Function}(alg -> Set{Symbol}())
CodeGenerationApplicable(alg::Symbol) = APPLICABLE_TRANSFORMS[](alg)

"""
    rank_candidates(cands, size, world, mem, model, limits) -> Vector{NamedTuple}

Returns feasible candidates sorted by cost, each with its estimate and evidence
class, plus the infeasible ones with their reason (kept for the decision log).
"""
function rank_candidates(cands::Vector{Schema.Candidate}, size::Vector{Int},
                         w::WorldModel.WorldState, ms::MemorySystem,
                         model::CostModel, limits)
    vram = isempty(w.hardware.profile.gpus) ? 0 : w.hardware.profile.gpus[1].vram_total_bytes
    scored = NamedTuple[]
    rejected = NamedTuple[]
    for c in cands
        e = estimate(c, size, w, ms)
        ok, why = feasible(e, limits; vram_available = vram)
        if ok
            push!(scored, (candidate = c, estimate = e, cost = cost(model, e),
                           source = e.source))
        else
            push!(rejected, (candidate = c, estimate = e, reason = why))
        end
    end
    sort!(scored; by = x -> x.cost)
    return scored, rejected
end

"""
UCB1 over a finite candidate set, seeded with the cost-model prior.

Why a bandit rather than "benchmark everything": benchmarking is the expensive
operation.  UCB spends measurements on candidates that are either promising or
poorly known, and provably bounds cumulative regret, which is the right guarantee
for an agent that must stop after a fixed budget.
"""
mutable struct UCBSelector
    candidates::Vector{Schema.Candidate}
    counts::Vector{Int}
    means::Vector{Float64}       # mean observed cost (lower is better)
    prior::Vector{Float64}
    total::Int
    c::Float64
end

function UCBSelector(cands::Vector{Schema.Candidate}, prior_costs::Vector{Float64};
                     c::Float64 = 1.0)
    p = copy(prior_costs)
    mx = maximum(p)
    mn = minimum(p)
    norm = mx > mn ? (p .- mn) ./ (mx - mn) : zeros(length(p))
    return UCBSelector(cands, zeros(Int, length(cands)), copy(norm), norm, 0, c)
end

function select!(s::UCBSelector)
    i = findfirst(==(0), s.counts)
    i === nothing || return (i, s.candidates[i])
    scores = [s.means[k] - s.c * sqrt(2 * log(s.total) / s.counts[k])
              for k in eachindex(s.candidates)]
    k = argmin(scores)
    return (k, s.candidates[k])
end

function update!(s::UCBSelector, i::Int, observed_cost::Float64)
    s.counts[i] += 1
    s.total += 1
    n = s.counts[i]
    s.means[i] = ((n - 1) * s.means[i] + observed_cost) / n
    return s
end

# ----------------------------------------------- Bayesian parameter tuning --

"Zero-dependency GP with an RBF kernel; used for 1-D tuning (chunk, tile, block)."
struct GPModel
    x::Vector{Float64}
    y::Vector{Float64}
    lengthscale::Float64
    sigma_f::Float64
    sigma_n::Float64
end

rbf(a, b, l, sf) = sf^2 * exp(-0.5 * ((a - b) / l)^2)

function posterior(gp::GPModel, xs::Vector{Float64})
    n = length(gp.x)
    if n == 0
        return (zeros(length(xs)), fill(gp.sigma_f^2, length(xs)))
    end
    K = [rbf(gp.x[i], gp.x[j], gp.lengthscale, gp.sigma_f) for i in 1:n, j in 1:n]
    K += (gp.sigma_n^2 + 1e-10) * I
    F = cholesky(Symmetric(K))
    ybar = mean(gp.y)
    alpha = F \ (gp.y .- ybar)
    mu = Vector{Float64}(undef, length(xs))
    va = Vector{Float64}(undef, length(xs))
    for (t, xt) in enumerate(xs)
        k = [rbf(xt, gp.x[i], gp.lengthscale, gp.sigma_f) for i in 1:n]
        mu[t] = ybar + dot(k, alpha)
        v = F.L \ k
        va[t] = max(gp.sigma_f^2 - dot(v, v), 1e-12)
    end
    return (mu, va)
end

"Abramowitz-Stegun 7.1.26 -- avoids a SpecialFunctions dependency."
function _erf(x::Float64)
    s = sign(x)
    z = abs(x)
    t = 1 / (1 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
                t * (-1.453152027 + t * 1.061405429))))
    return s * (1 - poly * exp(-z * z))
end
_normcdf(z::Float64) = 0.5 * (1 + _erf(z / sqrt(2)))
_normpdf(z::Float64) = exp(-0.5z^2) / sqrt(2pi)

"Expected improvement for *minimisation*."
function expected_improvement(mu::Vector{Float64}, var::Vector{Float64},
                              best::Float64; xi::Float64 = 0.01)
    ei = similar(mu)
    for i in eachindex(mu)
        sd = sqrt(var[i])
        if sd < 1e-12
            ei[i] = 0.0
            continue
        end
        z = (best - mu[i] - xi) / sd
        ei[i] = (best - mu[i] - xi) * _normcdf(z) + sd * _normpdf(z)
    end
    return ei
end

"""
    tune_discrete(objective, grid; budget) -> (best_x, best_y, trace)

GP-EI over a discrete grid (chunk size, tile size, block size).  Inputs are mapped
to log2 space because these parameters act multiplicatively.
"""
function tune_discrete(objective::Function, grid::Vector{Int}; budget::Int = 8,
                       rng = MersenneTwister(3))
    xs = Float64[log2(max(g, 1)) for g in grid]
    seen = Int[]
    X, Y = Float64[], Float64[]
    trace = Tuple{Int,Float64}[]
    first_idx = clamp(div(length(grid) + 1, 2), 1, length(grid))
    order = [first_idx, 1, length(grid)]
    for step in 1:budget
        idx = if step <= length(order)
            order[step]
        else
            gp = GPModel(X, Y, max(std(xs), 1.0), max(std(Y), 1.0), 0.05 * max(mean(Y), 1e-9))
            mu, va = posterior(gp, xs)
            ei = expected_improvement(mu, va, minimum(Y))
            for s in seen
                ei[s] = -Inf
            end
            argmax(ei)
        end
        idx in seen && (idx = something(findfirst(i -> !(i in seen), eachindex(grid)), idx))
        idx in seen && break
        y = objective(grid[idx])
        push!(seen, idx)
        push!(X, xs[idx])
        push!(Y, y)
        push!(trace, (grid[idx], y))
    end
    isempty(Y) && return (grid[1], NaN, trace)
    b = argmin(Y)
    return (grid[seen[b]], Y[b], trace)
end

# ------------------------------------------------------ out-of-core planning --

struct ChunkPlan
    n_total::Int
    chunk_elements::Int
    n_chunks::Int
    bytes_per_chunk::Int
    double_buffered::Bool
    rationale::String
end

"""
    chunk_plan(n_total, precision, vram_bytes; safety=0.6, buffers=3) -> ChunkPlan

The 100 GB workflow of section 39.  VRAM is not divided by the dataset size but by
`buffers * element_width`, with a safety factor for the allocator, the CUDA context
and fragmentation; double buffering costs one extra buffer and hides the PCIe
transfer behind the kernel, which is the difference between a bandwidth-bound and a
latency-bound out-of-core run.
"""
function chunk_plan(n_total::Int, precision::Symbol, vram_bytes::Int;
                    safety::Float64 = 0.6, buffers::Int = 3,
                    double_buffered::Bool = true)
    w = precision === :Float64 ? 8 : precision === :Float32 ? 4 : 2
    budget = floor(Int, vram_bytes * safety)
    eff_buffers = buffers * (double_buffered ? 2 : 1)
    chunk = max(1 << 16, div(budget, max(eff_buffers * w, 1)))
    chunk = min(chunk, n_total)
    nch = cld(n_total, chunk)
    return ChunkPlan(n_total, chunk, nch, chunk * w, double_buffered,
                     @sprintf("VRAM %.1f GiB * %.2f safety / (%d buffers * %d B) -> %d elements/chunk, %d chunks",
                              vram_bytes / 1024^3, safety, eff_buffers, w, chunk, nch))
end

# ------------------------------------------------------------ stopping rule --

mutable struct OptimizationBudget
    max_iterations::Int
    max_benchmark_s::Float64
    max_compile_s::Float64
    min_relative_improvement::Float64
    iterations::Int
    benchmark_s::Float64
    compile_s::Float64
    stalled::Int
end

OptimizationBudget(; max_iterations = 8, max_benchmark_s = 600.0, max_compile_s = 300.0,
                   min_relative_improvement = 0.02) =
    OptimizationBudget(max_iterations, max_benchmark_s, max_compile_s,
                       min_relative_improvement, 0, 0.0, 0.0, 0)

"""
    should_continue(budget, improved) -> (Bool, reason)

Terminates on budget exhaustion or on two consecutive statistically insignificant
improvements.  `improved` must come from `significantly_faster`, not from comparing
point estimates.
"""
function should_continue(b::OptimizationBudget, improved::Bool)
    improved ? (b.stalled = 0) : (b.stalled += 1)
    b.iterations >= b.max_iterations &&
        return (false, "iteration budget exhausted ($(b.max_iterations))")
    b.benchmark_s >= b.max_benchmark_s &&
        return (false, @sprintf("benchmark time budget exhausted (%.0fs)", b.max_benchmark_s))
    b.compile_s >= b.max_compile_s &&
        return (false, @sprintf("compile time budget exhausted (%.0fs)", b.max_compile_s))
    b.stalled >= 2 &&
        return (false, "no statistically significant improvement in 2 consecutive rounds")
    return (true, "continue")
end

end # module
