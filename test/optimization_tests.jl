using Test, Statistics, Random
using AutonomousAI
const O = AutonomousAI.Optimization
const S = AutonomousAI.Schema

@testset "cold and warm timings are separated" begin
    f(x) = sum(abs2, x)
    st = O.measure(f, randn(10_000); samples = 5)
    @test st.compile_s >= st.min_s
    @test st.min_s <= st.median_s
    @test length(st.samples) == 5
end

@testset "bootstrap decides significance, not point estimates" begin
    rng = MersenneTwister(7)
    slow = 1.0 .+ 0.30 .* randn(rng, 60) .* 0.1
    noisy_same = 1.0 .+ 0.30 .* randn(rng, 60) .* 0.1
    fast = 0.5 .+ 0.05 .* randn(rng, 60) .* 0.1
    a = O.BenchmarkStats(0.0, minimum(slow), median(slow), mean(slow), 0.0, slow, 0, 0, 0.0)
    b = O.BenchmarkStats(0.0, minimum(fast), median(fast), mean(fast), 0.0, fast, 0, 0, 0.0)
    c = O.BenchmarkStats(0.0, minimum(noisy_same), median(noisy_same),
                         mean(noisy_same), 0.0, noisy_same, 0, 0, 0.0)
    sig, point, lo, hi = O.significantly_faster(a, b)
    @test sig && point > 1.5 && lo > 1
    sig2, _, lo2, _ = O.significantly_faster(a, c)
    @test !sig2
end

@testset "hard constraints are not weights" begin
    e = O.Estimate(0.001, 0.1, 0.9, 64 * 1024^3, 0, 0, 1.0, 1e-9, 0.01, :measured)
    lim = AutonomousAI.Safety.ResourceLimits(0.9, 0.9, 8 * 1024^3, 0, 85.0, 300.0,
                                             60.0, 30.0, 1024^3, 4)
    ok, why = O.feasible(e, lim)
    @test !ok && occursin("RAM", why)
    # no weighting can rescue it: it never reaches the cost function
    cheap = O.cost(O.default_cost_model(), e)
    @test isfinite(cheap)
end

@testset "cost model amortises compilation over n_calls" begin
    slow_compile_fast_run = O.Estimate(0.001, 4.0, 0.9, 0, 0, 0, 1.0, 1e-9, 0.01, :roofline)
    fast_compile_slow_run = O.Estimate(0.010, 0.2, 0.9, 0, 0, 0, 1.0, 1e-9, 0.01, :roofline)
    one_shot = O.default_cost_model(n_calls = 1)
    many = O.default_cost_model(n_calls = 10_000)
    @test O.cost(one_shot, fast_compile_slow_run) < O.cost(one_shot, slow_compile_fast_run)
    @test O.cost(many, slow_compile_fast_run) < O.cost(many, fast_compile_slow_run)
end

@testset "UCB explores before it exploits" begin
    cands = [S.Candidate(:stencil3, b, :Float32) for b in
             (:cpu_serial, :cpu_simd, :cpu_blas)]
    sel = O.UCBSelector(cands, [1.0, 0.5, 0.8])
    seen = Set{Int}()
    for _ in 1:3
        i, _ = O.select!(sel)
        push!(seen, i)
        O.update!(sel, i, i == 2 ? 0.1 : 0.9)
    end
    @test length(seen) == 3                     # every arm tried once first
    for _ in 1:20
        i, _ = O.select!(sel)
        O.update!(sel, i, i == 2 ? 0.1 : 0.9)
    end
    @test sel.counts[2] > sel.counts[1]         # then concentrates on the winner
end

@testset "GP tuner finds a discrete optimum" begin
    grid = [2^k for k in 6:16]
    truth(x) = (log2(x) - 12.0)^2 + 1.0
    best, besty, trace = O.tune_discrete(truth, grid; budget = 8)
    @test abs(log2(best) - 12.0) <= 1.0
    @test besty <= truth(grid[1])
    @test length(trace) <= 8
end

@testset "chunk plan respects VRAM, not dataset size" begin
    p = O.chunk_plan(div(100_000_000_000, 4), :Float32, 12 * 1024^3)
    @test p.n_chunks > 1
    @test p.bytes_per_chunk < 12 * 1024^3
    @test p.chunk_elements * p.n_chunks >= p.n_total
end

@testset "stopping rule tolerates noise" begin
    b = O.OptimizationBudget(max_iterations = 100)
    cont, why = O.should_continue(b, false)
    @test cont
    cont, why = O.should_continue(b, false)
    @test !cont && occursin("significant", why)
end
