using Test, Statistics, Random
using AutonomousAI
const V = AutonomousAI.Verification
const C = AutonomousAI.Compute

@testset "mixed error is defined at zero" begin
    @test V.mixed_error(0.0, 0.0; atol = 1e-12) == 0.0
    @test isfinite(V.mixed_error(1e-18, 0.0; atol = 1e-12))
    @test V.mixed_error(1.0, 1.0 + 1e-9) < 1e-8
end

@testset "tolerance scales with precision and size" begin
    t64 = V.expected_tolerance(:matmul, :Float64, 10^6)
    t32 = V.expected_tolerance(:matmul, :Float32, 10^6)
    t16 = V.expected_tolerance(:matmul, :Float16, 10^6)
    @test t64 < t32 < t16
    @test V.expected_tolerance(:matmul, :Float32, 10^8) >
          V.expected_tolerance(:matmul, :Float32, 10^4)
    # transforms that change arithmetic widen the budget explicitly
    @test V.expected_tolerance(:matmul, :Float32, 10^6; transforms = [:fastmath]) >
          8 * t32 - 1e-30
end

@testset "reference implementations agree with properties" begin
    inp = C.make_inputs(:zscore_anomaly, [10_000], Float64)
    r = C.reference_run(:zscore_anomaly, inp)
    @test r.count > 0
    @test r.count < 10_000
    # planted outliers must be recovered by the oracle itself
    @test count(i -> r.mask[i], inp.planted) / length(inp.planted) > 0.9

    s = C.reference_run(:sum_reduction, C.make_inputs(:sum_reduction, [100_000], Float64))
    @test isfinite(s.s)
end

@testset "degenerate output fails even with small numeric error" begin
    n         = 10_000
    ref       = Dict{String,Any}("count" => 12, "indices" => collect(1:12), "index_sum" => 78.0, "planted" => collect(1:12))
    all_false = Dict{String,Any}("count" => 0, "indices" => Int[], "index_sum" => 0.0)
    rep       = AutonomousAI.Execute.compare_summaries(:zscore_anomaly, all_false, ref, :Float32, Symbol[], n)
    @test !rep.passed
    @test any(c -> c[1] == "non-degenerate" && !c[2], rep.checks)
end

@testset "Freivalds probe catches a wrong product" begin
    n     = 64
    A     = randn(MersenneTwister(1), n, n)
    B     = randn(MersenneTwister(2), n, n)
    probe = randn(MersenneTwister(AutonomousAI.Execute.PROBE_SEED), n)
    good  = Dict{String,Any}("freivalds" => (A * B) * probe, "norm" => sqrt(sum(abs2, A * B)), "sum" => sum(A * B))
    Cbad  = copy(A * B)
    Cbad[7, 9] += 1.0
    bad = Dict{String,Any}("freivalds" => Cbad * probe, "norm" => sqrt(sum(abs2, Cbad)), "sum" => sum(Cbad))
    @test AutonomousAI.Execute.compare_summaries(:matmul, good, good, :Float64, Symbol[], n * n).passed
    @test !AutonomousAI.Execute.compare_summaries(:matmul, bad, good, :Float64, Symbol[], n * n).passed
end
