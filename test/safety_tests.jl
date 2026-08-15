using Test
using AutonomousAI
const Sf = AutonomousAI.Safety
const S = AutonomousAI.Schema
const WM = AutonomousAI.WorldModel

@testset "resource envelope" begin
    lim = Sf.ResourceLimits(0.9, 0.9, 8 * 1024^3, 4 * 1024^3, 85.0, 300.0, 60.0, 30.0,
                            1024^3, 4)
    hot = WM.SystemState(AutonomousAI.AgentCore.Dates.now(), 0.5, 0.1, 1024^3,
                         16 * 1024^3, 0, 0, 95.0, NaN, NaN, 0, 0.0)
    ok, viol = Sf.within_limits(hot, lim)
    @test !ok
    @test any(v -> occursin("temperature", v), viol)
    # unmeasured telemetry is reported, never treated as compliant silently
    @test any(v -> occursin("unmeasured", v), viol)
end

@testset "capability gating" begin
    w = WM.WorldState()
    lim = Sf.default_limits(ram_total = w.hardware.profile.memory.total_bytes)
    caps = Sf.Capabilities([pwd()], [pwd()], false, Symbol[], false, false, false,
                           false, false)
    engine = Sf.PolicyEngine(Sf.Policy(lim, caps))
    gpu_act = S.GenerateCode(S.Candidate(:matmul, :cuda, :Float32))
    d = Sf.authorize(engine, gpu_act, w)
    @test !d.allowed
    @test occursin("gpu", d.reason)

    thr = S.GenerateCode(S.Candidate(:matmul, :cpu_threads, :Float32))
    @test !Sf.authorize(engine, thr, w).allowed
end

@testset "fastmath denial offers a downgrade rather than a hard stop" begin
    w = WM.WorldState()
    lim = Sf.default_limits(ram_total = w.hardware.profile.memory.total_bytes)
    engine = Sf.PolicyEngine(Sf.Policy(lim, Sf.default_capabilities();
                                       allow_fastmath = false))
    act = S.GenerateCode(S.Candidate(:zscore_anomaly, :cpu_simd, :Float32,
                                     [:fastmath], Dict{String,Int}()))
    d = Sf.authorize(engine, act, w)
    @test !d.allowed
    @test d.downgrade !== nothing
    @test !(:fastmath in d.downgrade.candidate.transforms)
end

@testset "footprint refusal precedes execution" begin
    w = WM.WorldState()
    lim = Sf.ResourceLimits(0.9, 0.9, 1024^2, 0, 85.0, 300.0, 60.0, 30.0, 1024^2, 4)
    engine = Sf.PolicyEngine(Sf.Policy(lim, Sf.default_capabilities()))
    act = S.BenchmarkAlgorithm(S.Candidate(:zscore_anomaly, :cpu_serial, :Float64),
                               [100_000_000], 3)
    d = Sf.authorize(engine, act, w)
    @test !d.allowed
    @test occursin("working set", d.reason) || occursin("RAM", d.reason)
end

@testset "emergency stop latches" begin
    es = Sf.EmergencyStop()
    @test !Sf.is_tripped(es)
    Sf.trip!(es, "thermal")
    @test Sf.is_tripped(es)
    @test_throws ErrorException Sf.check_stop(es)
    Sf.reset!(es)
    @test !Sf.is_tripped(es)
end
