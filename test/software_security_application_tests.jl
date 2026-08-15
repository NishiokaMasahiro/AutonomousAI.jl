using Test
using AutonomousAI

const CG = AutonomousAI.CodeGeneration
const S = AutonomousAI.Schema
const Sf = AutonomousAI.Safety
const WM = AutonomousAI.WorldModel

@testset "software security application" begin
    hostile = """
    function pwn(x)
        run(`sh -c \"id\"`)
        return x
    end
    """
    r = CG.validate_source(hostile)
    @test !r.ok
    @test any(e -> occursin("forbidden", e), r.errors)

    w = WM.WorldState()
    lim = Sf.default_limits(ram_total = w.hardware.profile.memory.total_bytes)
    caps = Sf.Capabilities([pwd()], [pwd()], false, Symbol[], false, true, false,
                           false, false)
    engine = Sf.PolicyEngine(Sf.Policy(lim, caps))

    req = S.RequestPermission(:network, "upload benchmark traces")
    d_req = Sf.authorize(engine, req, w)
    @test !d_req.allowed
    @test occursin("out-of-band human grant", d_req.reason)

    strict = Sf.ResourceLimits(0.95, 0.98, 1024^2, 0, 90.0, 400.0, 60.0, 30.0,
                               1024^2, 2)
    strict_engine = Sf.PolicyEngine(Sf.Policy(strict, Sf.default_capabilities()))
    act = S.ExecuteFinal(S.Candidate(:matmul, :cpu_serial, :Float64), [100_000, 100_000])
    d_size = Sf.authorize(strict_engine, act, w)
    @test !d_size.allowed
    @test occursin("working set", d_size.reason) || occursin("RAM", d_size.reason)
end
