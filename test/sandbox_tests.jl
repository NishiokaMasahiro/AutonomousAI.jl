using Test
using AutonomousAI
const SB = AutonomousAI.Sandbox
const S = AutonomousAI.Schema
const E = AutonomousAI.Execute

@testset "sandbox" begin
    if !SB.sandbox_available()
        @info "no julia binary reachable; sandbox tests skipped"
    else
        spec = SB.SandboxSpec(AutonomousAI.Safety.default_limits(); isolation = :process)

        @testset "result marker round trip" begin
            r = SB.run_sandboxed("", "println(RESULT_MARKER * \"{\\\"x\\\":41}\")", spec)
            @test r.ok
            @test r.payload["x"] == 41
        end

        @testset "a crashing unit is a failed run, not a crashed agent" begin
            r = SB.run_sandboxed("", "error(\"boom\")", spec)
            @test !r.ok
            @test r.exit_code != 0
        end

        @testset "wall-clock kill" begin
            spec2 = SB.SandboxSpec(:process, "", joinpath(Sys.BINDIR, "julia"), 3.0,
                                   2 * 1024^2, 10, 1024, 1, false)
            r = SB.run_sandboxed("", "sleep(60)", spec2)
            @test !r.ok
            @test r.killed
            @test r.wall_s < 20
        end

        @testset "generated unit runs and is measured out of process" begin
            c = S.Candidate(:stencil3, :cpu_serial, :Float64)
            run = E.run_candidate(c, [50_000], spec; samples = 3)
            @test run.ok
            @test run.stats.median_s > 0
            @test haskey(run.summary, "norm")
            ref = E.reference_summary(:stencil3, [50_000], Float64)
            rep = E.compare_summaries(:stencil3, run.summary, ref, :Float64,
                                      Symbol[], 50_000)
            @test rep.passed
        end

        @testset "sandbox and host fixtures are identical" begin
            # If these ever diverge, every verification verdict becomes meaningless.
            c = S.Candidate(:sum_reduction, :cpu_serial, :Float64)
            run = E.run_candidate(c, [10_000], spec; samples = 3)
            @test run.ok
            ref = E.reference_summary(:sum_reduction, [10_000], Float64)
            @test isapprox(run.summary["value"], ref["value"]; rtol = 1e-12)
        end
    end
end
