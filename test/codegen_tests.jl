using Test
using AutonomousAI
const CG = AutonomousAI.CodeGeneration
const S = AutonomousAI.Schema

@testset "validator rejects escape attempts" begin
    hostile = [
        :(eval(:(rm("/", recursive = true)))),
        :(ccall(:system, Cint, (Cstring,), "curl evil.example")),
        :(run(`sh -c "cat /etc/passwd"`)),
        :(open("/etc/shadow")),
        :(unsafe_load(pointer(x))),
        :(include("payload.jl")),
        :(Base.invokelatest(f)),
        :(Core.eval(Main, :(1 + 1))),
        :(module Sneaky end),
        :(x = read("/etc/passwd", String)),
        :(@eval f() = 1),
        :(getfield(Main, :ENV)),
    ]
    for ex in hostile
        r = CG.validate(ex)
        @test !r.ok
        @test !isempty(r.errors)
    end
end

@testset "validator accepts the template it generated" begin
    for alg in [:zscore_anomaly, :mad_anomaly, :matmul, :sum_reduction, :stencil3]
        c = S.Candidate(alg, :cpu_serial, :Float64)
        u = CG.generate(c)
        @test u.report.ok
        @test occursin(CG.entrypoint_name(alg), u.source)
    end
end

@testset "policy gates on transforms" begin
    c = S.Candidate(:zscore_anomaly, :cpu_simd, :Float32, [:fastmath], Dict{String,Int}())
    @test !CG.generate(c; allow_fastmath = false).report.ok
    @test CG.generate(c; allow_fastmath = true).report.ok

    ci = S.Candidate(:zscore_anomaly, :cpu_simd, :Float32, [:inbounds], Dict{String,Int}())
    @test CG.generate(ci; allow_inbounds = true).report.ok
    @test !CG.generate(ci; allow_inbounds = false).report.ok
end

@testset "inbounds requires a discharged guard" begin
    # hand-written unit with @inbounds but no axes guard: must be rejected
    unguarded = quote
        function f(x, out)
            for i in eachindex(x)
                @inbounds out[i] = x[i]
            end
            return out
        end
    end
    r = CG.validate(unguarded)
    @test !r.ok
    @test any(e -> occursin("axes guard", e), r.errors)

    guarded = quote
        function f(x, out)
            _axes_guard(x, out)
            for i in eachindex(x)
                @inbounds out[i] = x[i]
            end
            return out
        end
    end
    @test CG.validate(guarded).ok
end

@testset "AST transforms are structural, not textual" begin
    c = S.Candidate(:zscore_anomaly, :cpu_simd, :Float64, [:inbounds, :simd],
                    Dict{String,Int}())
    u = CG.generate(c)
    @test u.report.ok
    @test occursin("@inbounds", u.source)
    @test occursin("@simd", u.source)

    # fma contraction rewrites a*b+c
    ex = :(y = a * b + c)
    out, notes = CG.apply_transforms(ex, :matmul, [:fma])
    @test occursin("muladd", string(out))
    @test any(n -> occursin("rounding", n), notes)

    # a transform outside its applicability set is refused, not silently dropped
    _, notes2 = CG.apply_transforms(ex, :mad_anomaly, [:fastmath])
    @test any(n -> occursin("not applicable", n), notes2)
end

@testset "GPU code generation is absent, not faked" begin
    c = S.Candidate(:zscore_anomaly, :cuda, :Float32)
    if isempty(CG.GPU_TEMPLATES)
        @test_throws S.SchemaError CG.generate(c)
    else
        @test CG.generate(c).report.ok
    end
end
