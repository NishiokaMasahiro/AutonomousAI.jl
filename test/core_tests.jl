using Test
using AutonomousAI
const S = AutonomousAI.Schema
const MJ = AutonomousAI.MiniJSON
const M = AutonomousAI.Memory

@testset "MiniJSON round trip" begin
    for v in (1, -3, 2.5, "abc", true, nothing, [1, 2, 3],
              Dict("a" => 1, "b" => [1.5, "x"], "c" => Dict("d" => false)))
        @test MJ.parse_json(MJ.to_json(v)) == (v isa Dict ? Dict{String,Any}(v) : v)
    end
    @test MJ.parse_json("{\"a\": [1, {\"b\": null}]}")["a"][2]["b"] === nothing
    @test MJ.parse_json("\"\\u00e9\"") == "\u00e9"
    @test_throws MJ.JSONError MJ.parse_json("{\"a\": }")
    @test_throws MJ.JSONError MJ.parse_json("[1,2] junk")
    # NaN has no JSON representation and must not silently become 0.0
    @test MJ.parse_json(MJ.to_json(Dict("x" => NaN)))["x"] === nothing
end

@testset "Schema is a closed action space" begin
    ok = """{"action":"benchmark_algorithm",
             "candidate":{"algorithm":"zscore_anomaly","backend":"cpu_simd",
                          "precision":"Float32","transforms":["inbounds"]},
             "input_size":[1024],"samples":5}"""
    a = S.parse_action(ok)
    @test a isa S.BenchmarkAlgorithm
    @test a.candidate.backend === :cpu_simd

    # every one of these must be rejected, not "interpreted charitably"
    bad = [
        """{"action":"run_shell","cmd":"rm -rf /"}""",
        """{"action":"benchmark_algorithm","candidate":{"algorithm":"rm_rf",
            "backend":"cpu_simd","precision":"Float32"},"input_size":[8],"samples":1}""",
        """{"action":"benchmark_algorithm","candidate":{"algorithm":"zscore_anomaly",
            "backend":"quantum","precision":"Float32"},"input_size":[8],"samples":1}""",
        """{"action":"benchmark_algorithm","candidate":{"algorithm":"zscore_anomaly",
            "backend":"cpu_simd","precision":"Float128"},"input_size":[8],"samples":1}""",
        """{"action":"generate_code","candidate":{"algorithm":"zscore_anomaly",
            "backend":"cpu_simd","precision":"Float32","transforms":["eval"]}}""",
        """{"action":"generate_code","candidate":{"algorithm":"zscore_anomaly",
            "backend":"cpu_simd","precision":"Float32"},"extra":"payload"}""",
        """{"action":"benchmark_algorithm","candidate":{"algorithm":"zscore_anomaly",
            "backend":"cpu_simd","precision":"Float32"},"input_size":[-1],"samples":1}""",
        """{"action":"benchmark_algorithm","candidate":{"algorithm":"zscore_anomaly",
            "backend":"cpu_simd","precision":"Float32"},"input_size":[8],
            "samples":100000}""",
    ]
    for b in bad
        @test_throws S.SchemaError S.parse_action(b)
    end
end

@testset "Schema serialisation round trip" begin
    c = S.Candidate(:matmul, :cpu_blas, :Float64, [:inbounds], Dict("tile" => 64))
    a = S.ExecuteFinal(c, [512, 512])
    a2 = S.parse_action(MJ.to_json(S.to_dict(a)))
    @test a2.candidate == c
    @test a2.input_size == [512, 512]
end

@testset "Memory separation and keying" begin
    wm = M.WorkingMemory(capacity_bytes = 4096)
    M.remember!(wm, "k", collect(1:10))
    @test M.recall(wm, "k") == collect(1:10)
    @test M.recall(wm, "missing") === nothing
    M.forget!(wm, "k")
    @test M.recall(wm, "k") === nothing

    hm = M.HardwareMemory()
    c = S.Candidate(:stencil3, :cpu_simd, :Float32)
    mk(n, t, hw, sw) = M.BenchmarkRecord(:stencil3, :cpu_simd, :Float32, Symbol[], n,
        [n], 0.1, t, t * 1.05, 0.0, 5, 0, 0, 1.0, 0, 0, 0, NaN, NaN, false, hw, sw,
        AutonomousAI.AgentCore.Dates.now())
    for (n, t) in [(1000, 1e-5), (10_000, 1e-4), (100_000, 1e-3)]
        M.record!(hm, mk(n, t, "hwA", "swA"))
    end
    @test length(M.lookup(hm, c; hw = "hwA", sw = "swA")) == 3
    # a timing from another machine must never be returned for this one
    @test isempty(M.lookup(hm, c; hw = "hwB", sw = "swA"))
    # nor after a software-stack change
    @test isempty(M.lookup(hm, c; hw = "hwA", sw = "swB"))

    t, conf = M.predict_runtime(hm, c, 10_000; hw = "hwA", sw = "swA")
    @test isapprox(t, 1e-4; rtol = 0.2)
    @test conf > 0.5
    t2, conf2 = M.predict_runtime(hm, c, 1_000_000; hw = "hwA", sw = "swA")
    @test t2 > 1e-3                  # power-law extrapolation
    @test conf2 < conf               # and it says it is less sure
    t3, conf3 = M.predict_runtime(hm, c, 1000; hw = "unknown", sw = "swA")
    @test isnan(t3) && conf3 == 0.0
end

@testset "Goal is immutable" begin
    g = AAI.AgentCore.Goal("find outliers"; input_size = [1024])
    @test g.algorithm === :zscore_anomaly
    @test !ismutable(g)
    @test AAI.AgentCore.infer_algorithm("robust median based detection") === :mad_anomaly
    @test AAI.AgentCore.infer_algorithm("multiply two big matrices") === :matmul
end

@testset "Compute.parallel_for! uses JACC only" begin
    # GPUバックエンドでは「ホスト配列を捕捉するクロージャ」は不正。
    # ここでは JACC 呼び出し自体が成立することのみ確認する。
    n = 10_000
    @test isnothing(AutonomousAI.Compute.parallel_for!(i -> nothing, n; backend=:jacc))

    @test_throws ErrorException AutonomousAI.Compute.parallel_for!(i -> nothing, 16; backend=:threads)
end
