# LLM planning benchmark for AutonomousAI.jl
#
# Compares Mimase (offline heuristic) with external providers such as
# Anthropic Opus 5 and Gemini 3.7 on the *planning* task.
#
# Run:
#   julia --project=. benchmark/llm_model_comparison.jl
#
# Optional environment variables:
#   AAI_BENCH_REPEATS=3
#   AAI_OPUS_MODEL=claude-opus-5
#   AAI_GEMINI_MODEL=gemini-3.7-pro
#   AAI_GEMINI_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
#   ANTHROPIC_API_KEY=...
#   GEMINI_API_KEY=...

using AutonomousAI
using Printf, Statistics, Dates

const AC = AutonomousAI.AgentCore
const LLM = AutonomousAI.LLM
const SC = AutonomousAI.Schema
const MJ = AutonomousAI.MiniJSON

struct PlanEval
    name::String
    task::String
    ok_transport::Bool
    ok_schema::Bool
    latency_s::Float64
    quality::Float64
    note::String
end

function plan_quality(plan::SC.Plan)
    steps = plan.steps
    isempty(steps) && return 0.0

    has_bench = any(s -> s isa SC.BenchmarkAlgorithm, steps)
    has_verify = any(s -> s isa SC.VerifyResult, steps)
    has_exec = any(s -> s isa SC.ExecuteFinal, steps)

    idx_bench = findfirst(s -> s isa SC.BenchmarkAlgorithm, steps)
    idx_verify = findfirst(s -> s isa SC.VerifyResult, steps)
    idx_exec = findfirst(s -> s isa SC.ExecuteFinal, steps)

    ordered = idx_exec === nothing ? false :
              ((idx_bench !== nothing && idx_bench < idx_exec) &&
               (idx_verify !== nothing && idx_verify < idx_exec))

    score = 0.0
    has_bench && (score += 0.3)
    has_verify && (score += 0.3)
    has_exec && (score += 0.2)
    ordered && (score += 0.2)
    return score
end

function eval_one(name::String, backend::LLM.LLMBackend, task::String, n::Int)
    if backend isa LLM.AnthropicLLM
        isempty(get(ENV, backend.api_key_env, "")) &&
            return PlanEval(name, task, false, false, NaN, 0.0,
                            "missing $(backend.api_key_env)")
    elseif backend isa LLM.OpenAICompatibleLLM
        isempty(get(ENV, backend.api_key_env, "")) &&
            return PlanEval(name, task, false, false, NaN, 0.0,
                            "missing $(backend.api_key_env)")
    end

    goal = AC.Goal(task; input_size = [n], n_calls = 20)
    agent = AC.Agent(llm = LLM.MimaseLLM(), verbose = false)
    ctx = AC.build_context(agent, goal)
    user = MJ.to_json(ctx)

    r = LLM.complete(backend, LLM.SYSTEM_PROMPT, user)
    if !r.ok
        return PlanEval(name, task, false, false, r.latency_s, 0.0, r.error)
    end

    plan = try
        SC.parse_plan(LLM.extract_json(r.text), goal.id)
    catch err
        return PlanEval(name, task, true, false, r.latency_s, 0.0, string(err))
    end

    return PlanEval(name, task, true, true, r.latency_s, plan_quality(plan), "ok")
end

function aggregate(rows::Vector{PlanEval})
    n = length(rows)
    t_ok = count(r -> r.ok_transport, rows)
    s_ok = count(r -> r.ok_schema, rows)
    lats = [r.latency_s for r in rows if r.ok_transport]
    q = [r.quality for r in rows if r.ok_schema]
    return (
        samples = n,
        transport_ok_rate = n == 0 ? 0.0 : t_ok / n,
        schema_ok_rate = n == 0 ? 0.0 : s_ok / n,
        latency_median_s = isempty(lats) ? NaN : median(lats),
        latency_p95_s = isempty(lats) ? NaN : quantile(lats, 0.95),
        quality_mean = isempty(q) ? NaN : mean(q),
    )
end

function write_csv(path::String, rows::Vector{PlanEval})
    open(path, "w") do io
        println(io, "backend,task,transport_ok,schema_ok,latency_s,quality,note")
        for r in rows
            note = replace(r.note, '"' => "'")
            println(io, string(r.name, ",\"", replace(r.task, '"' => "'"), "\",",
                r.ok_transport, ",", r.ok_schema, ",",
                @sprintf("%.6f", r.latency_s), ",",
                @sprintf("%.3f", r.quality), ",\"", note, "\""))
        end
    end
end

function main()
    repeats = try parse(Int, get(ENV, "AAI_BENCH_REPEATS", "3")) catch; 3 end

    backends = [
        ("Mimase", LLM.MimaseLLM()),
        ("Anthropic Opus 5", LLM.AnthropicLLM(model = get(ENV, "AAI_OPUS_MODEL", "claude-opus-5"))),
        ("Gemini 3.7", LLM.OpenAICompatibleLLM(
            model = get(ENV, "AAI_GEMINI_MODEL", "gemini-3.7-pro"),
            url = get(ENV, "AAI_GEMINI_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
            api_key_env = "GEMINI_API_KEY")),
    ]

    tasks = [
        ("detect anomalies in a sensor stream", 2_000_000),
        ("multiply two large matrices safely", 4096 * 4096),
        ("optimize a stencil smoothing kernel", 3_000_000),
    ]

    rows = PlanEval[]
    for (name, backend) in backends
        for (task, n) in tasks
            for _ in 1:repeats
                push!(rows, eval_one(name, backend, task, n))
            end
        end
    end

    out_csv = joinpath(@__DIR__, "llm_model_comparison_results.csv")
    write_csv(out_csv, rows)

    println("LLM planning benchmark (higher is better for rates/quality, lower better for latency)")
    println(@sprintf("%-20s %7s %11s %11s %11s %11s %9s", "backend", "samples",
        "transport", "schema", "lat_med[s]", "lat_p95[s]", "quality"))

    for name in unique(r.name for r in rows)
        agg = aggregate(filter(r -> r.name == name, rows))
        println(@sprintf("%-20s %7d %10.1f%% %10.1f%% %11.3f %11.3f %9.3f",
            name, agg.samples, 100 * agg.transport_ok_rate, 100 * agg.schema_ok_rate,
            agg.latency_median_s, agg.latency_p95_s, agg.quality_mean))
    end

    println("\nDetailed rows written to: ", out_csv)
    println("Note: External providers require valid API keys and reachable endpoints.")
end

main()
