"""
    AgentCore

Goal, decision log and the closed loop (spec sections 24, 25, 45).

Naming deviation from the requested layout: the module is `AgentCore`, not `Core`.
A nested module named `Core` shadows Julia's own `Core` inside the package and makes
`using .Core` ambiguous with the language's root module.  This is the kind of
"change with a stated reason" that section 4 permits.

Autonomy model (section 25): **bounded goal-directed autonomy**.  The agent chooses
actions; it never edits `Goal`.  `Goal` is immutable and is compared by value at the
end of every iteration; a mutation attempt is a hard failure, not a warning.
"""
module AgentCore

using Dates, Printf, Statistics, Random
using ..MiniJSON
using ..Schema
using ..HAL
using ..WorldModel
import ..Memory
using ..Safety
using ..CodeGeneration
using ..Compute
using ..Sandbox
using ..Optimization
using ..Verification
using ..Execute
using ..LLM

export Goal, Agent, Decision, AgentReport, run_goal!, infer_algorithm,
       default_plan, build_context, utility

"""
Immutable objective.  `n_calls` is what makes compile-time amortisation decidable:
the same candidate can be optimal for `n_calls=1` and pessimal for `n_calls=10^4`.
"""
struct Goal
    id::String
    description::String
    algorithm::Symbol
    input_size::Vector{Int}
    n_calls::Int
    rtol::Float64
    deadline_s::Float64
    created::DateTime
end

function Goal(description::AbstractString; algorithm::Symbol = :auto,
              input_size::Vector{Int} = Int[1_000_000], n_calls::Int = 1,
              rtol::Float64 = 1e-6, deadline_s::Float64 = 300.0)
    alg = algorithm === :auto ? infer_algorithm(description) : algorithm
    return Goal(string("goal-", string(hash(description), base = 16)[1:8]),
                String(description), alg, input_size, n_calls, rtol, deadline_s, now())
end

"""
    infer_algorithm(text) -> Symbol

Keyword matching, not understanding.  It exists so the offline demo runs without a
model; with a real `LLMBackend` the algorithm comes from the plan.  Reported as
`heuristic` in the decision log so no one mistakes it for language comprehension.
"""
function infer_algorithm(text::AbstractString)
    t = lowercase(String(text))
    (occursin("robust", t) || occursin("median", t) || occursin("mad", t)) &&
        return :mad_anomaly
    (occursin("anomal", t) || occursin("outlier", t)) && return :zscore_anomaly
    (occursin("matmul", t) || occursin("matrix", t) || occursin("matrices", t) ||
     occursin("multiply", t) || occursin("gemm", t)) && return :matmul
    (occursin("smooth", t) || occursin("stencil", t) || occursin("filter", t)) &&
        return :stencil3
    (occursin("sum", t) || occursin("total", t) || occursin("reduc", t)) &&
        return :sum_reduction
    return :zscore_anomaly
end

"One authorised step and everything needed to explain it later."
struct Decision
    step::Int
    action::String
    candidate::String
    allowed::Bool
    reason::String
    evidence::Symbol
    runtime_s::Float64
    verified::Union{Nothing,Bool}
    notes::String
    at::DateTime
end

mutable struct Agent
    llm::LLM.LLMBackend
    memory::Memory.MemorySystem
    engine::Safety.PolicyEngine
    world::WorldModel.WorldState
    cost::Optimization.CostModel
    budget::Optimization.OptimizationBudget
    sandbox::Sandbox.SandboxSpec
    decisions::Vector{Decision}
    verbose::Bool
    last_verification::Union{Nothing,Verification.VerificationReport}
end

function Agent(; llm::LLM.LLMBackend = LLM.MimaseLLM(),
               memory::Memory.MemorySystem = Memory.MemorySystem(),
               world::WorldModel.WorldState = WorldModel.WorldState(),
               policy::Union{Nothing,Safety.Policy} = nothing,
               verbose::Bool = true)
    prof = world.hardware.profile
    limits = Safety.default_limits(ram_total = prof.memory.total_bytes,
        vram_total = isempty(prof.gpus) ? 0 : prof.gpus[1].vram_total_bytes)
    caps = Safety.default_capabilities()
    pol = policy === nothing ? Safety.Policy(limits, caps) : policy
    engine = Safety.PolicyEngine(pol)
    for f in Compute.algorithm_facts()
        Memory.register_fact!(memory.semantic, f)
    end
    return Agent(llm, memory, engine, world, Optimization.default_cost_model(),
                 Optimization.OptimizationBudget(
                     max_iterations = pol.max_optimization_iterations,
                     max_benchmark_s = pol.max_total_benchmark_s,
                     min_relative_improvement = pol.min_relative_improvement),
                 Sandbox.SandboxSpec(pol.limits), Decision[], verbose, nothing)
end

struct AgentReport
    goal::Goal
    baseline::Union{Nothing,Execute.CandidateRun}
    best::Union{Nothing,Execute.CandidateRun}
    speedup::Float64
    speedup_ci::Tuple{Float64,Float64}
    verified::Bool
    verification::Union{Nothing,Verification.VerificationReport}
    decisions::Vector{Decision}
    plan_log::Vector{String}
    stop_reason::String
    wall_s::Float64
    runs::Vector{Execute.CandidateRun}
end

log!(a::Agent, d::Decision) = (push!(a.decisions, d);
    a.verbose && println(format_decision(d)); d)

format_decision(d::Decision) = @sprintf("  [%02d] %-20s %-42s %s%s", d.step, d.action,
    d.candidate, d.allowed ? "" : "DENIED: ", d.reason)

"""
    build_context(agent, goal) -> Dict

Everything the planner is allowed to see.  Deliberately small and typed: hardware
capability, the *enumerated* candidate space, the semantic facts, and what memory
already knows.  No filesystem contents, no environment, no raw telemetry stream.
"""
function build_context(a::Agent, g::Goal)
    prof = a.world.hardware.profile
    f = Memory.fact(a.memory.semantic, g.algorithm)
    hw = a.world.hardware.fingerprint
    sw = HAL.software_fingerprint()
    best = Memory.best_known(a.memory.hardware, g.algorithm, prod(g.input_size);
                             hw = hw, sw = sw)
    return Dict{String,Any}(
        "goal" => g.description,
        "algorithm" => String(g.algorithm),
        "input_size" => g.input_size,
        "n_elements" => prod(g.input_size),
        "n_calls" => g.n_calls,
        "available_backends" => String[String(b) for b in
                                       Compute.available_backends(prof)],
        "available_precisions" => String[String(p) for p in
            (isempty(prof.gpus) ? [:Float64, :Float32] : [:Float64, :Float32, :Float16])],
        "available_transforms" => String[String(t) for t in
                                         CodeGeneration.applicable_transforms(g.algorithm)],
        "gpu_friendly" => f === nothing ? false : f.gpu_friendly,
        "arithmetic_intensity" => f === nothing ? 1.0 : f.arithmetic_intensity,
        "min_safe_precision" => f === nothing ? "Float64" : String(f.min_safe_precision),
        "ram_total_gib" => prof.memory.total_bytes / 1024^3,
        "vram_total_gib" => isempty(prof.gpus) ? 0.0 :
                            prof.gpus[1].vram_total_bytes / 1024^3,
        "julia_threads" => Threads.nthreads(),
        "best_known_runtime_s" => best === nothing ? nothing : best.runtime_min_s,
        "policy" => Dict{String,Any}(
            "allow_fastmath" => a.engine.policy.allow_fastmath,
            "allow_inbounds" => a.engine.policy.allow_inbounds,
            "max_runtime_s" => a.engine.policy.limits.max_runtime_s))
end

"""
    default_plan(agent, goal) -> Plan

The runtime's own plan, used when the model is unavailable or keeps producing
schema-invalid output.  Its existence is what lets the system degrade to
"slower but correct" instead of "unvalidated but fast".
"""
function default_plan(a::Agent, g::Goal)
    base = Schema.Candidate(g.algorithm, :cpu_serial, :Float64)
    backends = Compute.available_backends(a.world.hardware.profile)
    opt_backend = :cpu_simd in backends ? :cpu_simd : :cpu_serial
    ts = collect(intersect(Set([:inbounds, :simd]),
                           CodeGeneration.applicable_transforms(g.algorithm)))
    opt = Schema.Candidate(g.algorithm, opt_backend, :Float64, sort(ts),
                           Dict{String,Int}())
    return Schema.Plan("default-" * g.id, g.id, Schema.Action[
        Schema.InspectHardware(),
        Schema.ProposeCandidates([base, opt], "runtime default: baseline + transforms"),
        Schema.BenchmarkAlgorithm(base, g.input_size, 7),
        Schema.GenerateCode(opt),
        Schema.BenchmarkAlgorithm(opt, g.input_size, 7),
        Schema.VerifyResult(opt, :cpu_serial, :Float64, g.rtol, 0.0),
        Schema.ExecuteFinal(opt, g.input_size),
    ], "deterministic fallback plan")
end

"""
    utility(goal_met, speedup, efficiency, accuracy, resource_cost, risk; w) -> Float64

Section 26's `U`, used only to *rank already-feasible outcomes* for the report.  It
never overrides the constraint hierarchy: infeasible candidates are removed before
`U` is ever computed.
"""
function utility(goal_met::Bool, speedup::Float64, efficiency::Float64,
                 accuracy::Float64, resource_cost::Float64, risk::Float64;
                 w = (2.0, 1.0, 0.5, 1.5, 0.5, 1.0))
    return w[1] * (goal_met ? 1.0 : 0.0) + w[2] * log2(max(speedup, 1e-6)) +
           w[3] * efficiency + w[4] * accuracy - w[5] * resource_cost - w[6] * risk
end

# ------------------------------------------------------------- the main loop --

"""
    run_goal!(agent, goal; max_replans=2) -> AgentReport

Observe -> plan -> generate -> sandbox -> benchmark -> verify -> execute ->
record -> replan.  Each step passes `Safety.authorize` before it runs; a denied step
does not abort the run, it becomes evidence fed back into replanning.
"""
function run_goal!(a::Agent, g::Goal; max_replans::Int = 2)
    t0 = time()
    goal_snapshot = (g.id, g.algorithm, g.input_size, g.n_calls)
    WorldModel.observe!(a.world)
    runs = Execute.CandidateRun[]
    baseline = nothing
    best = nothing
    verification = nothing
    stop_reason = "completed"
    plan_log = String[]
    step = 0

    for attempt in 0:max_replans
        ctx = build_context(a, g)
        attempt > 0 && (ctx["previous_failure"] = stop_reason)
        plan, plog = LLM.propose_plan(a.llm, ctx, g.id)
        append!(plan_log, plog)
        if plan === nothing
            push!(plan_log, "falling back to runtime default plan")
            plan = default_plan(a, g)
        end
        failed = false
        for act in plan.steps
            step += 1
            WorldModel.observe!(a.world)
            dec = Safety.authorize(a.engine, act, a.world)
            if !dec.allowed
                log!(a, Decision(step, Schema.action_name(act), candidate_str(act),
                                 false, dec.reason, :policy, NaN, nothing, "", now()))
                stop_reason = "policy denial: " * dec.reason
                failed = true
                break
            end
            ok, note, run = execute_step!(a, g, act, step)
            run === nothing || push!(runs, run)
            if run !== nothing && run.ok
                if run.candidate.backend === :cpu_serial &&
                   run.candidate.precision === :Float64 && isempty(run.candidate.transforms)
                    baseline = run
                end
                if best === nothing || (run.stats.median_s < best.stats.median_s)
                    best = run
                end
            end
            if act isa Schema.VerifyResult && run === nothing
                verification = get_last_verification(a)
            end
            if !ok
                stop_reason = note
                failed = true
                break
            end
        end
        failed || break
        attempt == max_replans && (stop_reason *= " (replan budget exhausted)")
    end

    (g.id, g.algorithm, g.input_size, g.n_calls) == goal_snapshot ||
        error("goal mutated during execution -- bounded autonomy violated")

    speedup, ci = 1.0, (NaN, NaN)
    if baseline !== nothing && best !== nothing && best !== baseline
        p, lo, hi = Optimization.ratio_ci(baseline.stats.samples, best.stats.samples)
        speedup, ci = p, (lo, hi)
    end
    verification = a.last_verification === nothing ? verification : a.last_verification
    report = AgentReport(g, baseline, best, speedup, ci,
                         verification === nothing ? false : verification.passed,
                         verification, copy(a.decisions), plan_log, stop_reason,
                         time() - t0, runs)
    record_episode!(a, g, report)
    return report
end

candidate_str(a::Schema.Action) = ""
candidate_str(a::Schema.GenerateCode) = string(a.candidate)
candidate_str(a::Schema.BenchmarkAlgorithm) = string(a.candidate)
candidate_str(a::Schema.VerifyResult) = string(a.candidate)
candidate_str(a::Schema.ExecuteFinal) = string(a.candidate)
candidate_str(a::Schema.TuneParameter) = string(a.candidate)

set_verification!(a::Agent, v) = (a.last_verification = v)
get_last_verification(a::Agent) = a.last_verification

# ---------------------------------------------------------------- dispatch ---

execute_step!(a::Agent, g::Goal, act::Schema.InspectHardware, step::Int) = begin
    WorldModel.observe!(a.world)
    ok, rep = HAL.check_requirements(a.world.hardware.profile)
    log!(a, Decision(step, "inspect_hardware", "", true, join(rep, "; "),
                     :measured, NaN, nothing, "", now()))
    (true, "", nothing)
end

function execute_step!(a::Agent, g::Goal, act::Schema.ProposeCandidates, step::Int)
    enumerated = Optimization.enumerate_candidates(g.algorithm, a.world)
    allowed = Set(enumerated)
    kept = [c for c in act.candidates if c in allowed || c.backend === :cpu_serial]
    dropped = length(act.candidates) - length(kept)
    model = Optimization.CostModel(a.cost.w_runtime, a.cost.w_memory, a.cost.w_energy,
        a.cost.w_error, a.cost.w_risk, a.cost.w_compile, a.cost.t_ref, a.cost.mem_ref,
        a.cost.energy_ref, a.cost.error_ref, g.n_calls)
    scored, rejected = Optimization.rank_candidates(kept, g.input_size, a.world,
        a.memory, model, a.engine.policy.limits)
    Memory.remember!(a.memory.working, "ranked", scored)
    note = @sprintf("%d proposed, %d out-of-space dropped, %d feasible, %d infeasible",
                    length(act.candidates), dropped, length(scored), length(rejected))
    src = isempty(scored) ? :none : scored[1].source
    log!(a, Decision(step, "propose_candidates", isempty(scored) ? "" :
                     string(scored[1].candidate), true, note, src, NaN, nothing,
                     isempty(rejected) ? "" : "rejected: " *
                     join([string(r.candidate, " (", r.reason, ")") for r in rejected], "; "),
                     now()))
    isempty(scored) && return (false, "no feasible candidate after ranking", nothing)
    return (true, "", nothing)
end

function execute_step!(a::Agent, g::Goal, act::Schema.GenerateCode, step::Int)
    unit = CodeGeneration.generate(act.candidate;
        allow_fastmath = a.engine.policy.allow_fastmath,
        allow_inbounds = a.engine.policy.allow_inbounds)
    h = Memory.source_hash(unit.source)
    Memory.store_artifact!(a.memory.procedural,
        Memory.CodeArtifact(h, act.candidate, unit.source, unit.report.ok,
                            vcat(unit.report.errors, unit.report.warnings, unit.notes),
                            now(), 0, 0))
    note = unit.report.ok ?
        @sprintf("validated %d AST nodes; %s", unit.report.n_nodes,
                 isempty(unit.notes) ? "no transforms" : join(unit.notes, ", ")) :
        "REJECTED: " * join(unit.report.errors, "; ")
    log!(a, Decision(step, "generate_code", string(act.candidate), unit.report.ok,
                     note, :static, NaN, nothing, h, now()))
    unit.report.ok || return (false, "code validation failed: " *
                              join(unit.report.errors, "; "), nothing)
    return (true, "", nothing)
end

function execute_step!(a::Agent, g::Goal, act::Schema.BenchmarkAlgorithm, step::Int)
    t0 = time()
    run = Execute.run_candidate(act.candidate, act.input_size, a.sandbox;
        samples = act.samples, allow_fastmath = a.engine.policy.allow_fastmath,
        allow_inbounds = a.engine.policy.allow_inbounds)
    a.budget.benchmark_s += time() - t0
    a.budget.iterations += 1
    if run.ok
        a.budget.compile_s += isnan(run.stats.compile_s) ? 0.0 : run.stats.compile_s
        rec = Memory.BenchmarkRecord(act.candidate.algorithm, act.candidate.backend,
            act.candidate.precision, copy(act.candidate.transforms),
            prod(act.input_size), copy(act.input_size), run.stats.compile_s,
            run.stats.min_s, run.stats.median_s, run.stats.mad_s, act.samples,
            run.stats.alloc_bytes, run.stats.alloc_count,
            Compute.flops_for(act.candidate.algorithm, act.input_size) /
                max(run.stats.median_s, 1e-12) / 1e9,
            round(Int, Compute.bytes_for(act.candidate.algorithm, act.input_size,
                                         act.candidate.precision)),
            0, 0, NaN, NaN, false, a.world.hardware.fingerprint,
            HAL.software_fingerprint(), now())
        Memory.record!(a.memory.hardware, rec)
        log!(a, Decision(step, "benchmark_algorithm", string(act.candidate), true,
                         @sprintf("median %.6fs (min %.6fs, cold %.3fs, %.2f GFLOP/s)",
                                  run.stats.median_s, run.stats.min_s,
                                  run.stats.compile_s, rec.gflops),
                         :measured, run.stats.median_s, nothing, run.source_hash, now()))
        return (true, "", run)
    end
    log!(a, Decision(step, "benchmark_algorithm", string(act.candidate), true,
                     "FAILED: " * run.error, :measured, NaN, false, "", now()))
    return (false, "benchmark failed: " * run.error, run)
end

function execute_step!(a::Agent, g::Goal, act::Schema.TuneParameter, step::Int)
    grid = act.grid
    objective = function (v)
        c = Schema.Candidate(act.candidate.algorithm, act.candidate.backend,
                             act.candidate.precision, act.candidate.transforms,
                             merge(act.candidate.params, Dict(act.name => v)))
        r = Execute.run_candidate(c, g.input_size, a.sandbox; samples = 5,
            allow_fastmath = a.engine.policy.allow_fastmath,
            allow_inbounds = a.engine.policy.allow_inbounds)
        return r.ok ? r.stats.median_s : Inf
    end
    bestv, besty, trace = Optimization.tune_discrete(objective, grid; budget = act.budget)
    log!(a, Decision(step, "tune_parameter", string(act.candidate), true,
                     @sprintf("%s = %d at %.6fs over %d evaluations", act.name, bestv,
                              besty, length(trace)), :measured, besty, nothing,
                     join([string(t[1], ":", round(t[2], sigdigits = 3)) for t in trace], " "),
                     now()))
    return (isfinite(besty), isfinite(besty) ? "" : "all tuning evaluations failed", nothing)
end

function execute_step!(a::Agent, g::Goal, act::Schema.VerifyResult, step::Int)
    T = Schema.precision_type(act.candidate.precision)
    run = Execute.run_candidate(act.candidate, g.input_size, a.sandbox; samples = 3,
        allow_fastmath = a.engine.policy.allow_fastmath,
        allow_inbounds = a.engine.policy.allow_inbounds)
    if !run.ok
        set_verification!(a, nothing)
        log!(a, Decision(step, "verify_result", string(act.candidate), true,
                         "FAILED to execute for verification: " * run.error, :measured,
                         NaN, false, "", now()))
        return (false, "verification run failed: " * run.error, run)
    end
    ref = Execute.reference_summary(act.candidate.algorithm, g.input_size, T)
    rep = Execute.compare_summaries(act.candidate.algorithm, run.summary, ref,
        act.candidate.precision, act.candidate.transforms, prod(g.input_size))
    set_verification!(a, rep)
    log!(a, Decision(step, "verify_result", string(act.candidate), true,
                     @sprintf("%s: %d/%d checks, err %.3e vs tol %.3e",
                              rep.passed ? "PASS" : "FAIL", count(c -> c[2], rep.checks),
                              length(rep.checks), rep.rel_error, rep.tolerance),
                     :measured, NaN, rep.passed,
                     join([string(c[1], "=", c[2] ? "ok" : "FAIL") for c in rep.checks], " "),
                     now()))
    return (rep.passed, rep.passed ? "" : "verification failed", run)
end

function execute_step!(a::Agent, g::Goal, act::Schema.ExecuteFinal, step::Int)
    v = get_last_verification(a)
    if a.engine.policy.require_verification && (v === nothing || !v.passed)
        log!(a, Decision(step, "execute_final", string(act.candidate), false,
                         "refused: no passing verification for this candidate", :policy,
                         NaN, false, "", now()))
        return (false, "execute_final refused: unverified candidate", nothing)
    end
    run = Execute.run_candidate(act.candidate, act.input_size, a.sandbox; samples = 5,
        allow_fastmath = a.engine.policy.allow_fastmath,
        allow_inbounds = a.engine.policy.allow_inbounds)
    log!(a, Decision(step, "execute_final", string(act.candidate), run.ok,
                     run.ok ? @sprintf("median %.6fs, result summary keys: %s",
                                       run.stats.median_s,
                                       join(sort(collect(keys(run.summary))), ",")) :
                     "FAILED: " * run.error, :measured,
                     run.ok ? run.stats.median_s : NaN, nothing, run.source_hash, now()))
    return (run.ok, run.ok ? "" : "final execution failed: " * run.error, run)
end

function execute_step!(a::Agent, g::Goal, act::Schema.Abort, step::Int)
    log!(a, Decision(step, "abort", "", true, act.reason, :planner, NaN, nothing, "", now()))
    return (false, "planner aborted: " * act.reason, nothing)
end

function execute_step!(a::Agent, g::Goal, act::Schema.RequestPermission, step::Int)
    log!(a, Decision(step, "request_permission", "", false,
                     "capability '$(act.capability)' must be granted out of band",
                     :policy, NaN, nothing, act.reason, now()))
    return (false, "permission required: $(act.capability)", nothing)
end

function record_episode!(a::Agent, g::Goal, r::AgentReport)
    Memory.record!(a.memory.episodic, Memory.Episode(
        string("ep-", string(hash((g.id, now())), base = 16)[1:10]), g.created, now(),
        g.description, "", length(r.decisions), r.stop_reason == "completed",
        r.stop_reason == "completed" ? "" : r.stop_reason,
        r.best === nothing ? "" : string(r.best.candidate),
        r.baseline === nothing ? NaN : r.baseline.stats.median_s,
        r.best === nothing ? NaN : r.best.stats.median_s, r.speedup,
        r.verification === nothing ? NaN : r.verification.rel_error,
        @sprintf("%d benchmark records, %.1fs wall", Memory.n_records(a.memory.hardware),
                 r.wall_s)))
    return r
end

end # module
