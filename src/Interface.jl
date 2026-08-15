"""
    Interface

CLI and human-readable reporting (spec section 41, 45).

The report is deliberately explicit about *evidence class*: every number is labelled
`measured`, `fitted` or `roofline`, because a plan built on an analytic prior and a
plan built on last week's benchmark are not equally trustworthy and the operator has
to be able to tell which one produced a given decision.
"""
module Interface

using Printf, Dates, Statistics
using ..Schema
using ..HAL
using ..WorldModel
import ..Memory
using ..Optimization
using ..Verification
using ..Execute
using ..AgentCore
using ..LLM

export report, print_report, comparison_table, main

function report(r::AgentCore.AgentReport)
    io = IOBuffer()
    g = r.goal
    println(io, "=" ^ 78)
    println(io, "GOAL   : ", g.description)
    println(io, @sprintf("TARGET : %s  size=%s  n_calls=%d  rtol=%.1e", g.algorithm,
                         string(g.input_size), g.n_calls, g.rtol))
    println(io, @sprintf("STATUS : %s   (%.2fs wall)", r.stop_reason, r.wall_s))
    println(io, "=" ^ 78)
    println(io, "\nDECISION LOG")
    for d in r.decisions
        println(io, AgentCore.format_decision(d))
        isempty(d.notes) || println(io, " " ^ 8, "notes: ", d.notes)
    end
    println(io, "\nMEASUREMENTS")
    if isempty(r.runs)
        println(io, "  (none)")
    else
        println(io, @sprintf("  %-40s %10s %10s %10s %8s", "candidate", "median[s]",
                             "min[s]", "cold[s]", "MiB"))
        for run in r.runs
            run.ok || continue
            println(io, @sprintf("  %-40s %10.6f %10.6f %10.3f %8.2f",
                                 string(run.candidate), run.stats.median_s,
                                 run.stats.min_s, run.stats.compile_s,
                                 run.stats.alloc_bytes / 1024^2))
        end
    end
    if r.baseline !== nothing && r.best !== nothing
        println(io, "\nSPEEDUP")
        println(io, @sprintf("  baseline : %-38s %.6fs", string(r.baseline.candidate),
                             r.baseline.stats.median_s))
        println(io, @sprintf("  best     : %-38s %.6fs", string(r.best.candidate),
                             r.best.stats.median_s))
        if isnan(r.speedup_ci[1])
            println(io, @sprintf("  ratio    : %.3fx (no CI: identical candidate)", r.speedup))
        else
            sig = r.speedup_ci[1] > 1.02
            println(io, @sprintf("  ratio    : %.3fx  95%% CI [%.3f, %.3f]  -> %s",
                                 r.speedup, r.speedup_ci[1], r.speedup_ci[2],
                                 sig ? "significant" : "NOT distinguishable from noise"))
        end
    end
    println(io, "\nVERIFICATION")
    if r.verification === nothing
        println(io, "  not performed")
    else
        print(io, "  ")
        show(io, r.verification)
    end
    println(io, "\nPLANNER LOG")
    for l in r.plan_log
        println(io, "  ", l)
    end
    return String(take!(io))
end

print_report(r::AgentCore.AgentReport) = println(report(r))

"""
    comparison_table(runs, baseline) -> String

The section 46 self-improvement table: one row per optimisation level with runtime,
speedup, memory and accuracy.
"""
function comparison_table(labelled::Vector{<:Tuple}, baseline_s::Float64)
    io = IOBuffer()
    println(io, @sprintf("%-28s %-34s %11s %9s %10s %12s", "level", "candidate",
                         "median[s]", "speedup", "alloc[MiB]", "rel.error"))
    println(io, "-" ^ 108)
    for (label, run, err) in labelled
        run === nothing && continue
        sp = baseline_s / max(run.stats.median_s, 1e-12)
        println(io, @sprintf("%-28s %-34s %11.6f %8.2fx %10.2f %12s", label,
                             string(run.candidate), run.stats.median_s, sp,
                             run.stats.alloc_bytes / 1024^2,
                             isnan(err) ? "n/a" : @sprintf("%.2e", err)))
    end
    return String(take!(io))
end

const USAGE = """
AutonomousAI.jl -- bounded-autonomy computational optimisation agent

usage:
  julia --project -e 'using AutonomousAI; AutonomousAI.Interface.main(ARGS)' -- [cmd]

commands:
  hardware              print the discovered hardware profile and capability report
  demo <goal text>      run the full closed loop on a goal (offline mock planner)
  selftest              validate templates and the code validator without a sandbox
  memory                summarise the persisted benchmark memory
"""

function main(args::Vector{String} = String[])
    cmd = isempty(args) ? "help" : args[1]
    if cmd == "hardware"
        w = WorldModel.WorldState()
        println(WorldModel.describe(w))
        ok, rep = HAL.check_requirements(w.hardware.profile)
        println("\nrequirements: ", ok ? "met" : "NOT met")
        for l in rep
            println("  - ", l)
        end
    elseif cmd == "memory"
        ms = Memory.load_memory()
        println("records: ", Memory.n_records(ms.hardware))
        println("episodes: ", length(ms.episodic.episodes))
        for e in Iterators.take(Iterators.reverse(ms.episodic.episodes), 10)
            println(@sprintf("  %s  %-40s speedup %.2fx  %s", e.started, e.goal,
                             e.speedup, e.success ? "ok" : e.failure_reason))
        end
    elseif cmd == "demo"
        text = length(args) > 1 ? join(args[2:end], " ") :
               "detect anomalies in a large sensor array"
        agent = AgentCore.Agent()
        goal = AgentCore.Goal(text; input_size = Int[1_000_000], n_calls = 100)
        r = AgentCore.run_goal!(agent, goal)
        print_report(r)
        Memory.save_memory(agent.memory)
    else
        println(USAGE)
    end
    return nothing
end

end # module
