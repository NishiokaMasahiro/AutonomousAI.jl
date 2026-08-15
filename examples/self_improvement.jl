# Example 2 -- the section 46 self-improvement demonstration.
#
# The same task is run at five optimisation levels and compared on runtime,
# speedup, allocation and accuracy.  Every level is measured in a separate OS
# process and verified against the FP64 reference before it is allowed to count.
#
# Run:  julia --project=. examples/self_improvement.jl

using AutonomousAI, Printf
const S  = AutonomousAI.Schema
const E  = AutonomousAI.Execute
const O  = AutonomousAI.Optimization
const SB = AutonomousAI.Sandbox

alg  = :stencil3
size = [4_000_000]
spec = SB.SandboxSpec(AutonomousAI.Safety.default_limits())

levels = [
    ("1 baseline",          S.Candidate(alg, :cpu_serial,  :Float64)),
    ("2 code transforms",   S.Candidate(alg, :cpu_serial,  :Float64, [:inbounds],        Dict{String,Int}())),
    ("3 vectorised",        S.Candidate(alg, :cpu_simd,    :Float64, [:inbounds, :simd], Dict{String,Int}())),
    ("4 threaded",          S.Candidate(alg, :cpu_threads, :Float64, [:inbounds, :simd], Dict{String,Int}())),
    ("5 reduced precision", S.Candidate(alg, :cpu_simd,    :Float32, [:inbounds, :simd], Dict{String,Int}())),
]

results    = Tuple{String,Union{Nothing,E.CandidateRun},Float64}[]
baseline_s = NaN

for (label, c) in levels
    run = E.run_candidate(c, size, spec; samples = 7)
    if !run.ok
        @warn "level failed" label error = run.error
        push!(results, (label, nothing, NaN))
        continue
    end
    ref = E.reference_summary(alg, size, S.precision_type(c.precision))
    rep = E.compare_summaries(alg, run.summary, ref, c.precision, c.transforms, prod(size))
    global baseline_s
    isnan(baseline_s) && (baseline_s = run.stats.median_s)
    push!(results, (label, rep.passed ? run : nothing, rep.rel_error))
    rep.passed || @warn "level rejected by verification" label report = rep
end

println(AutonomousAI.Interface.comparison_table(results, baseline_s))

# Statistical honesty: report whether each improvement survives a bootstrap CI.
base = results[1][2]
if base !== nothing
    for (label, run, _) in results[2:end]
        run === nothing && continue
        sig, point, lo, hi = O.significantly_faster(base.stats, run.stats)
        @printf("%-24s %.3fx  95%% CI [%.3f, %.3f]  %s\n", label, point, lo, hi,
                sig ? "significant" : "not distinguishable from noise")
    end
end
