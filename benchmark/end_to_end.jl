# End-to-end agent benchmark: how much wall time the *agent* spends, split into
# planning, compilation, benchmarking and useful execution.  Optimisation overhead
# is a first-class metric (spec section 41): an agent that spends 40 s to save 2 s
# has lost.
using AutonomousAI, Printf
const AC = AutonomousAI.AgentCore

for n in (10^5, 10^6, 10^7)
    agent = AC.Agent(verbose = false)
    goal = AC.Goal("detect anomalies"; input_size = [n], n_calls = 100)
    t0 = time()
    r = AC.run_goal!(agent, goal)
    wall = time() - t0
    bench_s = agent.budget.benchmark_s
    compile_s = agent.budget.compile_s
    best = r.best === nothing ? NaN : r.best.stats.median_s
    base = r.baseline === nothing ? NaN : r.baseline.stats.median_s
    saved = isnan(base) || isnan(best) ? NaN : (base - best) * goal.n_calls
    @printf("n=%-9d wall %6.2fs (bench %5.2fs, compile %5.2fs)  speedup %5.2fx  " *
            "saved-over-%d-calls %6.3fs  payback %s\n", n, wall, bench_s, compile_s,
            r.speedup, goal.n_calls, saved,
            isnan(saved) ? "n/a" : (saved > wall ? "YES" : "no"))
end
