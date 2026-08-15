# CPU benchmark suite (spec section 40): single thread, SIMD, threads, precision.
using AutonomousAI, Printf
const S = AutonomousAI.Schema
const E = AutonomousAI.Execute
const SB = AutonomousAI.Sandbox

spec = SB.SandboxSpec(AutonomousAI.Safety.default_limits())
sizes = [10^4, 10^5, 10^6, 10^7]

println(@sprintf("%-16s %10s %-14s %12s %12s %10s", "algorithm", "n", "backend",
                 "median[s]", "cold[s]", "GB/s"))
for alg in (:stencil3, :sum_reduction, :zscore_anomaly)
    for n in sizes, backend in (:cpu_serial, :cpu_simd), prec in (:Float64, :Float32)
        c = S.Candidate(alg, backend, prec, [:inbounds, :simd], Dict{String,Int}())
        run = E.run_candidate(c, [n], spec; samples = 7)
        run.ok || continue
        gbs = AutonomousAI.Compute.bytes_for(alg, [n], prec) / run.stats.median_s / 1e9
        println(@sprintf("%-16s %10d %-14s %12.6f %12.3f %10.2f", alg, n,
                         string(backend, "/", prec), run.stats.median_s,
                         run.stats.compile_s, gbs))
    end
end
