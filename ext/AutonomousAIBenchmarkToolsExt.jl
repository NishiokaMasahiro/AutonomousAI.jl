"""
    AutonomousAIBenchmarkToolsExt

Replaces the built-in timing loop with BenchmarkTools.jl when it is available
(spec section 15).  BenchmarkTools gives adaptive sample counts, interpolation of
arguments and better outlier handling; the built-in loop exists so the core package
has no dependency, not because it is better.
"""
module AutonomousAIBenchmarkToolsExt

using AutonomousAI
using BenchmarkTools
using Statistics

const Opt = AutonomousAI.Optimization

function bt_measure(f::Function, args...; samples::Int = 9)
    compile_s = @elapsed f(args...)
    b = @benchmark $f($(args)...) samples = samples evals = 1 seconds = 30
    ts = b.times ./ 1e9
    med = median(ts)
    return Opt.BenchmarkStats(compile_s, minimum(ts), med, mean(ts),
                              median(abs.(ts .- med)), ts, Int(b.memory),
                              Int(b.allocs), b.gctimes === nothing ? 0.0 :
                              sum(b.gctimes) / 1e9)
end

__init__() = (Opt.MEASURE_HOOK[] = bt_measure; nothing)

end # module
