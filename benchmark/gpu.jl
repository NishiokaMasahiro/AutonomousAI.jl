# GPU benchmark suite (spec section 40): H2D transfer, kernel, D2H, end to end.
#
# Requires CUDA.jl and KernelAbstractions.jl.  Without them this script reports
# that GPU benchmarking is unavailable rather than emitting placeholder numbers.
using AutonomousAI, Printf
const C = AutonomousAI.Compute
const CG = AutonomousAI.CodeGeneration

if !C.CUDA_FUNCTIONAL[]
    println("CUDA is not functional in this session.")
    println("Install CUDA.jl + KernelAbstractions.jl and re-run; the candidate space")
    println("will then include :cuda automatically.")
else
    using CUDA
    for n in (10^6, 10^7, 10^8)
        x = randn(Float32, n)
        d = CUDA.zeros(Float32, n)
        t_h2d = CUDA.@elapsed copyto!(d, x)
        t_d2h = CUDA.@elapsed copyto!(x, d)
        @printf("n=%-10d H2D %8.4f s (%.1f GB/s)  D2H %8.4f s (%.1f GB/s)\n", n,
                t_h2d, 4n / t_h2d / 1e9, t_d2h, 4n / t_d2h / 1e9)
    end
    println("\nKernel and end-to-end timings are produced by benchmark/end_to_end.jl,")
    println("which routes through the same sandbox as the agent so the numbers are")
    println("comparable with the ones the planner sees.")
end
