using Test
using AutonomousAI
const C = AutonomousAI.Compute
const HAL = AutonomousAI.HAL

@testset "GPU" begin
    prof = HAL.probe_hardware()
    if !C.CUDA_FUNCTIONAL[] || isempty(prof.gpus)
        @info "no functional CUDA device; GPU tests skipped (this is a supported state)"
        @test !(:cuda in C.available_backends(prof))     # capability, not configuration
    else
        g = prof.gpus[1]
        @test g.vram_total_bytes > 0
        @test g.compute_capability >= v"5.0"
        @test :cuda in C.available_backends(prof)
        @test 0 < g.fp64_ratio <= 1
        # a GeForce part must not be chosen as the FP64 oracle
        if g.fp64_ratio <= 1 / 32
            @test HAL.peak_flops(g, :Float64) < HAL.peak_flops(g, :Float32)
        end
    end
end
