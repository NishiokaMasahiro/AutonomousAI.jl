"""
    AutonomousAICUDAExt

Authoritative NVIDIA probe and telemetry.  Loaded automatically when CUDA.jl is
present; without it the `:cuda` backend simply never appears in the candidate set.

Compute capability matters here beyond bookkeeping: an RTX 5070 is Blackwell
(sm_120) and needs CUDA >= 12.8 and a recent CUDA.jl.  Reporting the real capability
lets `Optimization` refuse FP64-heavy plans on a GeForce part, whose FP64 throughput
is 1/64 of FP32.
"""
module AutonomousAICUDAExt

using AutonomousAI
using CUDA

const HAL = AutonomousAI.HAL
const Compute = AutonomousAI.Compute

function cuda_probe()
    CUDA.functional() || return HAL.GPUInfo[]
    out = HAL.GPUInfo[]
    for (i, dev) in enumerate(CUDA.devices())
        CUDA.device!(dev)
        cap = CUDA.capability(dev)
        free, total = CUDA.available_memory(), CUDA.total_memory()
        name = CUDA.name(dev)
        sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        clock_khz = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
        bus_bits = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
        # effective bandwidth = 2 (DDR) * clock * bus / 8, in GB/s
        bw = 2.0 * (clock_khz * 1e3) * (bus_bits / 8) / 1e9
        push!(out, HAL.GPUInfo(i - 1, name, :nvidia,
                               VersionNumber(cap.major, cap.minor), Int(total),
                               Int(free), Int(sms), bw, CUDA.runtime_version(),
                               VersionNumber(CUDA.driver_version()),
                               cap >= v"7.0", cap >= v"5.3", cap >= v"8.0",
                               HAL.geforce_fp64_ratio(name)))
    end
    return out
end

function __init__()
    HAL.register_gpu_probe!(cuda_probe)
    Compute.CUDA_FUNCTIONAL[] = CUDA.functional()
    return nothing
end

end # module
