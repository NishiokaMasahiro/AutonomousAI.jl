# Example 3 -- the section 39 out-of-core workflow.
#
# "Analyse this 100 GB scientific dataset as fast as possible."  The point of this
# example is that the interesting decision is not which kernel to use but how to
# stage data through a device whose memory is two orders of magnitude too small.
#
# Run:  julia --project=. examples/out_of_core.jl

using AutonomousAI, Printf
const O = AutonomousAI.Optimization
const HAL = AutonomousAI.HAL

profile = HAL.probe_hardware()
println(HAL.summarize(profile))

dataset_bytes = 100 * 1024^3
precision = :Float32
n_total = div(dataset_bytes, 4)

vram = isempty(profile.gpus) ? 0 : profile.gpus[1].vram_total_bytes
ram = profile.memory.available_bytes

if vram == 0
    @info "no GPU present: the same planner produces a RAM-staged plan instead"
    plan = O.chunk_plan(n_total, precision, min(ram, 4 * 1024^3); safety = 0.5,
                        buffers = 3, double_buffered = false)
else
    plan = O.chunk_plan(n_total, precision, vram; safety = 0.6, buffers = 3,
                        double_buffered = true)
end

println("\nchunking decision")
println("  ", plan.rationale)
@printf("  %d chunks of %d elements (%.2f GiB each)\n", plan.n_chunks,
        plan.chunk_elements, plan.bytes_per_chunk / 1024^3)

# What the chunk size actually trades off:
#   too small -> kernel launch and PCIe latency dominate (n_chunks * overhead)
#   too large -> allocator pressure, no room for double buffering, OOM risk
# The agent does not guess between them; it tunes on the real machine.
grid = [2^k for k in 20:26]
@printf("\ntuning grid: %s\n", join(grid, ", "))
println("  (in a real run, `TuneParameter` measures each point in the sandbox and")
println("   fits a GP over log2(chunk); here we only show the search space.)")

transfer_s = plan.bytes_per_chunk * plan.n_chunks / (25e9)
@printf("\nlower bound from PCIe alone: %.1f s for %d GiB at 25 GB/s\n",
        transfer_s, dataset_bytes / 1024^3)
println("If the kernel is faster than this, the run is transfer bound and the only")
println("useful optimisations are overlap (double buffering) and precision.")
