"""
    AutonomousAI

A Julia-native, bounded-autonomy computational intelligence system: an LLM plans over
a closed action space, the Julia runtime generates / measures / verifies / executes,
and every decision is recorded so the next one is better informed.

Layering (each module may only use the ones above it):

    MiniJSON -> Schema -> HAL -> WorldModel -> Memory -> Safety -> CodeGeneration
             -> Compute -> Sandbox -> Verification -> Optimization -> Execute
             -> LLM -> AgentCore -> Interface

Quick start:

    using AutonomousAI
    AutonomousAI.Interface.main(["hardware"])

    agent = AutonomousAI.AgentCore.Agent()
    goal  = AutonomousAI.AgentCore.Goal("find outliers in a 10M sample sensor stream";
                                        input_size = [10_000_000], n_calls = 50)
    report = AutonomousAI.AgentCore.run_goal!(agent, goal)
    AutonomousAI.Interface.print_report(report)

The package has **no non-stdlib dependencies**.  CUDA.jl, KernelAbstractions.jl and
BenchmarkTools.jl are optional weak dependencies loaded through package extensions;
their absence removes capabilities from the candidate space instead of breaking the
system.
"""
module AutonomousAI

include("MiniJSON.jl")
include("Schema.jl")
include("HAL.jl")
include("WorldModel.jl")
include("Memory.jl")
include("Safety.jl")
include("CodeGeneration.jl")
include("Compute.jl")
include("Sandbox.jl")
include("Verification.jl")
include("Optimization.jl")
include("Execute.jl")
include("LLM.jl")
include("AgentCore.jl")
include("Interface.jl")

using .MiniJSON, .Schema, .HAL, .WorldModel, .Memory, .Safety, .CodeGeneration
using .Compute, .Sandbox, .Verification, .Optimization, .Execute, .LLM
using .AgentCore, .Interface

export Goal, Agent, run_goal!, print_report, Candidate

const Goal = AgentCore.Goal
const Agent = AgentCore.Agent
const run_goal! = AgentCore.run_goal!
const print_report = Interface.print_report
const Candidate = Schema.Candidate

"Version of the architecture, not of the package: bumped when the layering changes."
const ARCHITECTURE_VERSION = v"0.1.0"

function __init__()
    # Late binding so `Optimization` does not depend on `CodeGeneration` at load time.
    Optimization.APPLICABLE_TRANSFORMS[] = CodeGeneration.applicable_transforms
    return nothing
end

end # module
