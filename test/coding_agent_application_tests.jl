using Test
using AutonomousAI

const AC = AutonomousAI.AgentCore
const CG = AutonomousAI.CodeGeneration
const O = AutonomousAI.Optimization
const S = AutonomousAI.Schema

@testset "coding agent application" begin
    agent = AC.Agent(verbose = false)
    goal = AC.Goal("smooth a noisy sensor stream";
                   algorithm = :stencil3, input_size = [4096], n_calls = 64)

    plan = AC.default_plan(agent, goal)
    @test any(a -> a isa S.InspectHardware, plan.steps)
    @test any(a -> a isa S.ProposeCandidates, plan.steps)
    @test any(a -> a isa S.GenerateCode, plan.steps)
    @test any(a -> a isa S.BenchmarkAlgorithm, plan.steps)
    @test any(a -> a isa S.VerifyResult, plan.steps)
    @test any(a -> a isa S.ExecuteFinal, plan.steps)

    proposed = only([a for a in plan.steps if a isa S.ProposeCandidates])
    @test length(proposed.candidates) >= 2
    @test all(c -> c.algorithm === goal.algorithm, proposed.candidates)

    # A coding-agent style flow: synthesize candidate code and validate statically.
    optimized = proposed.candidates[end]
    unit = CG.generate(optimized;
        allow_fastmath = agent.engine.policy.allow_fastmath,
        allow_inbounds = agent.engine.policy.allow_inbounds)
    @test unit.report.ok
    @test occursin(CG.entrypoint_name(goal.algorithm), unit.source)

    ranked, rejected = O.rank_candidates(proposed.candidates, goal.input_size,
                                         agent.world, agent.memory,
                                         agent.cost, agent.engine.policy.limits)
    @test !isempty(ranked)
    @test all(x -> x.candidate.algorithm === goal.algorithm, ranked)
    @test length(ranked) + length(rejected) == length(proposed.candidates)
end
