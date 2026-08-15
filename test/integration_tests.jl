using Test
using AutonomousAI
const AC = AutonomousAI.AgentCore
const SB = AutonomousAI.Sandbox

@testset "end to end" begin
    dir = mktempdir()
    agent = AC.Agent(memory = AutonomousAI.Memory.MemorySystem(dir), verbose = false)
    goal = AC.Goal("detect outliers in a small sensor buffer";
                   input_size = [20_000], n_calls = 10)
    report = AC.run_goal!(agent, goal)

    @test report.goal.algorithm === :zscore_anomaly
    @test !isempty(report.decisions)
    @test all(d -> d.at >= goal.created, report.decisions)

    if SB.sandbox_available()
        @test report.baseline !== nothing
        @test report.verification !== nothing
        @test report.verification.passed
        @test AutonomousAI.Memory.n_records(agent.memory.hardware) >= 2
        # memory must now be able to answer the same question without measuring
        t, conf = AutonomousAI.Memory.predict_runtime(agent.memory.hardware,
            report.baseline.candidate, 20_000;
            hw = agent.world.hardware.fingerprint,
            sw = AutonomousAI.HAL.software_fingerprint())
        @test isfinite(t) && conf > 0
    else
        @info "sandbox unavailable: end-to-end assertions limited to planning"
    end

    @test length(agent.memory.episodic.episodes) == 1
    AutonomousAI.Memory.save_memory(agent.memory)
    reloaded = AutonomousAI.Memory.load_memory(dir)
    @test length(reloaded.episodic.episodes) == 1
end
