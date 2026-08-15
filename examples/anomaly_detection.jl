# Example 1 -- the section 45 workflow, end to end, on a single machine.
#
#   natural language goal -> plan -> hardware inspection -> candidate enumeration
#   -> code generation -> sandboxed benchmark -> verification -> execution
#   -> performance report -> memory update
#
# Run:  julia --project=. examples/anomaly_detection.jl

using AutonomousAI
const AC = AutonomousAI.AgentCore

agent = AC.Agent(verbose = true)

println(AutonomousAI.WorldModel.describe(agent.world))
println()

goal = AC.Goal("Analyse this sensor stream and detect anomalous samples as fast as possible";
               input_size = [2_000_000], n_calls = 200, rtol = 1e-6)

report = AC.run_goal!(agent, goal)
AutonomousAI.Interface.print_report(report)

# The second run over the same goal should consult memory instead of re-deriving
# its priors from the roofline model: watch the `evidence` column change from
# `roofline` to `measured`.
println("\n=== second run (memory-informed) ===\n")
report2 = AC.run_goal!(agent, goal)
AutonomousAI.Interface.print_report(report2)

AutonomousAI.Memory.save_memory(agent.memory)
println("\nmemory persisted to ", agent.memory.dir)
