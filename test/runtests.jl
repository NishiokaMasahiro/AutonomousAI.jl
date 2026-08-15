using Test
using AutonomousAI

const AAI = AutonomousAI

@testset "AutonomousAI.jl" begin
    include("core_tests.jl")
    include("codegen_tests.jl")
    include("safety_tests.jl")
    include("optimization_tests.jl")
    include("verification_tests.jl")
    include("coding_agent_application_tests.jl")
    include("software_security_application_tests.jl")
    include("sandbox_tests.jl")
    include("gpu_tests.jl")
    include("integration_tests.jl")
end
