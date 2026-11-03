using InfluenceFunctions
using Test
using Aqua
using JET

@testset "InfluenceFunctions.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(InfluenceFunctions)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(InfluenceFunctions; target_defined_modules = true)
    end
    # Write your tests here.
end
