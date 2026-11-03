using InfluenceFunctions
using Test

# Run a subset with: Pkg.test(; test_args=["statistics"])
const TEST_GROUPS = ["InfluenceFunctions"]
const SELECTED = isempty(ARGS) ? [TEST_GROUPS; "quality"] : ARGS

isfile(joinpath(@__DIR__, "testutils.jl")) && include("testutils.jl")

@testset "InfluenceFunctions.jl" begin
    for group in SELECTED
        @testset "$group" begin
            include("$group.jl")
        end
    end
end
