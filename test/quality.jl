using Aqua, JET

@testset "Aqua" begin
    Aqua.test_all(InfluenceFunctions)
end

# JET's analysis is tied to compiler internals, so skip it on prereleases.
if isempty(VERSION.prerelease)
    @testset "JET" begin
        JET.test_package(InfluenceFunctions; target_modules = (InfluenceFunctions,))
    end
end
