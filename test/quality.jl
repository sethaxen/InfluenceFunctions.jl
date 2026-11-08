using Aqua, JET, StatsBase

@testset "Aqua" begin
    Aqua.test_all(InfluenceFunctions)
end

# Aqua's ambiguity check does not catch ambiguities between core methods and those of the
# extension, and an ambiguous call throws a `MethodError` that `@test_throws` cannot tell
# from the one a method throws on purpose.
@testset "Ambiguities with StatsBaseExt" begin
    ext = Base.get_extension(InfluenceFunctions, :StatsBaseExt)
    @test isempty(Test.detect_ambiguities(InfluenceFunctions, ext))
end

# One representative call per method of the StatsBase extension.
const EXT_CALLS = Any[
    (geomean, (X,), (;)),
    (harmmean, (X,), (;)),
    (genmean, (X, 2.5), (;)),
    (moment, (X, 3), (;)),
    (moment, (X, 3, C), (;)),
    (skewness, (X,), (;)),
    (skewness, (X, C), (;)),
    (kurtosis, (X,), (;)),
    (kurtosis, (X, C), (;)),
    (variation, (X,), (; corrected = false)),
    (variation, (X, C), (;)),
    (percentile, (X, 30), (;)),
    (percentile, (X, [25, 75]), (;)),
    (iqr, (X,), (;)),
    (mad, (X,), (; normalize = true)),
    (mean_and_var, (X,), (;)),
    (mean_and_var, (reshape(X, 9, 4), 1), (;)),
    (mean_and_std, (X,), (;)),
    (mean_and_std, (reshape(X, 9, 4), 1), (;)),
    (mean_and_cov, (M,), (;)),
    (mean_and_cov, (M, 2), (;)),
    (scattermat, (M,), (;)),
    (scattermat, (M,), (; mean = 0, dims = 2)),
    (genvar, (M,), (;)),
    (genvar, (X,), (;)),
    (totalvar, (M,), (;)),
    (totalvar, (X,), (;)),
    (cumulant, (X, 4), (;)),
    (cumulant, (X, 1:4), (;)),
    (cumulant, (X, 1:4, C), (;)),
    (proportions, ([1, 2, 2],), (;)),
    (proportions, ([1, 2, 2], 3), (;)),
    (proportions, ([1, 2, 2], 1:3), (;)),
    (proportions, ([1, 2, 2], [1, 1, 2]), (;)),
    (proportions, ([1, 2, 2], [1, 1, 2], 2), (;)),
    (proportions, ([1, 2, 2], [1, 1, 2], (2, 2)), (;)),
    (proportions, ([1, 2, 2], [1, 1, 2], (1:2, 1:2)), (;)),
]

# JET's analysis is tied to compiler internals, so skip it on prereleases.
if isempty(VERSION.prerelease)
    @testset "JET" begin
        JET.test_package(InfluenceFunctions; target_modules = (InfluenceFunctions,))
    end
    # `test_package` does not analyze extensions, so check one call per extension method.
    # Target both modules, so errors in core code the extension calls are reported too.
    @testset "JET extension" begin
        ext = Base.get_extension(InfluenceFunctions, :StatsBaseExt)
        for (f, args, kwargs) in EXT_CALLS
            # As with `@inferred`, splatting empty keywords into a method without keywords
            # trips JET up, so pass keywords only when there are some.
            if isempty(kwargs)
                JET.@test_call target_modules = (InfluenceFunctions, ext) value_and_influence(f, args...)
            else
                JET.@test_call target_modules = (InfluenceFunctions, ext) value_and_influence(f, args...; kwargs...)
            end
        end
    end
end
