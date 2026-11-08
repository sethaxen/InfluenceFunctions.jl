using StatsBase

@testset "weighted mean" begin
    w = fweights(ones(Int, N))
    # Taken for the `x` of `mean(f, x)`, weights would make the data `f`, and calling it would
    # throw a `MethodError` too, so check whose error it is.
    err = @test_throws MethodError influence(mean, X, w)
    @test err.value.f === value_and_influence
end
