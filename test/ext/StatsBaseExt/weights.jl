using StatsBase

@testset "weighted mean" begin
    for W in (fweights, pweights, aweights, weights)
        check_weighted_against_ad(u -> mean(X, W(u)), value_and_influence(mean, X, W(WV)), WV)
    end
    # Influence values are scaled so that their standard error under the weights is that of the
    # estimate, wherever StatsBase defines one.
    for w in (fweights(WV), pweights(WV), uweights(N))
        @test sem(influence(mean, X, w), w) ≈ sem(X, w)
    end
    # Frequency weights count repeats of an observation.
    @test inverse_rle(influence(mean, X, fweights(WV)), WV) ≈ influence(mean, inverse_rle(X, WV))
    @test value_and_influence(mean, X, uweights(N)) == value_and_influence(mean, X)
    A = reshape(X, 9, 4)
    for dims in (1, 2)
        check_dims(mean, A, pweights(WV[1:size(A, dims)]); dims)
    end
    check_shapes(mean, X, fweights(WV))
    @test_throws DimensionMismatch influence(mean, X, pweights(WV[1:5]))
    @test_throws MethodError influence(mean, X, pweights(WV); density = nothing)
end
