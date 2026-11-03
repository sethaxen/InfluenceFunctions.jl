using StatsBase

skew(p, m) = wmean(p, (X .- m) .^ 3) / wmean(p, (X .- m) .^ 2)^1.5
kurt(p, m) = wmean(p, (X .- m) .^ 4) / wmean(p, (X .- m) .^ 2)^2 - 3

@testset "geomean" begin
    check_against_ad(p -> exp(wmean(p, log.(X))), value_and_influence(geomean, X), geomean(X))
    check_shapes(geomean, X)
    @test_throws DomainError influence(geomean, [-1.0, 2.0])
end

@testset "harmmean" begin
    check_against_ad(p -> inv(wmean(p, inv.(X))), value_and_influence(harmmean, X), harmmean(X))
    check_shapes(harmmean, X)
end

@testset "genmean" begin
    for r in (-1.5, 2.5)
        check_against_ad(p -> wmean(p, X .^ r)^(1 / r), value_and_influence(genmean, X, r), genmean(X, r))
    end
    check_shapes(genmean, X, 2.5)
    @test influence(genmean, X, 0) ≈ influence(geomean, X)
    @test influence(genmean, X, 1) ≈ influence(mean, X)
end

@testset "moment" begin
    for k in 2:5
        check_against_ad(p -> wmean(p, (X .- wmean(p, X)) .^ k), value_and_influence(moment, X, k), moment(X, k))
        check_against_ad(p -> wmean(p, (X .- C) .^ k), value_and_influence(moment, X, k, C), moment(X, k, C))
    end
    check_shapes(moment, X, 3)
    check_shapes(moment, X, 3, C)
    @test maximum(abs, influence(moment, X, 1)) < 1.0e-12
end

@testset "skewness" begin
    check_against_ad(p -> skew(p, wmean(p, X)), value_and_influence(skewness, X), skewness(X))
    check_against_ad(p -> skew(p, C), value_and_influence(skewness, X, C), skewness(X, C))
    check_shapes(skewness, X)
    check_shapes(skewness, X, C)
    @test all(isnan, influence(skewness, fill(2.0, 12)))
end

@testset "kurtosis" begin
    check_against_ad(p -> kurt(p, wmean(p, X)), value_and_influence(kurtosis, X), kurtosis(X))
    check_against_ad(p -> kurt(p, C), value_and_influence(kurtosis, X, C), kurtosis(X, C))
    check_shapes(kurtosis, X)
    check_shapes(kurtosis, X, C)
end

@testset "variation" begin
    for corrected in (true, false)
        b = corrected ? N / (N - 1) : 1.0
        check_against_ad(
            p -> (m = wmean(p, X); sqrt(b * wmean(p, (X .- m) .^ 2)) / m),
            value_and_influence(variation, X; corrected), variation(X; corrected)
        )
        check_against_ad(
            p -> sqrt(b * wmean(p, (X .- C) .^ 2)) / C,
            value_and_influence(variation, X, C; corrected), variation(X, C; corrected)
        )
    end
    check_shapes(variation, X)
    check_shapes(variation, X, C)
    @test_throws MethodError influence(variation, X; dims = 1)    # StatsBase.variation has no dims
    @test all(iszero, influence(variation, fill(2.0, 10)))        # constant data: V = 0, defined
    @test all(iszero, influence(variation, fill(2.0, 10), 2.0))
end
