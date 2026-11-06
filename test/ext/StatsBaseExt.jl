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

@testset "percentile" begin
    v, ψ = value_and_influence(percentile, X, 90; density = D)
    @test v ≈ percentile(X, 90)
    @test ψ ≈ influence(quantile, X, 90 * 0.01; density = D)
    # the same level as StatsBase (p * 0.01, which differs from p / 100 in the last bit for some p)
    @test all(p -> first(value_and_influence(percentile, X, p; density = D)) == percentile(X, p), 1:99)
    for p in (0, 100, -1, 150)
        @test_throws DomainError influence(percentile, X, p)
    end
    check_shapes(percentile, X, 30; vector_only = true)   # via Statistics.quantile: vectors only
end

@testset "iqr" begin
    v, ψ = value_and_influence(iqr, X)
    @test v ≈ iqr(X)
    @test ψ ≈ influence(quantile, X, 0.75) .- influence(quantile, X, 0.25)
    check_shapes(iqr, X)
    @test eltype(influence(iqr, big.(X))) === BigFloat
    @test all(isfinite, influence(iqr, rand(StableRNG(4), 1:5, 1000)))   # ties
end

function mad_expected(x, dens; center, normalize)
    m = center === nothing ? median(x) : center
    s = mad(x; center = m, normalize = false)
    fp, fm = dens(m + s), dens(m - s)
    ψm = center === nothing ? (0.5 .- (x .<= m)) ./ dens(m) : zero(x)
    k = normalize ? inv(quantile(Normal(), 3 / 4)) : 1.0
    return k .* ((0.5 .- (abs.(x .- m) .<= s)) .- (fp - fm) .* ψm) ./ (fp + fm)
end

@testset "mad" begin
    for normalize in (true, false), center in (nothing, C)
        v, ψ = value_and_influence(mad, X; center, normalize, density = D)
        @test v ≈ mad(X; center, normalize)
        @test ψ ≈ mad_expected(X, t -> pdf(D, t); center, normalize)
    end
    xn = rand(StableRNG(3), Normal(), 100_000)                          # asymptotic variance
    @test mean(abs2, influence(mad, xn; normalize = true)) ≈ 1.360459304311955 rtol = 0.1
    xg = rand(StableRNG(2), D, 100_000)
    @test mean(abs2, influence(mad, xg; normalize = false)) ≈
        mean(abs2, mad_expected(xg, t -> pdf(D, t); center = nothing, normalize = false)) rtol = 0.1
    check_shapes(mad, X; normalize = true)
    for x in (X, Float32.(X))                                           # the value matches StatsBase exactly
        @test first(value_and_influence(mad, x; normalize = true)) === mad(x; normalize = true)
    end
    @test all(isfinite, influence(mad, rand(StableRNG(4), 1:5, 1000); normalize = true))   # ties
    @test all(isfinite, influence(mad, Float16[1, 2, 3, 5, 9]; normalize = true))
    @test parent(influence(mad, OffsetArray(X, -3); normalize = true)) == influence(mad, X; normalize = true)
    # m̂ and m̂ ± ŝ inside atoms: the limiting form gives 0, not NaN
    for z in (rand(StableRNG(2), 0:2, 1000), [zeros(400); ones(600)], fill(2.0, 10), [3.0])
        @test all(iszero, influence(mad, z; normalize = true))
    end
    @test_throws MethodError influence(mad, X; constant = 1.0)            # deprecated in StatsBase
end
