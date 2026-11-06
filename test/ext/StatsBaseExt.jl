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
    ps = [25 50; 60 75]                                   # several levels
    v, ψ = value_and_influence(percentile, X, ps; density = D)
    @test v == percentile(X, ps)
    @test size(ψ) == size(ps)
    @test ψ[2, 1] == influence(percentile, X, 60; density = D)
    @test_throws DomainError influence(percentile, X, [50, 100])
    @test_throws MethodError influence(percentile, X, (25, 75))   # StatsBase.percentile rejects tuples
end

@testset "iqr" begin
    v, ψ = value_and_influence(iqr, X)
    @test v ≈ iqr(X)
    @test ψ ≈ influence(quantile, X, 0.75) .- influence(quantile, X, 0.25)
    check_shapes(iqr, X)
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
    # m̂ and m̂ ± ŝ inside atoms: the limiting form gives 0, not NaN
    for z in (rand(StableRNG(2), 0:2, 1000), [zeros(400); ones(600)], fill(2.0, 10), [3.0])
        @test all(iszero, influence(mad, z; normalize = true))
    end
    @test_throws MethodError influence(mad, X; constant = 1.0)            # deprecated in StatsBase
end

@testset "$f" for (f, g) in ((mean_and_var, var), (mean_and_std, std))
    A = reshape(X, 9, 4)
    for corrected in (true, false)
        v, ψ = value_and_influence(f, X; corrected)
        @test v == f(X; corrected)
        @test ψ == (influence(mean, X), influence(g, X; corrected))
        for dim in (1, 2)
            v, ψ = value_and_influence(f, A, dim; corrected)
            @test v == f(A, dim; corrected)
            @test ψ == (influence(mean, A; dims = dim), influence(g, A; dims = dim, corrected))
        end
    end
    xo = OffsetArray(A, -2, 5)
    @test all(ψ -> axes(ψ) == axes(xo), influence(f, xo))
    xd = DimArray(A, (Dim{:draw}(1:9), Dim{:chain}(1:4)))
    @test all(ψ -> dims(ψ) == dims(xd), influence(f, xd))
    @test all(ψ -> dims(ψ) == dims(xd), influence(f, xd, 1))
    @inferred value_and_influence(f, X)
    @inferred value_and_influence(f, A, 1)
    @test_throws MethodError influence(f, A; dims = 1)   # StatsBase takes the dimension positionally
    @test_throws MethodError influence(f, X; density = nothing)
end

@testset "delta method" begin
    v, ψ = value_and_influence(mean_and_std, X)
    g = ForwardDiff.gradient(w -> w[2] / w[1], collect(v))
    @test g[1] .* ψ[1] .+ g[2] .* ψ[2] ≈ influence(variation, X)
    v, ψ = value_and_influence(quantile, X, (0.25, 0.75))
    @test ψ[2] .- ψ[1] ≈ influence(iqr, X)
end

@testset "mean_and_cov" begin
    for dims in (1, 2), corrected in (true, false)
        x = dims == 1 ? M : permutedims(M)
        v, ψ = value_and_influence(mean_and_cov, x, dims; corrected)
        @test v == mean_and_cov(x, dims; corrected)
        @test ψ[1] == influence(mean, x; dims)
        @test ψ[2] == influence(cov, x; dims, corrected)
    end
    @test value_and_influence(mean_and_cov, M) == value_and_influence(mean_and_cov, M, 1)
    # Rows are the observations, so the mean's influence values are summed across columns.
    (m, S), (ψm, ψS) = value_and_influence(mean_and_cov, M)
    gm = ForwardDiff.gradient(w -> only(w * (S \ w')), m)
    gS = ForwardDiff.gradient(W -> only(m * (W \ m')), S)
    function T(p)
        c = sum(p .* M; dims = 1)
        return only(c * (wcov(p, N / (N - 1)) \ c'))
    end
    @test vec(sum(gm .* ψm; dims = 2)) .+ sum(gS .* ψS) ≈ ad_influence(T, N) rtol = 1.0e-10
    @inferred value_and_influence(mean_and_cov, M)
    # StatsBase.mean_and_cov takes only a DenseMatrix, with the dimension given positionally.
    @test_throws MethodError influence(mean_and_cov, DimArray(M, (Dim{:draw}(1:N), Dim{:param}(1:3))))
    @test_throws MethodError influence(mean_and_cov, M; dims = 1)
    @test_throws MethodError influence(mean_and_cov, M, pweights(ones(N)))
end

@testset "scattermat" begin
    function T(p, c)
        Z = M .- c
        return N .* (Z' * (p .* Z))
    end
    S, ψ = value_and_influence(scattermat, M)
    @test S == scattermat(M)
    @test size(ψ) == size(S)
    for I in CartesianIndices(S)
        check_against_ad(p -> T(p, sum(p .* M; dims = 1))[I], (S[I], ψ[I]), S[I])
    end
    @test all(ψ .≈ (N - 1) .* influence(cov, M))
    for c in (0, 1.5, [1.0 2.0 3.0])                        # a known center
        Sc, ψc = value_and_influence(scattermat, M; mean = c)
        @test Sc == scattermat(M; mean = c)
        for I in CartesianIndices(Sc)
            check_against_ad(p -> T(p, c)[I], (Sc[I], ψc[I]), Sc[I])
        end
    end
    St, ψt = value_and_influence(scattermat, permutedims(M); dims = 2)   # variables as rows
    @test St ≈ S
    @test all(ψt .≈ ψ)
    @test all(influence(scattermat, permutedims(M); dims = 2, mean = [1.0, 2.0, 3.0]) .≈ influence(scattermat, M; mean = [1.0 2.0 3.0]))
    @test eltype(influence(scattermat, Float32.(M))[1, 2]) === Float32
    @inferred value_and_influence(scattermat, M)
    @inferred value_and_influence(scattermat, M; dims = 2)
    @inferred value_and_influence(scattermat, M; mean = 0)
    @test_throws ArgumentError influence(scattermat, M; dims = 3)   # accepted by StatsBase.scattermat
    # StatsBase.scattermat takes only a DenseMatrix.
    @test_throws MethodError influence(scattermat, DimArray(M, (Dim{:draw}(1:N), Dim{:param}(1:3))))
    @test_throws MethodError influence(scattermat, M, pweights(ones(N)))
    @test_throws MethodError influence(scattermat, M; density = nothing)
end

@testset "cumulant" begin
    function T(p, c)
        m₂, m₃, m₄, m₅, m₆ = (wmean(p, (X .- c) .^ i) for i in 2:6)
        return (c, m₂, m₃, m₄ - 3 * m₂^2, m₅ - 10 * m₂ * m₃, m₆ - 15 * m₂ * m₄ - 10 * m₃^2 + 30 * m₂^3)
    end
    for k in 1:6
        check_against_ad(p -> T(p, wmean(p, X))[k], value_and_influence(cumulant, X, k), cumulant(X, k))
        check_against_ad(p -> T(p, C)[k], value_and_influence(cumulant, X, k, C), cumulant(X, k, C))
    end
    check_shapes(cumulant, X, 4)
    check_shapes(cumulant, X, 4, C)
    v, ψ = value_and_influence(cumulant, X, 2:2:6)          # orders given as a range
    @test v == cumulant(X, 2:2:6)
    @test size(ψ) == (3,)
    @test ψ[2] == influence(cumulant, X, 4)
    @test influence(cumulant, X, 1:3, C)[3] == influence(cumulant, X, 3, C)
    @inferred value_and_influence(cumulant, X, 1:4)
    @inferred value_and_influence(cumulant, X, 1:4, C)
    (κ₂, κ₄), (ψ₂, ψ₄) = value_and_influence(cumulant, X, 2:2:4)    # kurtosis is κ₄ / κ₂²
    @test ψ₄ ./ κ₂^2 .- (2 * κ₄ / κ₂^3) .* ψ₂ ≈ influence(kurtosis, X)
    @test_throws ArgumentError influence(cumulant, X, 0)
    @test_throws ArgumentError influence(cumulant, X, 0:2)
    @test_throws MethodError influence(cumulant, X, [1, 2])  # StatsBase takes an integer or a range
    @test_throws MethodError influence(cumulant, X, 2, pweights(ones(N)))
end

@testset "weights are not a second variable" begin
    w = fweights(ones(Int, N))
    @test_throws MethodError influence(cov, M, w)
    @test_throws MethodError influence(cor, M, w)
end
