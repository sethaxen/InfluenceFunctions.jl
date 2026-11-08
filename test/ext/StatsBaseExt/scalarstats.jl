using StatsBase

@testset "geomean" begin
    check_against_ad(p -> exp(wmean(p, log.(X))), value_and_influence(geomean, X), geomean(X))
    check_shapes(geomean, X)
    @test_throws DomainError influence(geomean, [-1.0, 2.0])
    for x in (zeros(3), [0.0, 1.0, 2.0])     # the value stays 0 while a zero keeps any mass
        @test all(iszero, influence(geomean, x))
    end
end

@testset "harmmean" begin
    check_against_ad(p -> inv(wmean(p, inv.(X))), value_and_influence(harmmean, X), harmmean(X))
    check_shapes(harmmean, X)
    for x in (zeros(3), [0.0, 1.0, 2.0])     # the value stays 0 while a zero keeps any mass
        @test all(iszero, influence(harmmean, x))
    end
end

@testset "genmean" begin
    for r in (-1.5, 2.5)
        check_against_ad(p -> wmean(p, X .^ r)^(1 / r), value_and_influence(genmean, X, r), genmean(X, r))
    end
    check_shapes(genmean, X, 2.5)
    @test influence(genmean, X, 0) ≈ influence(geomean, X)
    @test influence(genmean, X, 1) ≈ influence(mean, X)
    @test all(iszero, influence(genmean, zeros(3), 2))
    @test all(iszero, influence(genmean, [0.0, 1.0, 2.0], -1))   # the value stays 0 while a zero keeps any mass
    @test all(isfinite, influence(genmean, [0.0, 1.0, 2.0], 2))  # here the value is not 0
    @test all(isnan, influence(genmean, [-2.0, 1.0, 1.0], 1))    # terms cancel, so nothing keeps it at 0
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
    @test_throws MethodError influence(percentile, X, 30; alpha = 0.0)    # percentile takes no quantile keywords
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
    @test first(value_and_influence(variation, X)) === variation(X)
    @test first(value_and_influence(variation, X, C)) === variation(X, C)
    check_shapes(variation, X)
    check_shapes(variation, X, C)
    @test_throws MethodError influence(variation, X; dims = 1)    # StatsBase.variation has no dims
    @test_throws MethodError influence(variation, X; dims = :)
    @test_throws MethodError influence(variation, X, C; dims = :)
    @test all(iszero, influence(variation, fill(2.0, 10)))        # constant data: V = 0, defined
    @test all(iszero, influence(variation, fill(2.0, 10), 2.0))
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
    v, ψ = value_and_influence(mad, [1.0, NaN, 3.0]; normalize = true)  # as `mad`, NaN propagates
    @test isnan(v) && all(isnan, ψ)
    # m̂ and m̂ ± ŝ inside atoms: the limiting form gives 0, not NaN
    for z in (rand(StableRNG(2), 0:2, 1000), [zeros(400); ones(600)], fill(2.0, 10), [3.0])
        @test all(iszero, influence(mad, z; normalize = true))
    end
    # a center below the data puts m̂ - ŝ outside it, where the density is 0
    c = minimum(X) - 1
    s = mad(X; center = c, normalize = false)
    f₊ = InfluenceFunctions.fit_density(QuantileSpacing(), X)(c + s)
    @test influence(mad, X; center = c, normalize = false) ≈ (0.5 .- (abs.(X .- c) .<= s)) ./ f₊
    @test_throws MethodError influence(mad, X; constant = 1.0)            # deprecated in StatsBase
end

@testset "iqr" begin
    v, ψ = value_and_influence(iqr, X)
    @test v ≈ iqr(X)
    @test ψ ≈ influence(quantile, X, 0.75) .- influence(quantile, X, 0.25)
    check_shapes(iqr, X)
    @test eltype(influence(iqr, big.(X))) === BigFloat
    @test all(isfinite, influence(iqr, rand(StableRNG(4), 1:5, 1000)))   # ties
end

@testset "genvar" begin
    det2(S) = S[1, 1] * S[2, 2] - S[1, 2] * S[2, 1]
    det3(S) = S[1, 1] * det2(S[2:3, 2:3]) - S[1, 2] * det2(S[2:3, [1, 3]]) + S[1, 3] * det2(S[2:3, 1:2])
    b = N / (N - 1)
    G, ψ = value_and_influence(genvar, M)                   # the rows are the observations
    check_against_ad(p -> det3(wcov(p, b)), (G, ψ), genvar(M))
    @test G == genvar(M)
    G, ψ = value_and_influence(genvar, M[:, 1:2])
    check_against_ad(p -> det2(wcov(p, b)[1:2, 1:2]), (G, ψ), genvar(M[:, 1:2]))
    @test G == genvar(M[:, 1:2])
    G, ψ = value_and_influence(genvar, M[:, 2:2])           # one column: its variance
    @test G == genvar(M[:, 2:2])
    @test ψ ≈ influence(var, M[:, 2])
    for x in (X, reshape(X, 3, 4, 3))                       # not a matrix: the variance of all elements
        @test value_and_influence(genvar, x) == value_and_influence(var, x)
    end
    # A singular covariance: every centered observation lies in its column space, so moving mass
    # onto it keeps the determinant at 0.
    wcovof(p, A) = (Z = A .- sum(p .* A; dims = 1); b .* (Z' * (p .* Z)))
    for Ms in ([M[:, 1:2] fill(2.0, N)], [M[:, 1] M[:, 1] M[:, 3]])
        @test genvar(Ms) == 0
        ψs = influence(genvar, Ms)
        @test all(iszero, ψs)
        @test ψs ≈ ad_influence(p -> det3(wcovof(p, Ms)), N) atol = 1.0e-12
    end
    Mn = copy(M)
    Mn[2, 2] = NaN
    for Ms in (Mn, M[1:1, :])                               # a NaN value, from a NaN or one row
        Gn, ψn = value_and_influence(genvar, Ms)
        @test isnan(Gn)
        @test all(isnan, ψn)
    end
    Md = DimArray(M, (Dim{:draw}(1:N), Dim{:param}(1:3)))
    Gd, ψd = value_and_influence(genvar, Md)
    @test Gd == genvar(Md)
    @test ψd isa DimArray
    @test dims(ψd) == (dims(Md, 1),)
    @test parent(ψd) ≈ influence(genvar, M)
    @test eltype(influence(genvar, Float32.(M))) === Float32
    @inferred value_and_influence(genvar, M)
    @inferred value_and_influence(genvar, X)
    Mo = OffsetArray(M, -2, 5)
    err = @test_throws Exception genvar(Mo)                 # its type depends on the Julia version
    @test_throws typeof(err.value) influence(genvar, Mo)    # as in StatsBase
    @test_throws MethodError influence(genvar, M; dims = 2)                   # StatsBase.genvar has no dims
end

@testset "totalvar" begin
    trace(S) = sum(j -> S[j, j], axes(S, 1))
    V, ψ = value_and_influence(totalvar, M)                 # the rows are the observations
    check_against_ad(p -> trace(wcov(p, N / (N - 1))), (V, ψ), totalvar(M))
    @test V == totalvar(M)
    @test influence(totalvar, M[:, 2:2]) ≈ influence(var, M[:, 2])    # one column: its variance
    for x in (X, reshape(X, 3, 4, 3))                       # not a matrix: the variance of all elements
        @test value_and_influence(totalvar, x) == value_and_influence(var, x)
    end
    Md = DimArray(M, (Dim{:draw}(1:N), Dim{:param}(1:3)))
    ψd = influence(totalvar, Md)
    @test ψd isa DimArray
    @test dims(ψd) == (dims(Md, 1),)
    @test parent(ψd) ≈ ψ
    Mo = OffsetArray(M, -2, 5)
    ψo = influence(totalvar, Mo)
    @test axes(ψo) == (axes(Mo, 1),)
    @test parent(ψo) ≈ ψ
    @test eltype(influence(totalvar, Float32.(M))) === Float32
    @inferred value_and_influence(totalvar, M)
    @inferred value_and_influence(totalvar, X)
    @test_throws MethodError influence(totalvar, M; dims = 2)    # StatsBase.totalvar has no dims
end
