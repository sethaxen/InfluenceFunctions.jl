using StatsBase

skew(p, m) = wmean(p, (X .- m) .^ 3) / wmean(p, (X .- m) .^ 2)^1.5
kurt(p, m) = wmean(p, (X .- m) .^ 4) / wmean(p, (X .- m) .^ 2)^2 - 3

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
    Md = DimArray(M, (Dim{:draw}(1:N), Dim{:param}(1:3)))
    Gd, ψd = value_and_influence(genvar, Md)
    @test Gd == genvar(Md)
    @test ψd isa DimArray
    @test dims(ψd) == (dims(Md, 1),)
    @test parent(ψd) ≈ influence(genvar, M)
    @test eltype(influence(genvar, Float32.(M))) === Float32
    @inferred value_and_influence(genvar, M)
    @inferred value_and_influence(genvar, X)
    @test_throws DimensionMismatch influence(genvar, OffsetArray(M, -2, 5))   # as in StatsBase
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

@testset "partialcor" begin
    x, y = M[:, 1], M[:, 2]
    # The correlation of the residuals of weighted least squares on `[1 Z]`, whose weighted
    # means are 0.
    function T(p, Z)
        A = [ones(N) Z]
        r(v) = v .- A * ((A' * (p .* A)) \ (A' * (p .* v)))
        rx, ry = r(x), r(y)
        return wmean(p, rx .* ry) / sqrt(wmean(p, rx .^ 2) * wmean(p, ry .^ 2))
    end
    for Z in (M[:, 3], [M[:, 3] rand(StableRNG(4), Gamma(2, 1), N)])   # one or two covariates
        ρ, ψ = value_and_influence(partialcor, x, y, Z)              # observations are triples
        check_against_ad(p -> T(p, Z), (ρ, ψ), partialcor(x, y, Z))
        @test ρ == partialcor(x, y, Z)
    end
    Z = M[:, 3:3]
    xd, yd = DimArray(x, Dim{:draw}(1:N)), DimArray(y, Dim{:draw}(1:N))
    Zd = DimArray(Z, (Dim{:draw}(1:N), Dim{:covariate}(1:1)))
    ψd = influence(partialcor, xd, yd, Zd)
    @test ψd isa DimArray
    @test dims(ψd) == dims(xd)
    @test parent(ψd) ≈ influence(partialcor, x, y, Z)
    zd = DimArray(vec(Z), Dim{:draw}(1:N))                  # one covariate, as a vector
    @test parent(influence(partialcor, xd, yd, zd)) ≈ influence(partialcor, x, y, vec(Z))
    @test eltype(influence(partialcor, Float32.(x), Float32.(y), Float32.(Z))) === Float32
    ρm, ψm = value_and_influence(partialcor, Float32.(x), Float32.(y), Z)   # mixed precision
    @test eltype(ψm) === typeof(ρm)
    @inferred value_and_influence(partialcor, x, y, Z)
    @inferred value_and_influence(partialcor, x, y, vec(Z))
    @test_throws DimensionMismatch influence(partialcor, x, y[1:(end - 1)], Z)
end

@testset "cumulant" begin
    function T(p, c)
        m₂, m₃, m₄, m₅, m₆, m₇, m₈ = (wmean(p, (X .- c) .^ i) for i in 2:8)
        return (
            c, m₂, m₃, m₄ - 3 * m₂^2, m₅ - 10 * m₂ * m₃,
            m₆ - 15 * m₂ * m₄ - 10 * m₃^2 + 30 * m₂^3,
            m₇ - 21 * m₂ * m₅ - 35 * m₃ * m₄ + 210 * m₂^2 * m₃,
            m₈ - 28 * m₂ * m₆ - 56 * m₃ * m₅ - 35 * m₄^2 + 420 * m₂^2 * m₄ + 560 * m₂ * m₃^2 - 630 * m₂^4,
        )
    end
    for k in 1:8
        check_against_ad(p -> T(p, wmean(p, X))[k], value_and_influence(cumulant, X, k), cumulant(X, k))
        check_against_ad(p -> T(p, C)[k], value_and_influence(cumulant, X, k, C), cumulant(X, k, C))
    end
    check_shapes(cumulant, X, 4)
    check_shapes(cumulant, X, 4, C)
    v, ψ = value_and_influence(cumulant, X, 2:2:6)          # orders given as a range
    @test v ≈ cumulant(X, 2:2:6)
    @test size(ψ) == (3,)
    @test ψ[2] == influence(cumulant, X, 4)
    @test influence(cumulant, X, 1:3, C)[3] == influence(cumulant, X, 3, C)
    @inferred value_and_influence(cumulant, X, 1:4)
    @inferred value_and_influence(cumulant, X, 1:4, C)
    xo = OffsetArray(reshape(X, 9, 4), -2, 5)               # the data's axes on the inside
    @test all(ψ -> axes(ψ) == axes(xo), influence(cumulant, xo, 1:3))
    xd = DimArray(reshape(X, 9, 4), (Dim{:draw}(1:9), Dim{:chain}(1:4)))
    @test all(ψ -> dims(ψ) == dims(xd), influence(cumulant, xd, 1:3))
    v, ψ = value_and_influence(cumulant, Float64[], 2)      # no data: NaN, as for `var`
    @test isnan(v) && isempty(ψ)
    @test eltype(influence(cumulant, Float32.(X), 3, C)) === Float64    # the center is Float64
    x32 = rand(StableRNG(8), Float32, 10^6)                 # a running sum is too coarse for this
    @test first(value_and_influence(cumulant, x32, 2)) ≈ var(Float64.(x32); corrected = false) rtol = 1.0e-6
    @test influence(cumulant, [1, 2, 6], 3, 2) ≈ [-22, -21, 43]         # d³ - m₃, with m₃ = 21
    # The powers of these deviations overflow as integers, which StatsBase.moment does not avoid.
    @test influence(cumulant, [10^7, 2 * 10^7, 6 * 10^7], 3, 10^7) ≈ [-4.2e22, -4.1e22, 8.3e22]
    (κ₂, κ₄), (ψ₂, ψ₄) = value_and_influence(cumulant, X, 2:2:4)    # kurtosis is κ₄ / κ₂²
    @test ψ₄ ./ κ₂^2 .- (2 * κ₄ / κ₂^3) .* ψ₂ ≈ influence(kurtosis, X)
    @test_throws ArgumentError influence(cumulant, X, 0)
    @test_throws ArgumentError influence(cumulant, X, 0:2)
    @test_throws MethodError influence(cumulant, X, [1, 2])  # StatsBase takes an integer or a range
    @test_throws MethodError influence(cumulant, X, 2, pweights(ones(N)))
end

@testset "proportions" begin
    k = rand(StableRNG(6), 1:4, N)
    l = rand(StableRNG(7), 1:3, N)
    v, ψ = value_and_influence(proportions, k, 1:4)
    @test v == proportions(k, 1:4)
    @test size(ψ) == size(v)
    for c in 1:4
        check_against_ad(p -> wmean(p, k .== c), (v[c], ψ[c]), v[c])
    end
    @test value_and_influence(proportions, k) == (v, ψ)
    @test value_and_influence(proportions, k, 4) == (v, ψ)
    @test first(influence(proportions, k .+ 10)) == ψ[1]    # levels from the data
    v, ψ = value_and_influence(proportions, k, 2:3)         # a level is a level, wherever it is
    @test v == proportions(k, 2:3)
    @test ψ[1] == (k .== 2) .- v[1]

    v, ψ = value_and_influence(proportions, k, l, (1:4, 1:3))
    @test v == proportions(k, l, (1:4, 1:3))
    @test size(ψ) == size(v)
    for I in CartesianIndices(v)
        check_against_ad(p -> wmean(p, (k .== I[1]) .& (l .== I[2])), (v[I], ψ[I]), v[I])
    end
    @test value_and_influence(proportions, k, l) == (v, ψ)
    @test value_and_influence(proportions, k, l, (4, 3)) == (v, ψ)
    @test influence(proportions, k, l, 3) == ψ[1:3, 1:3]
    odds(q) = (q[1, 1] * q[2, 2]) / (q[1, 2] * q[2, 1])
    T(p) = odds([wmean(p, (k .== a) .& (l .== b)) for a in 1:4, b in 1:3])
    @test sum(ForwardDiff.gradient(odds, v) .* ψ) ≈ ad_influence(T, N) rtol = 1.0e-10

    km = reshape(k, 9, 4)                                   # the data's axes on the inside
    @test size(influence(proportions, km)[1]) == size(km)
    ko = OffsetArray(km, -2, 5)
    @test axes(influence(proportions, ko)[1]) == axes(ko)
    kd = DimArray(km, (Dim{:draw}(1:9), Dim{:chain}(1:4)))
    @test dims(influence(proportions, kd)[1]) == dims(kd)
    @inferred value_and_influence(proportions, k)
    @inferred value_and_influence(proportions, k, 2:3)
    @inferred value_and_influence(proportions, k, l)
    @inferred value_and_influence(proportions, k, l, (4, 3))
    @test_throws MethodError influence(proportions, float.(k))      # as in StatsBase
    @test_throws MethodError influence(proportions, k, l, 1:3)      # as in StatsBase
    @test_throws MethodError influence(proportions, k, pweights(ones(N)))
    # Integer weights are integer arrays too, but they are weights, not a second variable.
    @test_throws MethodError influence(proportions, k, fweights(ones(Int, N)))
    @test_throws DimensionMismatch influence(proportions, k, l[1:3])
end

@testset "weights are not taken for data" begin
    w = fweights(ones(Int, N))
    # Taken for the `x` of `mean(f, x)`, weights would make the data `f`, and calling it would
    # throw a `MethodError` too, so check whose error it is.
    err = @test_throws MethodError influence(mean, X, w)
    @test err.value.f === value_and_influence
    @test_throws MethodError influence(cov, M, w)
    @test_throws MethodError influence(cor, M, w)
end
