using DensityInterface: funcdensity
using InfluenceFunctions: fit_density, _density_at_quantile

@testset "mean(x)" begin
    check_against_ad(p -> wmean(p, X), value_and_influence(mean, X), mean(X))
    check_shapes(mean, X)
    @test influence(mean, [1, 2, 6]) == [-2.0, -1.0, 3.0]       # Int data → Float64
    @test influence(mean, 1.0:3.0) == [-1.0, 0.0, 1.0]          # ranges work
    v, ψ = value_and_influence(mean, Float64[])
    @test isnan(v) && ψ == Float64[]
    A, B = reshape(X, 9, 4), reshape(X, 3, 4, 3)
    for (x, dims) in ((A, 1), (A, 2), (A, (1, 2)), (A, :), (B, (1, 3)), (B, 2))
        check_dims(mean, x; dims)
    end
    @inferred value_and_influence(mean, A; dims = 1)
    @test_throws MethodError influence(mean, X; density = nothing)   # strict keywords
    @test_throws MethodError influence(mean, [1.0, missing])         # missing is unsupported
end

@testset "mean(f, x)" begin
    check_against_ad(p -> wmean(p, sqrt.(X)), value_and_influence(mean, sqrt, X), mean(sqrt, X))
    @inferred value_and_influence(mean, sqrt, X)
    @test size(influence(mean, abs2, reshape(X, 9, 4))) == (9, 4)
    @test eltype(influence(mean, t -> t // 1, [1, 2, 6])) === Rational{Int}   # no forced float
    A = reshape(X, 9, 4)
    v, ψ = value_and_influence(mean, sqrt, A; dims = 1)
    @test v ≈ mean(sqrt, A; dims = 1)
    @test size(ψ) == size(A)
    @test ψ[:, 2] ≈ influence(mean, sqrt, A[:, 2])
end

@testset "var" begin
    for corrected in (true, false)
        b = corrected ? N / (N - 1) : 1.0
        Tv = p -> b * wmean(p, (X .- wmean(p, X)) .^ 2)
        Tc = p -> b * wmean(p, (X .- C) .^ 2)
        check_against_ad(Tv, value_and_influence(var, X; corrected), var(X; corrected))
        check_against_ad(Tc, value_and_influence(var, X; corrected, mean = C), var(X; corrected, mean = C))
    end
    check_shapes(var, X; corrected = false)
    check_shapes(var, X; mean = C)
    A, B = reshape(X, 9, 4), reshape(X, 3, 4, 3)
    for corrected in (true, false), (x, dims) in ((A, 1), (A, 2), (A, (1, 2)), (A, :), (B, (1, 3)))
        check_dims(var, x; dims, corrected)
    end
    mc = fill(C, 1, 4)                                    # one fixed center per column
    @test influence(var, A; dims = 1, mean = mc)[:, 3] ≈ influence(var, A[:, 3]; mean = C)
    # as in Statistics: a number center goes with the whole array, an array center with `dims`
    @test_throws MethodError influence(var, A; dims = 1, mean = C)
    @test_throws MethodError influence(var, A; mean = mc)
    v, ψ = value_and_influence(var, [3.0])
    @test isnan(v) && all(isnan, ψ)                      # n = 1, corrected
    @test all(iszero, influence(var, fill(2.0, 10)))     # constant data
end

@testset "std" begin
    for corrected in (true, false)
        b = corrected ? N / (N - 1) : 1.0
        Tv = p -> b * wmean(p, (X .- wmean(p, X)) .^ 2)
        Tc = p -> b * wmean(p, (X .- C) .^ 2)
        check_against_ad(p -> sqrt(Tv(p)), value_and_influence(std, X; corrected), std(X; corrected))
        check_against_ad(p -> sqrt(Tc(p)), value_and_influence(std, X; corrected, mean = C), std(X; corrected, mean = C))
    end
    check_shapes(std, X)
    check_shapes(std, X; mean = C)
    A, B = reshape(X, 9, 4), reshape(X, 3, 4, 3)
    for corrected in (true, false), (x, dims) in ((A, 1), (A, 2), (A, (1, 2)), (A, :), (B, (1, 3)))
        check_dims(std, x; dims, corrected)
    end
    @test influence(std, A; dims = 1, mean = fill(C, 1, 4))[:, 3] ≈ influence(std, A[:, 3]; mean = C)
    # constant data: the value is 0 and moving mass onto any observation changes nothing
    @test all(iszero, influence(std, fill(2.0, 10)))
    @test all(iszero, influence(std, [fill(2.0, 5) X[1:5]]; dims = 1)[:, 1])   # a constant slice
    @test all(isnan, influence(std, [3.0]))              # n = 1, corrected: the value is NaN
end

@testset "varm" begin
    for corrected in (true, false)
        b = corrected ? N / (N - 1) : 1.0
        check_against_ad(p -> b * wmean(p, (X .- C) .^ 2), value_and_influence(varm, X, C; corrected), varm(X, C; corrected))
    end
    check_shapes(varm, X, C)
    A = reshape(X, 9, 4)
    @test influence(varm, A, fill(C, 1, 4); dims = 1)[:, 3] ≈ influence(varm, A[:, 3], C)
    @test_throws MethodError influence(varm, A, C; dims = 1)
    @test_throws MethodError influence(varm, A, fill(C, 1, 4))
end

@testset "stdm" begin
    for corrected in (true, false)
        b = corrected ? N / (N - 1) : 1.0
        check_against_ad(p -> sqrt(b * wmean(p, (X .- C) .^ 2)), value_and_influence(stdm, X, C; corrected), stdm(X, C; corrected))
    end
    check_shapes(stdm, X, C)
    A = reshape(X, 9, 4)
    @test influence(stdm, A, fill(C, 1, 4); dims = 1)[:, 3] ≈ influence(stdm, A[:, 3], C)
    @test all(iszero, influence(stdm, fill(2.0, 10), 2.0))   # constant data at the center
end

@testset "cov" begin
    Y = X .+ randn(StableRNG(2), N)
    for corrected in (true, false)
        b = corrected ? N / (N - 1) : 1.0
        Tcov = p -> b * wmean(p, (X .- wmean(p, X)) .* (Y .- wmean(p, Y)))
        check_against_ad(Tcov, value_and_influence(cov, X, Y; corrected), cov(X, Y; corrected))
    end
    @inferred value_and_influence(cov, X, Y)
    @test_throws DimensionMismatch influence(cov, X, Y[1:(end - 1)])
end

@testset "cor" begin
    Y = X .+ randn(StableRNG(2), N)
    Tcor = p -> begin
        mx, my = wmean(p, X), wmean(p, Y)
        wmean(p, (X .- mx) .* (Y .- my)) / sqrt(wmean(p, (X .- mx) .^ 2) * wmean(p, (Y .- my) .^ 2))
    end
    check_against_ad(Tcor, value_and_influence(cor, X, Y), cor(X, Y))
    @inferred value_and_influence(cor, X, Y)
    @test_throws DimensionMismatch influence(cor, OffsetArray(X, -3), Y)
end

@testset "quantile" begin
    v, ψ = value_and_influence(quantile, X, 0.9; density = D)
    @test v ≈ quantile(X, 0.9)
    @test ψ ≈ (0.9 .- (X .<= v)) ./ pdf(D, v)
    @test influence(quantile, X, 0.9; density = t -> pdf(D, t)) ≈ ψ
    @test influence(quantile, X, 0.9; density = funcdensity(t -> pdf(D, t))) ≈ ψ
    v0, ψ0 = value_and_influence(quantile, X, 0.9; alpha = 0.0, beta = 0.0, density = D)
    @test v0 ≈ quantile(X, 0.9; alpha = 0.0, beta = 0.0)
    @test ψ0 ≈ (0.9 .- (X .<= v0)) ./ pdf(D, v0)
    g = fit_density(QuantileSpacing(), X)
    @test influence(quantile, X, 0.9) ≈ (0.9 .- (X .<= v)) ./ _density_at_quantile(g, v, 0.9)
    @test influence(quantile, X, 0.9; density = ConstantDensity(0.4)) ≈ (0.9 .- (X .<= v)) ./ 0.4
    @test influence(quantile, sort(X), 0.9; sorted = true) ≈ influence(quantile, X, 0.9)[sortperm(X)]

    x = rand(StableRNG(2), D, 100_000)                                  # asymptotic variance
    for p in (0.1, 0.5, 0.9)
        @test mean(abs2, influence(quantile, x, p)) ≈ p * (1 - p) / pdf(D, quantile(D, p))^2 rtol = 0.1
    end

    check_shapes(quantile, X, 0.3; vector_only = true)                  # Statistics.quantile takes only vectors
    for (x, p) in ((X, 0.3f0), (big.(X), 0.3))                          # mixed precision
        vp, ψp = value_and_influence(quantile, x, p)
        @test eltype(ψp) === typeof(vp)
    end
    @test influence(quantile, 1.0e50 .* X, 0.3f0) ≈ 1.0e50 .* influence(quantile, X, 0.3f0)
    @test influence(quantile, 1.0:10.0, 0.3) == influence(quantile, collect(1.0:10.0), 0.3)   # ranges work
    atom = [zeros(300); rand(StableRNG(5), 700) .+ 0.5]                 # quantile inside an atom
    @test all(iszero, influence(quantile, atom, 0.2))
    @test all(iszero, influence(quantile, -atom, 0.8))                  # mirror image: same answer
    @test all(isfinite, influence(quantile, rand(StableRNG(4), 1:5, 1000), 0.3))   # ties
    @test all(iszero, influence(quantile, fill(2.0, 10), 0.3))          # constant data
    for p in (0.0, 1.0, -0.1, 1.5)
        @test_throws DomainError influence(quantile, X, p)
    end
    @test_throws ArgumentError influence(quantile, Float64[], 0.5)
end

@testset "median" begin
    for x in (X, X[1:35])                                               # even and odd n
        v, ψ = value_and_influence(median, x)
        @test v ≈ median(x)
        @test ψ ≈ influence(quantile, x, 0.5)
    end
    check_shapes(median, X)
    @test eltype(influence(median, [1, 2, 3, 4])) === Float64
    A, B = reshape(X, 9, 4), reshape(X, 3, 4, 3)
    for (x, dims) in ((A, 1), (A, 2), (A, (1, 2)), (A, :), (B, (1, 3)))
        check_dims(median, x; dims)
    end
    check_dims(median, A; dims = 1, extra = (; density = D))
    xo = OffsetArray(X, -3)                                             # offset vector
    @test axes(influence(median, xo)) == axes(xo)
    @test parent(influence(median, xo)) == influence(median, X)
    Ao = OffsetArray(A, -3, 2)                                          # each slice is an offset vector
    for dims in (1, 2)
        @test axes(influence(median, Ao; dims)) == axes(Ao)
        @test parent(influence(median, Ao; dims)) == influence(median, A; dims)
    end
    v, ψ = value_and_influence(median, [1.0, NaN, 3.0])                 # as `median`, NaN propagates
    @test isnan(v) && all(isnan, ψ)
    An = copy(A)
    An[3, 2] = NaN
    ψ = influence(median, An; dims = 1)                                 # only through its own slice
    @test all(isnan, ψ[:, 2])
    @test ψ[:, [1, 3, 4]] == influence(median, A; dims = 1)[:, [1, 3, 4]]
    @test influence(median, [3.0]) == [0.0]                             # n = 1
    @test all(isfinite, influence(median, rand(StableRNG(4), 1:5, 1000)))   # ties
    @test all(iszero, influence(median, fill(2.0, 10)))                 # constant data
    @test all(isinf, influence(median, X; density = Returns(0.0)))
    @test_throws ArgumentError influence(median, X; density = 42)
    @test_throws ArgumentError influence(median, Float64[])
end
