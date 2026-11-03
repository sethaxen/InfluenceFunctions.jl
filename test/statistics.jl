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
    @test all(isnan, influence(std, fill(2.0, 10)))      # 0/0
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
