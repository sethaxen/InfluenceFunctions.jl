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
    xd = DimArray(A, (Dim{:draw}(1:9), Dim{:chain}(1:4)))                # other `dims` forms that `var` takes
    for dims in (:chain, Dim{:chain}())
        @test parent(influence(var, xd; dims)) ≈ influence(var, A; dims = 2)
    end
    @test influence(var, A; dims = (1, 1)) == influence(var, A; dims = 1)
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
    xd = DimArray(A, (Dim{:draw}(1:9), Dim{:chain}(1:4)))                # other `dims` forms that `std` takes
    for dims in (:chain, Dim{:chain}())
        @test parent(influence(std, xd; dims)) ≈ influence(std, A; dims = 2)
    end
    @test influence(std, A; dims = (1, 1)) == influence(std, A; dims = 1)
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

@testset "cov(x)" begin
    for corrected in (true, false)
        b = corrected ? N / (N - 1) : 1.0
        Tv = p -> b * wmean(p, (X .- wmean(p, X)) .^ 2)
        check_against_ad(Tv, value_and_influence(cov, X; corrected), cov(X; corrected))
        @test value_and_influence(cov, X; corrected) == (cov(X; corrected), influence(var, X; corrected))
    end
    check_shapes(cov, X; vector_only = true)
    @test value_and_influence(cov, [1, 2, 6]) == (cov([1, 2, 6]), influence(var, [1, 2, 6]))
    for x in (1:10, 0.1:0.1:2.0, Float64[])                 # where Statistics.var computes it otherwise
        v, ψ = value_and_influence(cov, x)
        @test isequal(v, cov(x))
        @test ψ ≈ influence(var, collect(x))
    end
    @test_throws MethodError influence(cov, X; dims = 1)    # Statistics.cov(x) has no dims
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

@testset "cov(X)" begin
    for corrected in (true, false)
        b = corrected ? N / (N - 1) : 1.0
        C, ψ = value_and_influence(cov, M; corrected)
        @test C == cov(M; corrected)
        @test size(ψ) == size(C)
        for I in CartesianIndices(C)
            check_against_ad(p -> wcov(p, b)[I], (C[I], ψ[I]), C[I])
        end
        Ct, ψt = value_and_influence(cov, permutedims(M); dims = 2, corrected)   # variables as rows
        @test Ct ≈ C
        @test all(ψt .≈ ψ)
    end
    C, ψ = value_and_influence(cov, M)
    @test ψ[1, 2] ≈ influence(cov, M[:, 1], M[:, 2])
    # The gradient of the combined estimate is not symmetric, while ψ is.
    G = ForwardDiff.gradient(S -> S[1, 2] / sqrt(S[1, 1] * S[2, 2]), C)
    @test sum(G .* ψ) ≈ influence(cor, M[:, 1], M[:, 2])
    @inferred value_and_influence(cov, M)
    @inferred value_and_influence(cov, M; dims = 2)
    Md = DimArray(M, (Dim{:draw}(1:N), Dim{:param}(1:3)))   # value axes outside, data axes inside
    Cd, ψd = value_and_influence(cov, Md)
    @test ψd isa DimArray
    @test dims(ψd) == dims(Cd)
    @test dims(ψd[1, 2]) == (dims(Md, 1),)
    @test parent(ψd[1, 2]) ≈ ψ[1, 2]
    # Statistics.cov accepts this and returns a matrix that has nothing to do with the columns.
    @test_throws ArgumentError influence(cov, M; dims = 3)
    @test_throws MethodError influence(cov, M; density = nothing)
end

@testset "cov(X, Y)" begin
    A, B = M[:, 1:2], [M[:, 3] M[:, 1] .^ 2]
    for corrected in (true, false)
        b = corrected ? N / (N - 1) : 1.0
        function T(p)
            ZA, ZB = A .- sum(p .* A; dims = 1), B .- sum(p .* B; dims = 1)
            return b .* (ZA' * (p .* ZB))
        end
        C, ψ = value_and_influence(cov, A, B; corrected)
        @test C == cov(A, B; corrected)
        @test size(ψ) == size(C)
        for I in CartesianIndices(C)
            check_against_ad(p -> T(p)[I], (C[I], ψ[I]), C[I])
        end
        Ct, ψt = value_and_influence(cov, permutedims(A), permutedims(B); dims = 2, corrected)
        @test Ct ≈ C
        @test all(ψt .≈ ψ)
    end
    x = M[:, 3]                                             # a vector is one variable
    v, ψ = value_and_influence(cov, x, B; corrected = false)
    @test v == cov(x, B; corrected = false)
    @test size(ψ) == (1, 2)
    @test ψ[1, 2] ≈ influence(cov, x, B[:, 2]; corrected = false)
    v, ψ = value_and_influence(cov, A, x)
    @test size(ψ) == (2, 1)
    @test ψ[2, 1] ≈ influence(cov, A[:, 2], x)
    v, ψ = value_and_influence(cov, x, permutedims(B); dims = 2)    # also when variables are rows
    @test v == cov(x, permutedims(B); dims = 2)
    @test size(ψ) == (1, 2)
    @test ψ[1, 2] ≈ influence(cov, x, B[:, 2])
    @inferred value_and_influence(cov, A, B)
    @inferred value_and_influence(cov, x, B)
    @inferred value_and_influence(cov, A, B; dims = 2)
    Ad = DimArray(A, (Dim{:draw}(1:N), Dim{:a}(1:2)))
    Bd = DimArray(B, (Dim{:draw}(1:N), Dim{:b}(1:2)))
    Cd, ψd = value_and_influence(cov, Ad, Bd)
    @test ψd isa DimArray
    @test dims(ψd) == dims(Cd)
    @test dims(ψd[1, 2]) == (dims(Ad, 1),)
    @test_throws ArgumentError influence(cov, A, A; dims = 3)   # accepted by Statistics.cov
    @test_throws DimensionMismatch influence(cov, A, B[1:5, :])
    @test_throws MethodError influence(cov, x, x; dims = 1)     # two vectors have no dims
end

@testset "cor(x)" begin
    check_against_ad(p -> 1.0, value_and_influence(cor, X), cor(X))   # 1 for any data
    check_shapes(cor, X; vector_only = true)
    @test influence(cor, [1, 2, 6]) == zeros(3)
    @test influence(cor, [1.0, NaN, Inf]) == zeros(3)   # cor is still 1, so ψ is not NaN
    @test_throws MethodError influence(cor, X; dims = 1)
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

@testset "cor(X)" begin
    function Tcor(p)
        S = wcov(p)
        s = [sqrt(S[j, j]) for j in axes(S, 1)]
        return S ./ (s .* s')
    end
    R, ψ = value_and_influence(cor, M)
    @test R == cor(M)
    @test size(ψ) == size(R)
    for I in CartesianIndices(R)
        if I[1] == I[2]
            @test all(iszero, ψ[I])                     # a variable's correlation with itself is 1
        else
            check_against_ad(p -> Tcor(p)[I], (R[I], ψ[I]), R[I])
        end
    end
    @test ψ[1, 2] ≈ influence(cor, M[:, 1], M[:, 2])
    Rt, ψt = value_and_influence(cor, permutedims(M); dims = 2)   # variables as rows
    @test Rt ≈ R
    @test all(ψt .≈ ψ)
    @inferred value_and_influence(cor, M)
    @inferred value_and_influence(cor, M; dims = 2)
    Md = DimArray(M, (Dim{:draw}(1:N), Dim{:param}(1:3)))
    Rd, ψd = value_and_influence(cor, Md)
    @test ψd isa DimArray
    @test dims(ψd) == dims(Rd)
    @test dims(ψd[1, 2]) == (dims(Md, 1),)
    @test_throws ArgumentError influence(cor, M; dims = 3)
    @test_throws MethodError influence(cor, M; corrected = true)   # cor has no such keyword
end

@testset "cor(X, Y)" begin
    A, B = M[:, 1:2], [M[:, 3] M[:, 1] .^ 2]
    function T(p)
        ZA, ZB = A .- sum(p .* A; dims = 1), B .- sum(p .* B; dims = 1)
        sA, sB = sqrt.(vec(sum(p .* ZA .^ 2; dims = 1))), sqrt.(vec(sum(p .* ZB .^ 2; dims = 1)))
        return (ZA' * (p .* ZB)) ./ (sA .* sB')
    end
    R, ψ = value_and_influence(cor, A, B)
    @test R == cor(A, B)
    @test size(ψ) == size(R)
    for I in CartesianIndices(R)
        check_against_ad(p -> T(p)[I], (R[I], ψ[I]), R[I])
    end
    Rt, ψt = value_and_influence(cor, permutedims(A), permutedims(B); dims = 2)
    @test Rt ≈ R
    @test all(ψt .≈ ψ)
    x = M[:, 3]                                             # a vector is one variable
    v, ψ = value_and_influence(cor, x, B)
    @test v == cor(x, B)
    @test size(ψ) == (1, 2)
    @test ψ[1, 2] ≈ influence(cor, x, B[:, 2])
    v, ψ = value_and_influence(cor, x, permutedims(B); dims = 2)
    @test size(ψ) == (1, 2)
    @test ψ[1, 2] ≈ influence(cor, x, B[:, 2])
    @inferred value_and_influence(cor, A, B)
    @inferred value_and_influence(cor, x, B)
    @inferred value_and_influence(cor, A, B; dims = 2)
    Ad = DimArray(A, (Dim{:draw}(1:N), Dim{:a}(1:2)))
    Bd = DimArray(B, (Dim{:draw}(1:N), Dim{:b}(1:2)))
    Rd, ψd = value_and_influence(cor, Ad, Bd)
    @test ψd isa DimArray
    @test dims(ψd) == dims(Rd)
    @test dims(ψd[1, 2]) == (dims(Ad, 1),)
    # Statistics.cor accepts this, except for older versions, which throw the DimensionMismatch.
    @test_throws Union{ArgumentError, DimensionMismatch} influence(cor, A, A; dims = 3)
    @test_throws MethodError influence(cor, A, B; corrected = true)
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

@testset "quantile at several levels" begin
    expected(p, q) = (p .- (X .<= q)) ./ pdf(D, q)
    levels = (
        [0.1, 0.5, 0.9], [0.1 0.25 0.5; 0.6 0.75 0.9], reshape([0.1, 0.2, 0.3, 0.4], 2, 1, 2),
        0.1:0.4:0.9, fill(0.5),
    )
    for ps in levels
        v, ψ = value_and_influence(quantile, X, ps; density = D)
        @test v == quantile(X, ps)
        @test axes(ψ) == axes(v)
        @test all(I -> ψ[I] ≈ expected(ps[I], v[I]), CartesianIndices(v))
    end
    v, ψ = value_and_influence(quantile, X, (0.25, 0.75); density = D)
    @test v === quantile(X, (0.25, 0.75))
    @test ψ isa Tuple{Vector{Float64}, Vector{Float64}}
    @test ψ[2] ≈ expected(0.75, v[2])
    @test influence(quantile, X, ()) === ()

    ps = [0.1 0.25 0.5; 0.6 0.75 0.9]
    ψ = influence(quantile, X, ps)
    @test all(I -> ψ[I] == influence(quantile, X, ps[I]), CartesianIndices(ps))
    fits = Ref(0)                                                       # one fit serves every level
    influence(quantile, X, ps; density = CountingDensity(fits))
    @test fits[] == 1
    v, ψ = value_and_influence(quantile, X, ps; alpha = 0.0, beta = 0.0, density = D)
    @test v == quantile(X, ps; alpha = 0.0, beta = 0.0)
    @test ψ[2, 3] ≈ expected(0.9, v[2, 3])
    @test influence(quantile, sort(X), ps; sorted = true)[1, 2] ≈ influence(quantile, X, ps)[1, 2][sortperm(X)]

    po = OffsetArray([0.1, 0.9], -3)                                    # the levels' axes on the outside
    @test axes(influence(quantile, X, po)) == axes(po)
    pd = DimArray([0.1, 0.9], Dim{:level}([0.1, 0.9]))
    xd = DimArray(X, Dim{:draw}(1:N))                                   # the data's axes on the inside
    ψd = influence(quantile, xd, pd)
    @test ψd isa DimArray
    @test dims(ψd) == dims(pd)
    @test dims(ψd[1]) == dims(xd)
    @test dims(influence(quantile, xd, [0.1, 0.9])[1]) == dims(xd)

    @test isempty(influence(quantile, X, Float64[]))
    for ps in ([0.5, 1.0], (0.0, 0.5), [-0.1 0.5])
        @test_throws DomainError influence(quantile, X, ps)
    end
    @inferred value_and_influence(quantile, X, [0.1, 0.9])
    @inferred value_and_influence(quantile, X, ps)
    @inferred value_and_influence(quantile, X, (0.25, 0.75))
end

@testset "quantile(f, x, p)" begin
    f = t -> abs(t - 5)                                                 # not monotone
    if !hasmethod(quantile, Tuple{Function, Vector{Float64}, Float64})
        @test_throws MethodError influence(quantile, f, X, 0.3)         # older Statistics
    else
        fdensity = t -> pdf(D, 5 + t) + pdf(D, 5 - t)                   # the density of f(x)
        v, ψ = value_and_influence(quantile, f, X, 0.3; density = fdensity)
        @test v == quantile(f, X, 0.3)
        @test ψ ≈ (0.3 .- (f.(X) .<= v)) ./ fdensity(v)
        for p in (0.3, [0.1 0.25 0.5; 0.6 0.75 0.9], (0.25, 0.75))
            v, ψ = value_and_influence(quantile, f, X, p)
            @test v == quantile(f, X, p)
            @test ψ == influence(quantile, f.(X), p)
        end
        v, ψ = value_and_influence(quantile, f, X, [0.1, 0.9]; alpha = 0.0, beta = 0.0)
        @test v == quantile(f, X, [0.1, 0.9]; alpha = 0.0, beta = 0.0)
        @test ψ == influence(quantile, f.(X), [0.1, 0.9]; alpha = 0.0, beta = 0.0)
        perm = sortperm(f.(X))                                          # `sorted` means f.(x) is sorted
        @test influence(quantile, f, X[perm], 0.3; sorted = true) ≈ influence(quantile, f, X, 0.3)[perm]
        fits = Ref(0)                                                   # one fit serves every level
        influence(quantile, f, X, [0.1, 0.9]; density = CountingDensity(fits))
        @test fits[] == 1
        xd = DimArray(X, Dim{:draw}(1:N))
        @test dims(influence(quantile, f, xd, 0.3)) == dims(xd)
        @test dims(influence(quantile, f, xd, [0.1, 0.9])[1]) == dims(xd)
        @test eltype(influence(quantile, f, Float32.(X), 0.3f0)) === Float32
        @inferred value_and_influence(quantile, f, X, 0.3)
        @inferred value_and_influence(quantile, f, X, [0.1, 0.9])
        for p in (0.0, [0.5, 1.0])
            @test_throws DomainError influence(quantile, f, X, p)
        end
        @test_throws MethodError influence(quantile, f, X, 0.3; dims = 1)   # as in Statistics
    end
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
    xd = DimArray(A, (Dim{:draw}(1:9), Dim{:chain}(1:4)))               # other `dims` forms that `median` takes
    for dims in (:chain, Dim{:chain}())
        @test parent(influence(median, xd; dims)) ≈ influence(median, A; dims = 2)
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

@testset "median(f, x)" begin
    f = t -> abs(t - 5)                                                 # not monotone
    if !hasmethod(median, Tuple{Function, Vector{Float64}})
        @test_throws MethodError influence(median, f, X)                # older Statistics
    else
        for x in (X, X[1:35])                                           # even and odd n
            v, ψ = value_and_influence(median, f, x)
            @test v == median(f, x)
            @test ψ == influence(median, f.(x))
        end
        fdensity = t -> pdf(D, 5 + t) + pdf(D, 5 - t)                   # the density of f(x)
        v, ψ = value_and_influence(median, f, X; density = fdensity)
        @test ψ ≈ (0.5 .- (f.(X) .<= v)) ./ fdensity(v)
        A = reshape(X, 9, 4)                                            # all elements, of any shape
        v, ψ = value_and_influence(median, f, A)
        @test v == median(f, A)
        @test size(ψ) == size(A)
        @test vec(ψ) ≈ influence(median, f, X)
        xd = DimArray(A, (Dim{:draw}(1:9), Dim{:chain}(1:4)))
        @test dims(influence(median, f, xd)) == dims(xd)
        @test eltype(influence(median, f, Float32.(X))) === Float32
        @inferred value_and_influence(median, f, X)
        @test_throws MethodError influence(median, f, A; dims = 1)      # as in Statistics
    end
end
