using StatsBase

skew(p, m) = wmean(p, (X .- m) .^ 3) / wmean(p, (X .- m) .^ 2)^1.5
kurt(p, m) = wmean(p, (X .- m) .^ 4) / wmean(p, (X .- m) .^ 2)^2 - 3

@testset "weighted $f" for (f, scale) in ((var, identity), (std, sqrt))
    for W in (fweights, pweights, aweights, weights), corrected in (false, true), c in (nothing, C)
        # Analytic weights' correction depends on how the weights are spread (see below), and
        # StatsBase rejects `corrected` with plain `Weights`.
        corrected && (W === aweights || W === weights) && continue
        G = u -> f(X, W(u); corrected, mean = c)
        check_weighted_against_ad(G, value_and_influence(f, X, W(WV); corrected, mean = c), WV)
    end
    # The analytic weights' correction is held fixed, as Bessel's is.
    b = sum(WV) / (sum(WV) - sum(abs2, WV) / sum(WV))
    aw = aweights(WV)
    @test influence(f, X, aw; corrected = true) ≈ scale(b) .* influence(f, X, aw; corrected = false)
    ψ = influence(f, X, fweights(WV); corrected = true)    # repeats of an observation
    @test inverse_rle(ψ, WV) ≈ influence(f, inverse_rle(X, WV); corrected = true)
    w0 = [zeros(Int, 3); WV[4:end]]                         # zero weights leave the others alone
    for W in (fweights, pweights, aweights)
        v0, ψ0 = value_and_influence(f, X, W(w0); corrected = true)
        v1, ψ1 = value_and_influence(f, X[4:end], W(w0[4:end]); corrected = true)
        @test v0 ≈ v1
        @test ψ0[4:end] ≈ ψ1
    end
    # No spread among the observations with weight: moving mass onto the one without changes
    # the variance at a finite rate, so the standard deviation (its root) at an infinite one.
    x0, w0 = [2.0, 2.0, 2.0, 5.0], [1, 2, 3, 0]
    for W in (fweights, pweights, aweights)
        v, ψ = value_and_influence(f, x0, W(w0); corrected = false)
        @test iszero(v)
        @test ψ == [0, 0, 0, f === var ? 9 : Inf]
    end
    @test value_and_influence(f, X, uweights(N); corrected = true) == value_and_influence(f, X; corrected = true)
    w = pweights(WV)
    @test (@test_deprecated r"default to corrected=true" influence(f, X, w)) == influence(f, X, w; corrected = false)
    A = reshape(X, 9, 4)
    for dim in (1, 2), corrected in (false, true)
        wd = pweights(WV[1:size(A, dim)])
        check_weighted_dims(f, A, wd, dim; corrected)
        c = mean(A, wd; dims = dim) .+ 1                   # a given center
        @test influence(f, A, wd, dim; corrected, mean = c) ≈ influence(f, A .- c, wd, dim; corrected, mean = 0)
    end
    check_shapes(f, X, fweights(WV); corrected = true)
    @inferred value_and_influence(f, A, w[1:9], 1; corrected = true)
    @test_throws ArgumentError influence(f, X, weights(WV); corrected = true)   # as in StatsBase
    @test_throws DimensionMismatch influence(f, X, w[1:5]; corrected = true)
    @test_throws MethodError influence(f, A, w[1:9]; dims = 1)   # StatsBase takes it positionally
    @test_throws MethodError influence(f, X, w; density = nothing)
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

@testset "weighted $f" for (f, g) in ((mean_and_var, var), (mean_and_std, std))
    A = reshape(X, 9, 4)
    for corrected in (true, false)
        w = pweights(WV)
        v, ψ = value_and_influence(f, X, w; corrected)
        @test v == f(X, w; corrected)
        @test ψ == (influence(mean, X, w), influence(g, X, w; corrected))
        for dim in (1, 2)
            wd = pweights(WV[1:size(A, dim)])
            v, ψ = value_and_influence(f, A, wd, dim; corrected)
            @test v == f(A, wd, dim; corrected)
            @test ψ == (influence(mean, A, wd; dims = dim), influence(g, A, wd, dim; corrected))
        end
    end
    w = fweights(WV)
    @test (@test_deprecated r"default to corrected=true" value_and_influence(f, X, w)) == value_and_influence(f, X, w; corrected = false)
    @inferred value_and_influence(f, X, w; corrected = true)
    @inferred value_and_influence(f, A, w[1:9], 1; corrected = true)
    @test_throws ArgumentError influence(f, X, weights(WV); corrected = true)   # as in StatsBase
end

@testset "delta method" begin
    v, ψ = value_and_influence(mean_and_std, X)
    g = ForwardDiff.gradient(w -> w[2] / w[1], collect(v))
    @test g[1] .* ψ[1] .+ g[2] .* ψ[2] ≈ influence(variation, X)
    v, ψ = value_and_influence(quantile, X, (0.25, 0.75))
    @test ψ[2] .- ψ[1] ≈ influence(iqr, X)
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

@testset "weighted moment" begin
    for W in (fweights, pweights, aweights, weights), k in 2:5
        check_weighted_against_ad(u -> moment(X, k, W(u)), value_and_influence(moment, X, k, W(WV)), WV)
        check_weighted_against_ad(u -> moment(X, k, W(u), C), value_and_influence(moment, X, k, W(WV), C), WV)
    end
    @test inverse_rle(influence(moment, X, 3, fweights(WV)), WV) ≈ influence(moment, inverse_rle(X, WV), 3)
    @test value_and_influence(moment, X, 3, uweights(N)) == value_and_influence(moment, X, 3)
    @test value_and_influence(moment, X, 3, uweights(N), C) == value_and_influence(moment, X, 3, C)
    check_shapes(moment, X, 3, fweights(WV))
    check_shapes(moment, X, 3, fweights(WV), C)
    @test_throws DimensionMismatch influence(moment, X, 3, pweights(WV[1:5]))
end

@testset "skewness" begin
    check_against_ad(p -> skew(p, wmean(p, X)), value_and_influence(skewness, X), skewness(X))
    check_against_ad(p -> skew(p, C), value_and_influence(skewness, X, C), skewness(X, C))
    check_shapes(skewness, X)
    check_shapes(skewness, X, C)
    @test all(isnan, influence(skewness, fill(2.0, 12)))
end

@testset "weighted skewness" begin
    for W in (fweights, pweights, aweights, weights)
        check_weighted_against_ad(u -> skewness(X, W(u)), value_and_influence(skewness, X, W(WV)), WV)
        check_weighted_against_ad(u -> skewness(X, W(u), C), value_and_influence(skewness, X, W(WV), C), WV)
    end
    @test value_and_influence(skewness, X, uweights(N)) == value_and_influence(skewness, X)
    check_shapes(skewness, X, fweights(WV))
    check_shapes(skewness, X, fweights(WV), C)
end

@testset "kurtosis" begin
    check_against_ad(p -> kurt(p, wmean(p, X)), value_and_influence(kurtosis, X), kurtosis(X))
    check_against_ad(p -> kurt(p, C), value_and_influence(kurtosis, X, C), kurtosis(X, C))
    check_shapes(kurtosis, X)
    check_shapes(kurtosis, X, C)
end

@testset "weighted kurtosis" begin
    for W in (fweights, pweights, aweights, weights)
        check_weighted_against_ad(u -> kurtosis(X, W(u)), value_and_influence(kurtosis, X, W(WV)), WV)
        check_weighted_against_ad(u -> kurtosis(X, W(u), C), value_and_influence(kurtosis, X, W(WV), C), WV)
    end
    @test value_and_influence(kurtosis, X, uweights(N)) == value_and_influence(kurtosis, X)
    check_shapes(kurtosis, X, fweights(WV))
    check_shapes(kurtosis, X, fweights(WV), C)
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
end

@testset "weighted cumulant" begin
    for W in (fweights, pweights, aweights, weights), k in 1:6
        check_weighted_against_ad(u -> cumulant(X, k, W(u)), value_and_influence(cumulant, X, k, W(WV)), WV)
        G = u -> cumulant(X, k, W(u), C + zero(eltype(u)))
        check_weighted_against_ad(G, value_and_influence(cumulant, X, k, W(WV), C), WV)
    end
    w = pweights(WV)
    v, ψ = value_and_influence(cumulant, X, 2:2:6, w)
    @test v ≈ cumulant(X, 2:2:6, w)
    @test ψ[2] == influence(cumulant, X, 4, w)
    @test influence(cumulant, X, 1:3, w, C)[3] == influence(cumulant, X, 3, w, C)
    @test value_and_influence(cumulant, X, 4, uweights(N)) == value_and_influence(cumulant, X, 4)
    check_shapes(cumulant, X, 4, fweights(WV))
    check_shapes(cumulant, X, 4, fweights(WV), C)
    @inferred value_and_influence(cumulant, X, 1:4, w)
    @test_throws ArgumentError influence(cumulant, X, 0, w)
    @test_throws DimensionMismatch influence(cumulant, X, 2, w[1:5])
end
