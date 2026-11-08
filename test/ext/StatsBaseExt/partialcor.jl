using StatsBase

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
    xn = copy(x)
    xn[2] = NaN
    # A NaN value, from a NaN, a constant covariate, or collinear covariates
    for (xs, Zs) in ((xn, Z), (x, fill(2.0, N)), (x, [Z fill(2.0, N)]), (x, [Z Z]))
        ρn, ψn = value_and_influence(partialcor, xs, y, Zs)
        @test isnan(ρn)
        @test all(isnan, ψn)
    end
    @inferred value_and_influence(partialcor, x, y, Z)
    @inferred value_and_influence(partialcor, x, y, vec(Z))
    @test_throws DimensionMismatch influence(partialcor, x, y[1:(end - 1)], Z)
end
