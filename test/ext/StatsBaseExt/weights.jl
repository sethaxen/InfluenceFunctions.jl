using StatsBase
using InfluenceFunctions: fit_density, _density_at_quantile, _bandwidth, HallSheatherBandwidth

@testset "weighted mean" begin
    for W in (fweights, pweights, aweights, weights)
        check_weighted_against_ad(u -> mean(X, W(u)), value_and_influence(mean, X, W(WV)), WV)
    end
    # Influence values are scaled so that their standard error under the weights is that of the
    # estimate, wherever StatsBase defines one.
    for w in (fweights(WV), pweights(WV), uweights(N))
        @test sem(influence(mean, X, w), w) ≈ sem(X, w)
    end
    # Frequency weights count repeats of an observation.
    @test inverse_rle(influence(mean, X, fweights(WV)), WV) ≈ influence(mean, inverse_rle(X, WV))
    @test value_and_influence(mean, X, uweights(N)) == value_and_influence(mean, X)
    A = reshape(X, 9, 4)
    for dims in (1, 2)
        check_dims(mean, A, pweights(WV[1:size(A, dims)]); dims)
    end
    check_shapes(mean, X, fweights(WV))
    @test_throws DimensionMismatch influence(mean, X, pweights(WV[1:5]))
    @test_throws MethodError influence(mean, X, pweights(WV); density = nothing)
end

@testset "weighted quantile" begin
    expected(p, q) = (p .- (X .<= q)) ./ pdf(D, q)
    for W in (fweights, pweights, aweights, weights)
        w = W(WV)
        v, ψ = value_and_influence(quantile, X, w, 0.9; density = D)
        @test v == quantile(X, w, 0.9)
        @test ψ ≈ expected(0.9, v)
        @test influence(quantile, X, w, 0.9; density = t -> pdf(D, t)) ≈ ψ
    end
    ps = [0.1, 0.5, 0.9]
    v, ψ = value_and_influence(quantile, X, pweights(WV), ps; density = D)
    @test v == quantile(X, pweights(WV), ps)
    @test ψ isa Vector{Vector{Float64}}
    @test all(i -> ψ[i] ≈ expected(ps[i], v[i]), eachindex(ps))
    fits = Ref(0)                                                       # one fit serves every level
    influence(quantile, X, pweights(WV), ps; density = CountingDensity(fits))
    @test fits[] == 1
    @test_throws ArgumentError influence(quantile, X, pweights(WV), 0.9; density = ConstantDensity(0.4))
    for p in (0.0, 1.0, [0.5, 1.0])
        @test_throws DomainError influence(quantile, X, pweights(WV), p)
    end
    @test_throws MethodError influence(quantile, X, pweights(WV), (0.25, 0.75))   # StatsBase: no tuples

    # The default density: weighted QuantileSpacing, whose tests are below
    Xe = inverse_rle(X, WV)                                             # frequency weights count repeats
    @test inverse_rle(influence(quantile, X, fweights(WV), 0.3), WV) ≈ influence(quantile, Xe, 0.3)
    @test inverse_rle.(influence(quantile, X, fweights(WV), [0.1, 0.9]), Ref(WV)) ≈
        influence(quantile, Xe, [0.1, 0.9])
    @test value_and_influence(quantile, X, uweights(N), 0.3) == value_and_influence(quantile, X, 0.3)
    @test influence(quantile, X, pweights(fill(2, N)), 0.3) ≈ influence(quantile, X, 0.3)
    w0 = [WV[1:(N - 6)]; zeros(Int, 6)]                                 # zero weights carry no mass
    keep = 1:(N - 6)
    v, ψ = value_and_influence(quantile, X, pweights(w0), 0.3)
    v1, ψ1 = value_and_influence(quantile, X[keep], pweights(w0[keep]), 0.3)
    @test v == v1
    @test ψ[keep] ≈ ψ1
    @test ψ ≈ (0.3 .- (X .<= v)) ./ (0.3 - (X[1] <= v)) .* ψ1[1]        # same density at dropped points
    check_shapes(quantile, X, pweights(WV), 0.3; vector_only = true)    # StatsBase takes only vectors
    check_shapes(quantile, X, fweights(WV), 0.3; vector_only = true)
    @inferred value_and_influence(quantile, X, pweights(WV), [0.1, 0.9])
    @test_throws ArgumentError influence(quantile, X, pweights(WV[1:5]), 0.3)
end

@testset "weighted median" begin
    v, ψ = value_and_influence(median, X, pweights(WV); density = D)
    @test v == median(X, pweights(WV))
    @test ψ ≈ (0.5 .- (X .<= v)) ./ pdf(D, v)
    @test influence(median, X, pweights(WV)) ≈ influence(quantile, X, pweights(WV), 0.5)
    @test inverse_rle(influence(median, X, fweights(WV)), WV) ≈ influence(median, inverse_rle(X, WV))
    check_shapes(median, X, pweights(WV); vector_only = true)
end

@testset "weighted QuantileSpacing" begin
    # Four rows with masses 1/8, 1/8, 1/8, 5/8: knots at levels 1/16, 3/16, 5/16, 11/16.
    x, w = [1.0, 2.0, 3.0, 4.0], pweights([1, 1, 1, 5])
    g = fit_density(QuantileSpacing(; bandwidth = 0.1), x, w)
    @test _density_at_quantile(g, NaN, 0.25) ≈ 0.2 / (3.1 - 1.7)        # Q(0.35) = 3.1, Q(0.15) = 1.7
    @test g(2.0) ≈ 0.2 / (2.8 - 1.2)                                     # mid-CDF level 3/16
    @test _density_at_quantile(g, NaN, 0.9) ≈ 0.2 / (4 - 3 - 0.275 / 0.375)   # clamped to the last knot
    @test g(0.5) == g(4.5) == 0                                          # outside the data
    g0 = fit_density(QuantileSpacing(; bandwidth = 0.1), [x; 10.0], pweights([1, 1, 1, 5, 0]))
    @test _density_at_quantile(g0, NaN, 0.25) ≈ _density_at_quantile(g, NaN, 0.25)   # zero weight: dropped
    @test g0(10.0) == 0

    # Bandwidth: Korn and Graubard's effective size τ(1 - τ) / Var(F̂(τ)). With masses 1/200 on
    # rows 1-50 and 3/200 on rows 51-100, at τ = 0.2 the squared mass below τ is 0.2/200 and in
    # total 500/40000, so n = 0.16 / (0.64 * 0.001 + 0.04 * 0.0115) = 1600/11. (Kish: 80.)
    x, w = collect(1.0:100.0), pweights([fill(1, 50); fill(3, 50)])
    h = _bandwidth(HallSheatherBandwidth(), 0.2, 1600 / 11)
    @test _density_at_quantile(fit_density(QuantileSpacing(), x, w), NaN, 0.2) ≈
        _density_at_quantile(fit_density(QuantileSpacing(; bandwidth = h), x, w), NaN, 0.2)

    # Equal weights give the unweighted fit, also where nτ is not an integer.
    gu, ge = fit_density(QuantileSpacing(), X), fit_density(QuantileSpacing(), X, pweights(fill(2.0, N)))
    for τ in (0.025, 0.3, 0.5, 0.9)
        @test _density_at_quantile(ge, NaN, τ) ≈ _density_at_quantile(gu, NaN, τ)
    end
    @test ge.(X) ≈ gu.(X)
    @test fit_density(QuantileSpacing(), X, fill(2.0, N)).(X) ≈ ge.(X)  # plain vectors can be fitted
    @test fit_density(QuantileSpacing(), X, uweights(N)).(X) == gu.(X)

    # Frequency weights give the fit to the expanded data.
    gf, gx = fit_density(QuantileSpacing(), X, fweights(WV)), fit_density(QuantileSpacing(), inverse_rle(X, WV))
    for τ in (0.025, 0.3, 0.5, 0.9)
        @test _density_at_quantile(gf, NaN, τ) ≈ _density_at_quantile(gx, NaN, τ)
    end
    @test gf.(X) ≈ gx.(X)
    # StatsBase accepts floating-point counts when they are numerically integers, but its
    # quantile rejects fractional FrequencyWeights rather than treating them as scaled masses.
    @test fit_density(QuantileSpacing(), [1.0, 2.0], fweights([1.0, 2.0]))(1.0) isa Float64
    @test_throws ArgumentError fit_density(QuantileSpacing(), [1.0, 2.0], fweights([0.5, 1.5]))

    @test fit_density(QuantileSpacing(), fill(2.0, 5), pweights(1:5))(2.0) == Inf   # point mass
    @test isnan(fit_density(QuantileSpacing(), [1.0, NaN, 3.0], pweights([1, 1, 1]))(2.0))
    @test_throws DimensionMismatch fit_density(QuantileSpacing(), X, pweights(WV[1:5]))
    gw = @inferred fit_density(QuantileSpacing(), X, pweights(WV))
    @inferred gw(5.0)
    @inferred _density_at_quantile(gw, 5.0, 0.5)
    @test fit_density(QuantileSpacing(), Float32.(X), pweights(WV))(5.0f0) isa Float32
    @test fit_density(QuantileSpacing(), Float32.(X), pweights(Float64.(WV)))(5.0f0) isa Float64   # as StatsBase

    # Importance sampling: the density of the target, not of the draws.
    F, G = Normal(1, 1), Normal(0, 2)
    xg = rand(StableRNG(7), G, 100_000)
    wg = pweights(pdf.(F, xg) ./ pdf.(G, xg))
    gi = fit_density(QuantileSpacing(), xg, wg)
    for p in (0.05, 0.5, 0.9)
        q = quantile(F, p)
        @test gi(q) ≈ pdf(F, q) rtol = 0.1
        @test _density_at_quantile(gi, quantile(xg, wg, p), p) ≈ pdf(F, q) rtol = 0.1
    end
end
