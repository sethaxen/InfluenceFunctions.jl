using DensityInterface: funcdensity
using InfluenceFunctions: fit_density, HallSheatherBandwidth, _bandwidth, _density_at_quantile

@testset "fit_density: known and user-defined densities" begin
    @test fit_density(Normal(1, 2), X)(0.3) ≈ pdf(Normal(1, 2), 0.3)
    @test fit_density(funcdensity(t -> 2t), X)(0.25) == 0.5
    @test fit_density(t -> 3.0, X)(1.0) == 3.0
    @test fit_density(Returns(0.3), X)(1.0) == 0.3
    @test _density_at_quantile(Returns(0.3), 1.0, 0.5) == 0.3          # default: evaluate at q̂
    @test fit_density(ConstantDensity(0.4), X)(1.0) == 0.4              # user type, no supertype
    @test_throws ArgumentError fit_density(42, X)
    @test_throws ArgumentError fit_density(:foo, X)
    VERSION >= v"1.11" && @test Base.ispublic(InfluenceFunctions, :fit_density)

    w = rand(StableRNG(6), N)                                            # known densities ignore weights
    @test fit_density(Normal(1, 2), X, w)(0.3) ≈ pdf(Normal(1, 2), 0.3)
    @test fit_density(t -> 3.0, X, w)(1.0) == 3.0
    @test_throws ArgumentError fit_density(ConstantDensity(0.4), X, w)   # no weighted method: not ignored
    @test_throws ArgumentError fit_density(42, X, w)
end

@testset "HallSheatherBandwidth" begin
    @test _bandwidth(HallSheatherBandwidth(), 0.5, 1000) ≈ 0.09715590262051796 rtol = 1.0e-12
    @test _bandwidth(HallSheatherBandwidth(), 0.9, 1000) ≈ 0.03459946254739375 rtol = 1.0e-12
    @test _bandwidth(HallSheatherBandwidth(), 0.05, 10_000) ≈ 0.009851244664485101 rtol = 1.0e-12
    @test_throws DomainError HallSheatherBandwidth(; alpha = 1.5)
    @test _bandwidth(HallSheatherBandwidth(; alpha = 1.0e-20), 0.5, 1000) == Inf   # 1 - α/2 rounds to 1
    @test _bandwidth(HallSheatherBandwidth(), 0.3f0, 1000) isa Float32     # follows the level's type
end

@testset "QuantileSpacing" begin
    @test_throws DomainError QuantileSpacing(; bandwidth = 0.7)
    @test_throws DomainError QuantileSpacing(; bandwidth = -0.1)
    @test_throws DomainError QuantileSpacing(0.7)                       # positional constructor too

    g = fit_density(QuantileSpacing(; bandwidth = 0.05), collect(1.0:100.0))
    @test g(50.0) ≈ 1 / 100     # interior: level 0.495, 0.1 / (55.0 - 45.0)
    @test g(1.0) ≈ 1 / 50       # level 0.005, h halved to 0.003125; the EQF's flat end doubles it (documented)
    @test g(100.0) ≈ 1 / 50
    @test g(0.5) == g(100.5) == 0                                       # outside the data
    @test _density_at_quantile(g, 50.5, 0.5) ≈ 1 / 100                  # nominal level
    @test isfinite(_density_at_quantile(fit_density(QuantileSpacing(), X), 0.0, 0.001))   # level clamped
    @test fit_density(QuantileSpacing(), fill(2.0, 10))(2.0) == Inf      # point mass
    ti = rand(StableRNG(4), 1:5, 1000)                                   # ties
    @test 0 < fit_density(QuantileSpacing(), ti)(median(ti)) < Inf
    atom = [zeros(300); rand(StableRNG(5), 700) .+ 0.5]                  # 30% atom at the minimum
    g1, g2 = fit_density(QuantileSpacing(), atom), fit_density(QuantileSpacing(), -atom)
    @test _density_at_quantile(g1, 0.0, 0.2) == _density_at_quantile(g2, 0.0, 0.8) == Inf
    @test g1(0.0) == g2(0.0) == Inf                                       # mid-rank levels 0.15 and 0.85
    for x in (X, Float32.(X))                                            # an infinite bandwidth still halves
        tiny = QuantileSpacing(; bandwidth = HallSheatherBandwidth(; alpha = 1.0e-20))
        @test 0 < _density_at_quantile(fit_density(tiny, x), 5.0, one(eltype(x)) / 2) < Inf
    end
    b = HallSheatherBandwidth(; alpha = 1.0e-3)                          # a finite h > 1 halves as usual
    h = _bandwidth(b, 0.5, 2)
    @test h > 1
    @test _density_at_quantile(fit_density(QuantileSpacing(; bandwidth = b), [1.0, 2.0]), 1.5, 0.5) ≈ h / 2
    @test isnan(fit_density(QuantileSpacing(), [1.0, NaN, 3.0])(2.0))
    gx = @inferred fit_density(QuantileSpacing(), X)
    @inferred gx(5.0)
    @inferred _density_at_quantile(gx, 5.0, 0.5)
    @test @inferred(_density_at_quantile(gx, 5.0, 0.3f0)) isa Float64    # as wide as the data
    @test _density_at_quantile(fit_density(QuantileSpacing(), big.(X)), big(5.0), 0.3) isa BigFloat
    g32 = fit_density(QuantileSpacing(), Float32.(X))                    # Float32 data stays Float32
    @test @inferred(g32(5.0f0)) isa Float32
    @test @inferred(_density_at_quantile(g32, 5.0f0, 0.3f0)) isa Float32
    x16 = Float16[1, 2, 3, 5, 9]                                         # Float16 `norminvcdf` is inaccurate
    @test fit_density(QuantileSpacing(), x16)(x16[4]) isa Float16
    @test fit_density(QuantileSpacing(), x16)(x16[4]) ≈ fit_density(QuantileSpacing(), Float64.(x16))(5.0) rtol = 1.0e-2
    y16 = Float16.(rand(StableRNG(3), Normal(), 5000))                   # 1 - 1/(2n) rounds to 1 in Float16
    @test isfinite(fit_density(QuantileSpacing(), y16)(maximum(y16)))

    # accuracy against known densities
    for d in (Gamma(3, 2), Exponential(), Normal())
        x = rand(StableRNG(1), d, 100_000)
        g = fit_density(QuantileSpacing(), x)
        for p in (0.05, 0.5, 0.9)
            q = quantile(d, p)
            @test g(q) ≈ pdf(d, q) rtol = 0.1                                          # prototype: ≤ 0.035
            @test _density_at_quantile(g, quantile(x, p), p) ≈ pdf(d, q) rtol = 0.1    # prototype: ≤ 0.041
        end
    end
end

@testset "QuantileSpacing with plain weights" begin
    # Test in a fresh process: the other test dependencies may load StatsBase indirectly.
    project = dirname(@__DIR__)
    script = joinpath(@__DIR__, "core_weighted_density.jl")
    @test success(`$(Base.julia_cmd()) --project=$project $script`)
end
