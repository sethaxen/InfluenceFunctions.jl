using Test, InfluenceFunctions
using InfluenceFunctions: fit_density, _density_at_quantile

# @test Base.get_extension(InfluenceFunctions, :StatsBaseExt) === nothing

x, w = [1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 5.0]
g = fit_density(QuantileSpacing(; bandwidth = 0.1), x, w)
@test g(2.0) ≈ 0.125
@test _density_at_quantile(g, NaN, 0.25) ≈ 1 / 7
