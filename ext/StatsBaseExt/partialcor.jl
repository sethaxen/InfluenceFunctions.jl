############
# partialcor
############

# ρ is the correlation of the residuals of `x` and `y` regressed on `[1 Z]`. The residuals are
# orthogonal to `[1 Z]`, so the first-order change of the coefficients leaves their covariance
# unchanged, and ψ is the influence of the correlation of the residuals. If ρ is NaN, as from a
# NaN in the data or collinear covariates, for which the regression has no unique solution, then
# so is ψ.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.partialcor), x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
        Z::AbstractVecOrMat{<:Real}
    )
    ρ = StatsBase.partialcor(x, y, Z)
    isnan(ρ) && return ρ, fill!(similar(x, typeof(ρ)), ρ)
    μs, S = _mean_and_cov(Z)
    ex = _residuals(x, Z, μs, S)
    ey = _residuals(y, Z, μs, S)
    return ρ, InfluenceFunctions.influence(Statistics.cor, ex, ey)
end

# Residuals of `v` regressed on the columns of `Z` and an intercept, given their means `μs` and
# uncorrected covariance matrix `S`.
function _residuals(v, Z, μs, S)
    μ = Statistics.mean(v)
    β = S \ [InfluenceFunctions._mean_cross(view(Z, :, j), μs[j], v, μ) for j in axes(Z, 2)]
    e = similar(v, eltype(β))    # with the precision of `Z` too
    e .= v .- μ
    for j in axes(Z, 2)
        e .-= β[j] .* (view(Z, :, j) .- μs[j])
    end
    return e
end
