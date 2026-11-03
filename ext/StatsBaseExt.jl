module StatsBaseExt

using InfluenceFunctions: InfluenceFunctions
using Statistics: Statistics
using StatsBase: StatsBase

#########
# geomean
#########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.geomean), x::AbstractArray{<:Real},
    )
    m = Statistics.mean(log, x)
    G = exp(m)
    return G, G .* (log.(x) .- m)
end

##########
# harmmean
##########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.harmmean), x::AbstractArray{<:Real},
    )
    H = inv(Statistics.mean(inv, x))
    return H, H .* (1 .- H ./ x)
end

#########
# genmean
#########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.genmean), x::AbstractArray{<:Real}, p::Real,
    )
    iszero(p) && return InfluenceFunctions.value_and_influence(StatsBase.geomean, x)
    S = Statistics.mean(t -> float(t)^p, x)
    M = S^inv(p)    # as StatsBase.genmean computes it
    return M, (M / p) .* ((float.(x) .^ p) ./ S .- 1)
end

########
# moment
########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.moment), x::AbstractArray{<:Real}, k,
    )
    return _moment_value_and_influence(x, k, Statistics.mean(x), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.moment), x::AbstractArray{<:Real}, k, m::Real,
    )
    return _moment_value_and_influence(x, k, m, false)
end

function _moment_value_and_influence(x, k, c, center_est)
    mₖ = StatsBase.moment(x, k, c)
    mₖ₋₁ = center_est ? StatsBase.moment(x, k - 1, c) : zero(mₖ)
    return mₖ, _moment_influence.(x .- c, k, mₖ, mₖ₋₁, center_est)
end

# Influence of the k-th moment about a center `c` at the deviation `d = t - c`, given the
# moments mₖ and mₖ₋₁ about `c`. When `c` is the estimated mean, the last term accounts for
# estimating it.
_moment_influence(d, k, mₖ, mₖ₋₁, center_est) = d^k - mₖ - center_est * (k * mₖ₋₁ * d)

##########
# skewness
##########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.skewness), x::AbstractArray{<:Real}
    )
    return _skewness_value_and_influence(x, Statistics.mean(x), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.skewness), x::AbstractArray{<:Real}, m::Real
    )
    return _skewness_value_and_influence(x, m, false)
end

# g = m₃ / m₂^(3/2), so ψ = ψ₃ / m₂^(3/2) - (3g/2) ψ₂ / m₂.
function _skewness_value_and_influence(x, c, center_est)
    g = StatsBase.skewness(x, c)
    m₁, m₂, m₃ = (StatsBase.moment(x, k, c) for k in 1:3)
    ψ = broadcast(x) do t
        d = t - c
        ψ₂ = _moment_influence(d, 2, m₂, m₁, center_est)
        ψ₃ = _moment_influence(d, 3, m₃, m₂, center_est)
        return (ψ₃ / sqrt(m₂) - 3g / 2 * ψ₂) / m₂
    end
    return g, ψ
end

##########
# kurtosis
##########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.kurtosis), x::AbstractArray{<:Real}
    )
    return _kurtosis_value_and_influence(x, Statistics.mean(x), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.kurtosis), x::AbstractArray{<:Real}, m::Real
    )
    return _kurtosis_value_and_influence(x, m, false)
end

# Excess kurtosis κ = m₄ / m₂² - 3, so ψ = ψ₄ / m₂² - 2(κ + 3) ψ₂ / m₂.
function _kurtosis_value_and_influence(x, c, center_est)
    κ = StatsBase.kurtosis(x, c)
    m₁, m₂, m₃, m₄ = (StatsBase.moment(x, k, c) for k in 1:4)
    ψ = broadcast(x) do t
        d = t - c
        ψ₂ = _moment_influence(d, 2, m₂, m₁, center_est)
        ψ₄ = _moment_influence(d, 4, m₄, m₃, center_est)
        return (ψ₄ / m₂ - 2 * (κ + 3) * ψ₂) / m₂
    end
    return κ, ψ
end

###########
# variation
###########

# V = s / μ̂, so ψ = (ψ_std - V (x - μ̂)) / μ̂. StatsBase checks the keywords.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.variation), x::AbstractArray{<:Real}; kwargs...
    )
    V = StatsBase.variation(x; kwargs...)
    μ = Statistics.mean(x)
    _, ψ = InfluenceFunctions.value_and_influence(Statistics.std, x; kwargs...)
    ψ .= (ψ .- V .* (x .- μ)) ./ μ
    return V, ψ
end

# V = s / m, so ψ = ψ_stdm / m.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.variation), x::AbstractArray{<:Real}, m::Real; kwargs...
    )
    V = StatsBase.variation(x, m; kwargs...)
    _, ψ = InfluenceFunctions.value_and_influence(Statistics.stdm, x, m; kwargs...)
    ψ ./= m
    return V, ψ
end

end
