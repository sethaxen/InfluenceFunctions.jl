###############
# varcorrection
###############

# The bias factor `corrected` applies to a weighted mean of squared deviations, as StatsBase's
# `varcorrection` sets it for each type of weights, in type `T`; the number of observations is
# not needed. Like the Bessel factor, it is held fixed in the influence values, even for
# analytic weights, where it depends on how the weights are spread. StatsBase rejects
# `corrected` for plain `Weights` before this is reached.
function InfluenceFunctions._bessel_factor(
        ::Type{T}, ::Integer, w::StatsBase.AbstractWeights, corrected::Bool
    ) where {T}
    return corrected ? _bias_factor(T, w) : one(T)
end
_bias_factor(::Type{T}, w::StatsBase.UnitWeights) where {T} = T(length(w)) / T(length(w) - 1)
function _bias_factor(::Type{T}, w::StatsBase.FrequencyWeights) where {T}
    W = T(sum(w))
    return W / (W - 1)
end
function _bias_factor(::Type{T}, w::StatsBase.ProbabilityWeights) where {T}
    n = count(!iszero, w)
    return T(n) / T(n - 1)
end
function _bias_factor(::Type{T}, w::StatsBase.AnalyticWeights) where {T}
    W = T(sum(w))
    return inv(1 - sum(v -> abs2(T(v) / W), w))
end

###############
# weighted mean
###############

function InfluenceFunctions.value_and_influence(
        ::typeof(Statistics.mean), x::AbstractArray, w::StatsBase.AbstractWeights; dims = :
    )
    μ = Statistics.mean(x, w; dims)
    return μ, x .- μ
end

###################
# weighted quantile
###################

function InfluenceFunctions.value_and_influence(
        ::typeof(Statistics.quantile), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights, p;
        density = InfluenceFunctions.QuantileSpacing(),
    )
    foreach(InfluenceFunctions._check_level, p)
    q = Statistics.quantile(x, w, p)
    return q, InfluenceFunctions._quantile_influences(density, x, q, p, w)
end

#################
# weighted median
#################

function InfluenceFunctions.value_and_influence(
        ::typeof(Statistics.median), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights;
        density = InfluenceFunctions.QuantileSpacing(),
    )
    m = Statistics.median(x, w)
    return m, InfluenceFunctions._quantile_influences(density, x, m, oftype(float(m), 1 // 2), w)
end

##########################
# weighted QuantileSpacing
##########################

function InfluenceFunctions.fit_density(est::InfluenceFunctions.QuantileSpacing, x, w::StatsBase.UnitWeights)
    length(x) == length(w) || throw(DimensionMismatch("data and weights must have the same length"))
    return InfluenceFunctions.fit_density(est, x)
end

# A frequency weight of c is c copies of mass 1/N, so its squared mass is c/N².
InfluenceFunctions._squared_mass(::StatsBase.FrequencyWeights, m, W) = m / W

# Keep the first and last knot from each group of repeated observations.
function InfluenceFunctions._spacing_knots(::StatsBase.FrequencyWeights, xs, ws, cmass, W)
    all(isinteger, ws) || throw(ArgumentError("FrequencyWeights must be numerically integers"))
    M = eltype(cmass)
    V = eltype(xs)
    n = length(xs)
    levels = sizehint!(M[], 2n)
    knots = sizehint!(V[], 2n)
    for (j, (xj, wj)) in enumerate(zip(xs, ws))
        push!(levels, cmass[j] + 1 / 2W)
        push!(knots, xj)
        if wj > 1
            push!(levels, cmass[j + 1] - 1 / 2W)
            push!(knots, xj)
        end
    end
    return levels, knots
end
