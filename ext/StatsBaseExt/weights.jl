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
