######################
# weighted var and std
######################

# Here and below, StatsBase takes `corrected = nothing` as false, with a deprecation warning.

function InfluenceFunctions.value_and_influence(
        ::typeof(Statistics.var), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights;
        mean = nothing, corrected = nothing
    )
    c = @something(mean, Statistics.mean(x, w))
    v = Statistics.var(x, w; mean = c, corrected)
    return v, InfluenceFunctions._var_influence(x, c, v, something(corrected, false), w)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(Statistics.var), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights, dim::Int;
        mean = nothing, corrected = nothing
    )
    c = @something(mean, Statistics.mean(x, w; dims = dim))
    v = Statistics.var(x, w, dim; mean = c, corrected)
    return v, InfluenceFunctions._var_influence(x, c, v, something(corrected, false), w)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(Statistics.std), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights;
        mean = nothing, corrected = nothing
    )
    c = @something(mean, Statistics.mean(x, w))
    s = Statistics.std(x, w; mean = c, corrected)
    return s, InfluenceFunctions._std_influence(x, c, s, something(corrected, false), w)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(Statistics.std), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights, dim::Int;
        mean = nothing, corrected = nothing
    )
    c = @something(mean, Statistics.mean(x, w; dims = dim))
    s = Statistics.std(x, w, dim; mean = c, corrected)
    return s, InfluenceFunctions._std_influence(x, c, s, something(corrected, false), w)
end

############################
# mean_and_var, mean_and_std
############################

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_var), x::AbstractArray{<:Real}; corrected::Bool = true
    )
    return _mean_and_value_and_influence(Statistics.var, x, :, corrected)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_std), x::AbstractArray{<:Real}; corrected::Bool = true
    )
    return _mean_and_value_and_influence(Statistics.std, x, :, corrected)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_var), x::AbstractArray{<:Real}, dim::Int; corrected::Bool = true
    )
    return _mean_and_value_and_influence(Statistics.var, x, dim, corrected)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_std), x::AbstractArray{<:Real}, dim::Int; corrected::Bool = true
    )
    return _mean_and_value_and_influence(Statistics.std, x, dim, corrected)
end

function _mean_and_value_and_influence(f, x, dims, corrected)
    μ, ψμ = InfluenceFunctions.value_and_influence(Statistics.mean, x; dims)
    v, ψ = InfluenceFunctions.value_and_influence(f, x; corrected, mean = μ, dims)
    return (μ, v), (ψμ, ψ)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_var), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights;
        corrected = nothing
    )
    μ, v = StatsBase.mean_and_var(x, w; corrected)
    ψ = InfluenceFunctions._var_influence(x, μ, v, something(corrected, false), w)
    return (μ, v), (x .- μ, ψ)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_std), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights;
        corrected = nothing
    )
    μ, s = StatsBase.mean_and_std(x, w; corrected)
    ψ = InfluenceFunctions._std_influence(x, μ, s, something(corrected, false), w)
    return (μ, s), (x .- μ, ψ)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_var), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights,
        dim::Int; corrected = nothing
    )
    μ, v = StatsBase.mean_and_var(x, w, dim; corrected)
    ψ = InfluenceFunctions._var_influence(x, μ, v, something(corrected, false), w)
    return (μ, v), (x .- μ, ψ)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_std), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights,
        dim::Int; corrected = nothing
    )
    μ, s = StatsBase.mean_and_std(x, w, dim; corrected)
    ψ = InfluenceFunctions._std_influence(x, μ, s, something(corrected, false), w)
    return (μ, s), (x .- μ, ψ)
end

########
# moment
########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.moment), x::AbstractArray{<:Real}, k,
    )
    return _moment_value_and_influence(x, k, (), Statistics.mean(x), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.moment), x::AbstractArray{<:Real}, k, m::Real,
    )
    return _moment_value_and_influence(x, k, (), m, false)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.moment), x::AbstractArray{<:Real}, k, w::StatsBase.AbstractWeights,
    )
    return _moment_value_and_influence(x, k, (w,), Statistics.mean(x, w), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.moment), x::AbstractArray{<:Real}, k, w::StatsBase.AbstractWeights, m::Real,
    )
    return _moment_value_and_influence(x, k, (w,), m, false)
end

# In this and the helpers below, `w` is `()` or a 1-tuple of weights, splatted into StatsBase's
# calls. Weights change only the moments: the influence functions are those of the weighted
# empirical distribution.
function _moment_value_and_influence(x, k, w, c, center_est)
    mₖ = StatsBase.moment(x, k, w..., c)
    mₖ₋₁ = center_est ? StatsBase.moment(x, k - 1, w..., c) : zero(mₖ)
    ψ = broadcast(x) do t
        d = t - c
        return _moment_influence(d^k, d, k, mₖ, mₖ₋₁, center_est)
    end
    return mₖ, ψ
end

# Influence of the k-th moment about a center `c` at the deviation `d = t - c`, given
# `dᵏ = d^k` and the moments mₖ and mₖ₋₁ about `c`. When `c` is the estimated mean, the last
# term accounts for estimating it. `dᵏ` is an argument so that callers with a known `k` can
# multiply: `d^k` with a non-literal `k` is many times slower.
_moment_influence(dᵏ, d, k, mₖ, mₖ₋₁, center_est) = dᵏ - mₖ - center_est * (k * mₖ₋₁ * d)

##########
# skewness
##########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.skewness), x::AbstractArray{<:Real}
    )
    return _skewness_value_and_influence(x, (), Statistics.mean(x), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.skewness), x::AbstractArray{<:Real}, m::Real
    )
    return _skewness_value_and_influence(x, (), m, false)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.skewness), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights
    )
    return _skewness_value_and_influence(x, (w,), Statistics.mean(x, w), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.skewness), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights, m::Real
    )
    return _skewness_value_and_influence(x, (w,), m, false)
end

# g = m₃ / m₂^(3/2), so ψ = ψ₃ / m₂^(3/2) - (3g/2) ψ₂ / m₂.
function _skewness_value_and_influence(x, w, c, center_est)
    g = StatsBase.skewness(x, w..., c)
    m₂, m₃ = (StatsBase.moment(x, k, w..., c) for k in 2:3)
    ψ = broadcast(x) do t
        d = t - c
        d² = d * d
        ψ₂ = d² - m₂    # no center term: m₁ is 0 about the mean
        ψ₃ = _moment_influence(d² * d, d, 3, m₃, m₂, center_est)
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
    return _kurtosis_value_and_influence(x, (), Statistics.mean(x), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.kurtosis), x::AbstractArray{<:Real}, m::Real
    )
    return _kurtosis_value_and_influence(x, (), m, false)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.kurtosis), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights
    )
    return _kurtosis_value_and_influence(x, (w,), Statistics.mean(x, w), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.kurtosis), x::AbstractArray{<:Real}, w::StatsBase.AbstractWeights, m::Real
    )
    return _kurtosis_value_and_influence(x, (w,), m, false)
end

# Excess kurtosis κ = m₄ / m₂² - 3, so ψ = ψ₄ / m₂² - 2(κ + 3) ψ₂ / m₂.
function _kurtosis_value_and_influence(x, w, c, center_est)
    κ = StatsBase.kurtosis(x, w..., c)
    m₂, m₃ = (StatsBase.moment(x, k, w..., c) for k in 2:3)
    m₄ = (κ + 3) * m₂^2
    ψ = broadcast(x) do t
        d = t - c
        d² = d * d
        ψ₂ = d² - m₂    # no center term: m₁ is 0 about the mean
        ψ₄ = _moment_influence(d² * d², d, 4, m₄, m₃, center_est)
        return (ψ₄ / m₂ - 2 * (κ + 3) * ψ₂) / m₂
    end
    return κ, ψ
end

##########
# cumulant
##########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.cumulant), x::AbstractArray{<:Real},
        k::Union{Integer, AbstractRange{<:Integer}}
    )
    return _cumulant_value_and_influence(x, k, (), Statistics.mean(x), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.cumulant), x::AbstractArray{<:Real},
        k::Union{Integer, AbstractRange{<:Integer}}, m::Real
    )
    return _cumulant_value_and_influence(x, k, (), m, false)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.cumulant), x::AbstractArray{<:Real},
        k::Union{Integer, AbstractRange{<:Integer}}, w::StatsBase.AbstractWeights
    )
    return _cumulant_value_and_influence(x, k, (w,), Statistics.mean(x, w), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.cumulant), x::AbstractArray{<:Real},
        k::Union{Integer, AbstractRange{<:Integer}}, w::StatsBase.AbstractWeights, m::Real
    )
    return _cumulant_value_and_influence(x, k, (w,), m, false)
end

# Each cumulant's influence function is a polynomial in the deviation d = t - c, and a single
# sequence aⱼ gives the coefficients of all of them.
#
# Let mᵢ be the mean of (xₗ - c)ⁱ over the data. StatsBase.cumulant takes κ₁ = c, and κₖ for
# k ≥ 2 as k! times the coefficient of sᵏ in log G(s), where G(s) = 1 + ∑ᵢ₌₂ mᵢ sⁱ / i!. (It
# leaves out m₁, even when `c` is not the mean.) Let 1 / G(s) = ∑ⱼ aⱼ sʲ / j!. Then:
#   - G(s) ⋅ (1 / G(s)) = 1 gives a₀ = 1, a₁ = 0 and aⱼ = -∑ᵢ₌₂ʲ binomial(j, i) mᵢ aⱼ₋ᵢ.
#   - (log G)′ = G′ ⋅ (1 / G) gives κₖ = ∑ⱼ₌₂ᵏ binomial(k - 1, j - 1) mⱼ aₖ₋ⱼ.
#   - Moving mass onto t changes mᵢ at the rate dⁱ - mᵢ, less i mᵢ₋₁ d if `c` is the estimated
#     mean. Summed over i, log G then changes at the rate e^(sd) / G(s) - 1 - sd. So the
#     influence of κₖ is k! times the coefficient of sᵏ in e^(sd) / G(s),
#         ∑ⱼ₌₀ᵏ binomial(k, j) aⱼ dᵏ⁻ʲ,
#     which also holds for κ₁ = c. With a given center the rate is (e^(sd) - sd) / G(s) - 1,
#     and the extra -sd / G(s) cancels exactly the term in d.
function _cumulant_value_and_influence(x, k, w, c, center_est)
    minimum(k) > 0 || throw(ArgumentError("Cumulant orders must be strictly positive."))
    K = maximum(k)
    # About a float center, integer data don't overflow. m[1] is not used.
    m = [StatsBase.moment(x, i, w..., float(c)) for i in 1:K]
    a = similar(m, K + 1)    # a[j + 1] is aⱼ
    κ = similar(m)           # cumulants
    a[1] = 1
    a[2] = 0
    κ[1] = c
    for i in 2:K
        a[i + 1] = -sum(j -> binomial(i, j) * m[j] * a[i - j + 1], 2:i)
        κ[i] = sum(j -> binomial(i - 1, j - 1) * m[j] * a[i - j + 1], 2:i)
    end
    ψ = map(k) do i
        p = [binomial(i, l) * a[i - l + 1] for l in 0:i]    # p[l + 1] multiplies dˡ
        if !center_est
            p[2] = 0
        end
        return evalpoly.(x .- c, Ref(p))
    end
    return κ[k], ψ
end
