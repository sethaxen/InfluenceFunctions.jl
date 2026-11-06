module StatsBaseExt

using InfluenceFunctions: InfluenceFunctions
using Statistics: Statistics
using StatsBase: StatsBase
using StatsFuns: StatsFuns

#########
# geomean
#########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.geomean), x::AbstractArray{<:Real},
    )
    m = Statistics.mean(log, x)
    G = exp(m)
    # G = 0 means some observation is 0, and G stays 0 while it keeps any mass: ψ = 0, not NaN.
    return G, @. ifelse(iszero(G), zero(G), G * (log(x) - m))
end

##########
# harmmean
##########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.harmmean), x::AbstractArray{<:Real},
    )
    H = inv(Statistics.mean(inv, x))
    return H, @. ifelse(iszero(H), zero(H), H * (1 - H / x))    # H = 0 as for geomean
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
    # As for geomean, M stays 0 if a 0 keeps any mass (p < 0) or all observations are 0: ψ = 0,
    # not NaN. Any other M = 0 is terms of both signs cancelling, which doesn't stay 0.
    stays_zero = iszero(M) && (p < 0 || all(iszero, x))
    return M, @. ifelse(stays_zero, zero(M), (M / p) * (float(x)^p / S - 1))
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
    m₂, m₃ = (StatsBase.moment(x, k, c) for k in 2:3)
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
    m₂, m₃ = (StatsBase.moment(x, k, c) for k in 2:3)
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

###########
# variation
###########

# V = s / μ̂, so ψ = (ψ_std - V (x - μ̂)) / μ̂.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.variation), x::AbstractArray{<:Real}; corrected::Bool = true
    )
    μ = Statistics.mean(x)
    s, ψ = InfluenceFunctions.value_and_influence(Statistics.stdm, x, μ; corrected)
    V = s / μ
    ψ .= (ψ .- V .* (x .- μ)) ./ μ
    return V, ψ
end

# V = s / m, so ψ = ψ_stdm / m.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.variation), x::AbstractArray{<:Real}, m::Real; corrected::Bool = true
    )
    s, ψ = InfluenceFunctions.value_and_influence(Statistics.stdm, x, m; corrected)
    V = s / m
    ψ ./= m
    return V, ψ
end

############
# percentile
############

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.percentile), x::AbstractArray{<:Real}, p::Real;
        density = InfluenceFunctions.QuantileSpacing()
    )
    # p * 0.01 as in StatsBase.percentile; p / 100 differs in the last bit for some p
    return InfluenceFunctions.value_and_influence(Statistics.quantile, x, p * 0.01; density)
end

#####
# iqr
#####

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.iqr), x::AbstractArray{<:Real};
        density = InfluenceFunctions.QuantileSpacing()
    )
    ql, qu = Statistics.quantile(x, (0.25, 0.75))
    g = InfluenceFunctions.fit_density(density, x)
    fl = InfluenceFunctions._density_at_quantile(g, ql, 0.25)
    fu = InfluenceFunctions._density_at_quantile(g, qu, 0.75)
    ψ = InfluenceFunctions._quantile_influence.(x, qu, 0.75, fu) .-
        InfluenceFunctions._quantile_influence.(x, ql, 0.25, fl)
    return qu - ql, ψ
end

#####
# mad
#####

# With ŝ the unnormalized MAD about the center m̂ and f₀, f₊, f₋ the densities at m̂, m̂ + ŝ,
# m̂ - ŝ, ψ = k [(½ - 1[|x - m̂| ≤ ŝ]) / (f₊ + f₋) - r ψ_m], where r = (f₊ - f₋) / (f₊ + f₋) and
# ψ_m is the median's influence (0 for a given center). This form takes the formula's
# limits when f₊ or f₋ is infinite (an atom at m̂ ± ŝ), where it would be Inf/Inf.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mad), x::AbstractArray{<:Real};
        center = nothing, normalize = nothing, density = InfluenceFunctions.QuantileSpacing(),
    )
    m = center === nothing ? Statistics.median(x) : center
    s = StatsBase.mad(x; center = m, normalize = false)
    k = normalize === false ? one(s) : _mad_constant(typeof(s))    # `nothing` means true in StatsBase 0.34
    τ = oftype(float(s), 1 // 2)
    g = InfluenceFunctions.fit_density(density, x)
    lo, hi = extrema(x)
    tol = 4 * eps(float(max(abs(m), s)))    # bounds the rounding error of m ± s
    f₊ = g(_undo_overshoot(m + s, lo, hi, tol))
    f₋ = g(_undo_overshoot(m - s, lo, hi, tol))
    # A given center has no influence (r = 0, and f₀ is then unused). Each variable the
    # closure below captures is assigned once, so it isn't boxed.
    f₀ = center === nothing ? InfluenceFunctions._density_at_quantile(g, m, τ) : one(f₊)
    r = center === nothing ? _density_ratio(f₊, f₋) : zero(f₊)
    ψ = broadcast(x) do t
        ψ_med = InfluenceFunctions._quantile_influence(t, m, τ, f₀)
        return k * ((τ - (abs(t - m) <= s)) / (f₊ + f₋) - r * ψ_med)
    end
    return k * s, ψ
end

# 1/Φ⁻¹(3/4), which makes the MAD a consistent estimator of the standard deviation for normal
# data. Computed in at least Float64 precision and then rounded, so it is correctly rounded
# (as StatsBase's constant is) also for Float32; the compiler folds it to a literal.
function _mad_constant(::Type{T}) where {T <: Real}
    S = promote_type(Float64, float(T))
    return float(T)(inv(StatsFuns.norminvcdf(S(3 // 4))))
end

# `t`, moved into the range [lo, hi] of the data if it is outside by at most `tol`. m̂ ± ŝ can
# lie on the data but round to just past it, where a density estimate is 0.
_undo_overshoot(t, lo, hi, tol) = lo - tol <= t <= hi + tol ? clamp(t, lo, hi) : t

# (f₊ - f₋) / (f₊ + f₋), taking its limits when either density is infinite.
function _density_ratio(f₊, f₋)
    f₊ == f₋ && return zero(f₊)
    (isinf(f₊) || isinf(f₋)) && return sign(f₊ - f₋)
    return (f₊ - f₋) / (f₊ + f₋)
end

end
