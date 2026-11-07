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
        ::typeof(StatsBase.percentile), x::AbstractArray{<:Real},
        p::Union{Real, AbstractArray{<:Real}};
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
    f₊ = g(m + s)
    f₋ = g(m - s)
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

# (f₊ - f₋) / (f₊ + f₋), taking its limits when either density is infinite.
function _density_ratio(f₊, f₋)
    f₊ == f₋ && return zero(f₊)
    (isinf(f₊) || isinf(f₋)) && return sign(f₊ - f₋)
    return (f₊ - f₋) / (f₊ + f₋)
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
        ::typeof(StatsBase.mean_and_var), x::AbstractArray{<:Real}, dim::Int; corrected::Bool = true
    )
    return _mean_and_value_and_influence(Statistics.var, x, dim, corrected)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_std), x::AbstractArray{<:Real}; corrected::Bool = true
    )
    return _mean_and_value_and_influence(Statistics.std, x, :, corrected)
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

##############
# mean_and_cov
##############

# Each estimate's ψ is what its own estimator gives, so the mean keeps the shape of a `dims`
# reduction even though rows are the observations here.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_cov), x::DenseMatrix{<:Real}, dims::Int = 1;
        corrected::Bool = true
    )
    m, ψm = InfluenceFunctions.value_and_influence(Statistics.mean, x; dims)
    C, ψC = InfluenceFunctions.value_and_influence(Statistics.cov, x; dims, corrected)
    return (m, C), (ψm, ψC)
end

######################
# weighted cov and cor
######################

# These are unsupported, but weights are vectors, so without this method they would be taken for
# a second variable. The signature is that of StatsBase's methods: for any other `x`, they are
# a second variable to StatsBase too.
function InfluenceFunctions.value_and_influence(
        f::Union{typeof(Statistics.cov), typeof(Statistics.cor)}, x::DenseMatrix{<:Real},
        w::StatsBase.AbstractWeights; kwargs...
    )
    throw(MethodError(InfluenceFunctions.value_and_influence, (f, x, w)))
end

############
# scattermat
############

# S = n Ĉ, with Ĉ the uncorrected covariance about the center and n held fixed, so with `z` the
# centered variables, ψ = n zⱼ zₗ - S. Estimating the center adds nothing, as for `var`.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.scattermat), x::DenseMatrix{<:Real}; mean = nothing, dims::Int = 1
    )
    S = StatsBase.scattermat(x; mean, dims)
    z = x .- (mean === nothing ? Statistics.mean(x; dims) : mean)
    ψ = InfluenceFunctions._with_variables(z, dims) do vars
        return map(S, CartesianIndices(S)) do s, I
            j, l = Tuple(I)
            return length(vars[j]) .* vars[j] .* vars[l] .- s
        end
    end
    return S, ψ
end

##########
# cumulant
##########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.cumulant), x::AbstractArray{<:Real},
        k::Union{Integer, AbstractRange{<:Integer}}
    )
    return _cumulant_value_and_influence(x, k, Statistics.mean(x), true)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.cumulant), x::AbstractArray{<:Real},
        k::Union{Integer, AbstractRange{<:Integer}}, m::Real
    )
    return _cumulant_value_and_influence(x, k, m, false)
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
function _cumulant_value_and_influence(x, k, c, center_est)
    minimum(k) > 0 || throw(ArgumentError("Cumulant orders must be strictly positive."))
    K = maximum(k)
    # About a float center, integer data don't overflow. m[1] is not used.
    m = [StatsBase.moment(x, i, float(c)) for i in 1:K]
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

#############
# proportions
#############

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.proportions), x::AbstractArray{<:Integer},
        levels::UnitRange{<:Integer} = StatsBase.span(x)
    )
    p = StatsBase.proportions(x, levels)
    return p, map((pₗ, l) -> (x .== l) .- pₗ, p, levels)
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, k::Integer
    )
    return InfluenceFunctions.value_and_influence(f, x, 1:k)
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer},
        levels::NTuple{2, UnitRange{<:Integer}} = (StatsBase.span(x), StatsBase.span(y))
    )
    # Integer weights are integer arrays too, but StatsBase takes them as weights.
    y isa StatsBase.AbstractWeights &&
        throw(MethodError(InfluenceFunctions.value_and_influence, (f, x, y, levels)))
    p = StatsBase.proportions(x, y, levels)
    ψ = map(p, CartesianIndices(p)) do pₗ, I
        a, b = levels[1][I[1]], levels[2][I[2]]
        return ((x .== a) .& (y .== b)) .- pₗ
    end
    return p, ψ
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer},
        ks::NTuple{2, Integer}
    )
    return InfluenceFunctions.value_and_influence(f, x, y, (1:ks[1], 1:ks[2]))
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer},
        k::Integer
    )
    return InfluenceFunctions.value_and_influence(f, x, y, (1:k, 1:k))
end

end
