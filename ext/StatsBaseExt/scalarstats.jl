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

###########
# variation
###########

# V = s / m, so ψ = ψ_stdm / m.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.variation), x::AbstractArray{<:Real}, m::Real; corrected::Bool = true
    )
    s, ψ = InfluenceFunctions.value_and_influence(Statistics.stdm, x, m; corrected)
    V = s / m
    ψ ./= m
    return V, ψ
end

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
    m = @something(center, Statistics.median(x))
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

########
# genvar
########

# The delta method on `cov`. For `X` with k columns, let zᵢ be the centered row i, S the mean of
# the zᵢzᵢᵀ, and b the Bessel factor, so that C = cov(X) = bS. The influence values of C are
# b zᵢzᵢᵀ - C, and the gradient of the determinant at the symmetric C is G C⁻¹, so
# ψᵢ = G tr(C⁻¹(b zᵢzᵢᵀ - C)) = G (zᵢᵀS⁻¹zᵢ - k). If G = 0, then ψ = 0: the formula gives that
# wherever C⁻¹ exists, and where C is singular, every centered observation lies in its column
# space, which the gradient, its adjugate, maps to 0. If G is NaN, as from a NaN in the data or
# a single row, then so is ψ.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.genvar), X::AbstractMatrix{<:Real}
    )
    G = StatsBase.genvar(X)
    ψ = fill!(similar(X, typeof(G), axes(X, 1)), zero(G))
    iszero(G) && return G, ψ
    isnan(G) && return G, fill!(ψ, G)
    μs, S = _mean_and_cov(X)
    P = inv(S)
    for j in axes(X, 2), l in axes(X, 2)    # ψᵢ = zᵢᵀS⁻¹zᵢ
        ψ .+= P[j, l] .* (view(X, :, j) .- μs[j]) .* (view(X, :, l) .- μs[l])
    end
    ψ .= G .* (ψ .- size(X, 2))
    return G, ψ
end
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.genvar), x::AbstractArray{<:Real}
    )
    return InfluenceFunctions.value_and_influence(Statistics.var, x)
end

# Allocates less than mean_and_cov, does not correct, accepts an AbstractVecOrMat
function _mean_and_cov(X)
    μs = Statistics.mean(X; dims = 1)
    S = [
        InfluenceFunctions._mean_cross(view(X, :, j), μs[j], view(X, :, l), μs[l])
            for j in axes(X, 2), l in axes(X, 2)
    ]
    return μs, S
end

##########
# totalvar
##########

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.totalvar), X::AbstractMatrix{<:Real}
    )
    vars, ψs = InfluenceFunctions.value_and_influence(Statistics.var, X; dims = 1)
    return sum(vars), dropdims(sum(ψs; dims = 2); dims = 2)
end

# For any other array, StatsBase computes the variance of all elements.
function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.totalvar), x::AbstractArray{<:Real}
    )
    return InfluenceFunctions.value_and_influence(Statistics.var, x)
end
