######
# mean
######

function value_and_influence(::typeof(Statistics.mean), x::AbstractArray{<:Real}; dims = :)
    μ = Statistics.mean(x; dims)
    return μ, x .- μ
end

function value_and_influence(::typeof(Statistics.mean), f, x::AbstractArray; dims = :)
    μ = Statistics.mean(f, x; dims)
    return μ, f.(x) .- μ
end

##########
# var/varm
##########

function value_and_influence(
        ::typeof(Statistics.var), x::AbstractArray{<:Real};
        corrected::Bool = true, mean = nothing, dims = :
    )
    return _var_value_and_influence(x, mean, corrected, dims)
end

function value_and_influence(
        ::typeof(Statistics.varm), x::AbstractArray{<:Real}, m::Union{Real, AbstractArray{<:Real}};
        corrected::Bool = true, dims = :
    )
    return _var_value_and_influence(x, m, corrected, dims)
end

function _var_value_and_influence(x, ::Nothing, corrected::Bool, dims)
    return _var_value_and_influence(x, Statistics.mean(x; dims), corrected, dims)
end
_var_value_and_influence(x, c, corrected::Bool, dims) = _var_about(x, c, corrected, dims)
function _var_about(x, c, corrected, dims)
    v = Statistics.var(x; corrected, mean = c, dims)
    b = _bessel_factor(eltype(v), _slice_length(x, v), corrected)
    return v, b .* abs2.(x .- c) .- v
end

##########
# std/stdm
##########

function value_and_influence(
        ::typeof(Statistics.std), x::AbstractArray{<:Real};
        corrected::Bool = true, mean = nothing, dims = :
    )
    return _std_value_and_influence(x, mean, corrected, dims)
end

function value_and_influence(
        ::typeof(Statistics.stdm), x::AbstractArray{<:Real}, m::Union{Real, AbstractArray{<:Real}};
        corrected::Bool = true, dims = :
    )
    return _std_value_and_influence(x, m, corrected, dims)
end

function _std_value_and_influence(x, ::Nothing, corrected::Bool, dims)
    return _std_value_and_influence(x, Statistics.mean(x; dims), corrected, dims)
end
_std_value_and_influence(x, c, corrected::Bool, dims) = _std_about(x, c, corrected, dims)
function _std_about(x, c, corrected, dims)
    s = Statistics.std(x; corrected, mean = c, dims)
    b = _bessel_factor(eltype(s), _slice_length(x, s), corrected)
    # ψ = ψ_var / 2s, with ψ_var = b |x - c|² - s². Where s = 0, every value in the slice
    # equals the center, so moving mass onto any of them changes nothing: ψ = 0, not 0/0.
    return s, @. ifelse(iszero(s), zero(s), (b * abs2(x - c) / s - s) / 2)
end

#####
# cov
#####

function value_and_influence(
        ::typeof(Statistics.cov), x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
        corrected::Bool = true
    )
    μx = Statistics.mean(x)
    μy = Statistics.mean(y)
    Ĉ = _mean_cross(x, μx, y, μy)    # uncorrected covariance
    b = _bessel_factor(typeof(Ĉ), length(x), corrected)
    v = b * Ĉ
    return v, b .* (x .- μx) .* (y .- μy) .- v
end

#####
# cor
#####

function value_and_influence(
        ::typeof(Statistics.cor), x::AbstractVector{<:Real}, y::AbstractVector{<:Real}
    )
    μx = Statistics.mean(x)
    σx = Statistics.std(x; mean = μx, corrected = false)
    μy = Statistics.mean(y)
    σy = Statistics.std(y; mean = μy, corrected = false)
    ρ = clamp(_mean_cross(x, μx, y, μy) / (σx * σy), -1, 1)    # clamped, as Statistics.cor does
    ψ = broadcast(x, y) do s, t
        s̃, t̃ = (s - μx) / σx, (t - μy) / σy
        return s̃ * t̃ - ρ / 2 * (s̃^2 + t̃^2)
    end
    return ρ, ψ
end

##########
# quantile
##########

function value_and_influence(
        ::typeof(Statistics.quantile), x::AbstractArray{<:Real}, p::Real;
        density = QuantileSpacing(), kwargs...,
    )
    0 < p < 1 || throw(DomainError(p, "the influence function of a quantile requires 0 < p < 1"))
    q = Statistics.quantile(x, p; kwargs...)
    g = fit_density(density, x)
    return q, _quantile_influence.(x, q, p, _density_at_quantile(g, q, p))
end

# Influence at `t` of the quantile `q` at level `τ`, given the density `fq` at `q`.
_quantile_influence(t, q, τ, fq) = (τ - (t <= q)) / fq

########
# median
########

function value_and_influence(
        ::typeof(Statistics.median), x::AbstractArray{<:Real};
        density = QuantileSpacing(), kwargs...,
    )
    m = Statistics.median(x; kwargs...)
    τ = convert(float(eltype(m)), 1 // 2)
    dims = get(kwargs, :dims, :)
    if dims isa Colon
        fm = _density_at_quantile(fit_density(density, x), m, τ)
    else
        # Each slice is its own sample, with its own density fit at its own median.
        fm = map(CartesianIndices(m)) do I
            slice = view(x, ntuple(d -> size(m, d) == size(x, d) ? I[d] : Colon(), ndims(x))...)
            return _density_at_quantile(fit_density(density, slice), m[I], τ)
        end
    end
    return m, _quantile_influence.(x, m, τ, fm)
end

#######
# utils
#######

# Bessel factor `n/(n-1)` when `corrected`, else 1, in type `T`.
_bessel_factor(::Type{T}, n::Integer, corrected::Bool) where {T} = corrected ? T(n) / T(n - 1) : one(T)

# Number of observations in each slice of `x` reduced to one entry of the value `v`.
_slice_length(x, v) = length(x) ÷ max(length(v), 1)

# Mean of (x - μx)(y - μy)
_mean_cross(x, μx, y, μy) = Statistics.mean(i -> (x[i] - μx) * (y[i] - μy), eachindex(x, y))
