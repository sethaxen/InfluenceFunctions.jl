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
function _var_value_and_influence(x, c, corrected::Bool, dims)
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
function _std_value_and_influence(x, c, corrected::Bool, dims)
    s = Statistics.std(x; corrected, mean = c, dims)
    b = _bessel_factor(eltype(s), _slice_length(x, s), corrected)
    # ψ = ψ_var / 2s, with ψ_var = b |x - c|² - s². Where s = 0, every value in the slice
    # equals the center, so moving mass onto any of them changes nothing: ψ = 0, not 0/0.
    return s, @. ifelse(iszero(s), zero(s), (b * abs2(x - c) / s - s) / 2)
end

#####
# cov
#####

# Statistics.cov of a vector is its variance, though for ranges and empty vectors it is not
# computed as Statistics.var computes it.
function value_and_influence(::typeof(Statistics.cov), x::AbstractVector{<:Real}; corrected::Bool = true)
    v = Statistics.cov(x; corrected)
    b = _bessel_factor(typeof(v), length(x), corrected)
    return v, b .* abs2.(x .- Statistics.mean(x)) .- v
end

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

function value_and_influence(
        ::typeof(Statistics.cov), X::AbstractMatrix{<:Real}; dims::Int = 1, corrected::Bool = true
    )
    C = Statistics.cov(X; dims, corrected)
    return C, _with_variables(vars -> _cov_influences(C, vars, vars, corrected), X, dims)
end

function value_and_influence(
        ::typeof(Statistics.cov), X::AbstractVecOrMat{<:Real}, Y::AbstractVecOrMat{<:Real};
        dims::Int = 1, corrected::Bool = true
    )
    C = Statistics.cov(X, Y; dims, corrected)
    ψ = _with_variables(X, dims) do xs
        return _with_variables(ys -> _cov_influences(C, xs, ys, corrected), Y, dims)
    end
    return C, ψ
end

# Influence of each entry of `C`, the covariance matrix of the variables `xs` and `ys`.
function _cov_influences(C, xs, ys, corrected)
    μx, μy = map(Statistics.mean, xs), map(Statistics.mean, ys)
    return map(C, CartesianIndices(C)) do c, I
        j, l = Tuple(I)
        b = _bessel_factor(typeof(c), length(xs[j]), corrected)
        return b .* (xs[j] .- μx[j]) .* (ys[l] .- μy[l]) .- c
    end
end

#####
# cor
#####

# Statistics.cor of a vector is 1 for any data, so moving mass onto an observation changes nothing.
function value_and_influence(::typeof(Statistics.cor), x::AbstractVector{<:Real})
    ρ = Statistics.cor(x)
    return ρ, fill!(similar(x, typeof(ρ)), zero(ρ))
end

function value_and_influence(
        ::typeof(Statistics.cor), x::AbstractVector{<:Real}, y::AbstractVector{<:Real}
    )
    μx = Statistics.mean(x)
    σx = Statistics.std(x; mean = μx, corrected = false)
    μy = Statistics.mean(y)
    σy = Statistics.std(y; mean = μy, corrected = false)
    ρ = clamp(_mean_cross(x, μx, y, μy) / (σx * σy), -1, 1)    # clamped, as Statistics.cor does
    return ρ, _cor_influence(x, μx, σx, y, μy, σy, ρ)
end

function value_and_influence(::typeof(Statistics.cor), X::AbstractMatrix{<:Real}; dims::Int = 1)
    R = Statistics.cor(X; dims)
    return R, _with_variables(vars -> _cor_influences(R, vars, vars), X, dims)
end

function value_and_influence(
        ::typeof(Statistics.cor), X::AbstractVecOrMat{<:Real}, Y::AbstractVecOrMat{<:Real};
        dims::Int = 1
    )
    R = Statistics.cor(X, Y; dims)
    ψ = _with_variables(X, dims) do xs
        return _with_variables(ys -> _cor_influences(R, xs, ys), Y, dims)
    end
    return R, ψ
end

# Influence of each entry of `R`, the correlation matrix of the variables `xs` and `ys`.
function _cor_influences(R, xs, ys)
    μx, μy = map(Statistics.mean, xs), map(Statistics.mean, ys)
    σx = map((v, m) -> Statistics.std(v; mean = m, corrected = false), xs, μx)
    σy = map((v, m) -> Statistics.std(v; mean = m, corrected = false), ys, μy)
    return map(R, CartesianIndices(R)) do ρ, I
        j, l = Tuple(I)
        return _cor_influence(xs[j], μx[j], σx[j], ys[l], μy[l], σy[l], ρ)
    end
end

# Influence of the correlation `ρ` of `x` and `y`, given their means and standard deviations.
function _cor_influence(x, μx, σx, y, μy, σy, ρ)
    return broadcast(x, y) do s, t
        s̃, t̃ = (s - μx) / σx, (t - μy) / σy
        return s̃ * t̃ - ρ / 2 * (s̃^2 + t̃^2)
    end
end

##########
# quantile
##########

function value_and_influence(
        ::typeof(Statistics.quantile), x::AbstractArray{<:Real},
        p::Union{Real, AbstractArray{<:Real}, Tuple{Vararg{Real}}};
        density = QuantileSpacing(), kwargs...,
    )
    foreach(_check_level, p)
    q = Statistics.quantile(x, p; kwargs...)
    return q, _quantile_influences(density, x, q, p)
end

function value_and_influence(
        ::typeof(Statistics.quantile), f::Function, x::AbstractArray,
        p::Union{Real, AbstractArray{<:Real}, Tuple{Vararg{Real}}};
        density = QuantileSpacing(), kwargs...,
    )
    foreach(_check_level, p)
    z = f.(x)
    q = Statistics.quantile(z, p; kwargs...)
    return q, _quantile_influences(density, z, q, p)
end

function _check_level(p)
    0 < p < 1 || throw(DomainError(p, "the influence function of a quantile requires 0 < p < 1"))
    return nothing
end

# Influence at `t` of the quantile `q` at level `τ`, given the density `fq` at `q`.
_quantile_influence(t, q, τ, fq) = (τ - (t <= q)) / fq

# Influence values on the sample `y` of its quantiles `q` at levels `p`, from one density fit.
# `q` and `p` are numbers, or matching containers of them.
function _quantile_influences(density, y, q, p)
    g = fit_density(density, y)
    return map((qₗ, pₗ) -> _quantile_influence.(y, qₗ, pₗ, _density_at_quantile(g, qₗ, pₗ)), q, p)
end

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

function value_and_influence(
        ::typeof(Statistics.median), f::Function, x::AbstractArray; density = QuantileSpacing()
    )
    z = f.(x)
    m = Statistics.median(z)
    return m, _quantile_influences(density, z, m, oftype(float(m), 1 // 2))
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

# `f` applied to the variables of `X`: its slices along the axis that is not `dims`, or `X`
# itself if it is a vector, which is one variable for either `dims`. Each call of `f` sees one
# slice type, so its result is inferred. Statistics.cov and Statistics.cor accept any other
# `dims`, but what they return then says nothing about the variables.
function _with_variables(f, X, dims)
    dims in (1, 2) || throw(ArgumentError("dims must be 1 or 2, got $dims"))
    X isa AbstractVector && return f((X,))
    return dims == 1 ? f(eachcol(X)) : f(eachrow(X))
end
