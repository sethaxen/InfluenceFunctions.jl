############
# scattermat
############

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.scattermat), x::DenseMatrix{<:Real}; mean = nothing, dims::Int = 1
    )
    S = StatsBase.scattermat(x; mean, dims)
    z = x .- (mean === nothing ? Statistics.mean(x; dims) : mean)
    return S, _scattermat_influences(S, z, size(x, dims), dims)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.scattermat), x::DenseMatrix{<:Real}, w::StatsBase.AbstractWeights;
        mean = nothing, dims::Int = 1
    )
    S = StatsBase.scattermat(x, w; mean, dims)
    z = x .- (mean === nothing ? Statistics.mean(x, w; dims) : mean)
    return S, _scattermat_influences(S, z, sum(w), dims)
end

# S = n Ĉ, with Ĉ the uncorrected covariance about the center and n the number of observations
# (or the total weight) held fixed, so with `z` the centered variables, ψ = n zⱼ zₗ - S.
# Estimating the center adds nothing, as for `var`.
function _scattermat_influences(S, z, n, dims)
    return InfluenceFunctions._with_variables(z, dims) do vars
        return map(S, CartesianIndices(S)) do s, I
            j, l = Tuple(I)
            return n .* vars[j] .* vars[l] .- s
        end
    end
end

######################
# weighted cov and cor
######################

function InfluenceFunctions.value_and_influence(
        ::typeof(Statistics.cov), x::DenseMatrix{<:Real}, w::StatsBase.AbstractWeights, dims::Int = 1;
        corrected = nothing
    )
    C = Statistics.cov(x, w, dims; corrected)
    return C, _cov_influences(C, x, w, dims, corrected)
end

# Influence of each entry of `C`, the weighted covariance matrix of the variables of `x`: with
# `z` the variables centered at their weighted means and b the bias factor held fixed,
# ψ = b zⱼ zₗ - C. StatsBase takes `corrected = nothing` as false, with a deprecation warning.
function _cov_influences(C, x, w, dims, corrected)
    b = _bessel_factor(eltype(C), w, something(corrected, false))
    return InfluenceFunctions._with_variables(x, dims) do vars
        μ = map(v -> Statistics.mean(v, w), vars)
        return map(C, CartesianIndices(C)) do c, I
            j, l = Tuple(I)
            return b .* (vars[j] .- μ[j]) .* (vars[l] .- μ[l]) .- c
        end
    end
end

function InfluenceFunctions.value_and_influence(
        ::typeof(Statistics.cor), x::DenseMatrix{<:Real}, w::StatsBase.AbstractWeights, dims::Int = 1
    )
    R = Statistics.cor(x, w, dims)
    ψ = InfluenceFunctions._with_variables(x, dims) do vars
        μ = map(v -> Statistics.mean(v, w), vars)
        σ = map((v, m) -> Statistics.std(v, w; mean = m, corrected = false), vars, μ)
        return map(R, CartesianIndices(R)) do ρ, I
            j, l = Tuple(I)
            return InfluenceFunctions._cor_influence(vars[j], μ[j], σ[j], vars[l], μ[l], σ[l], ρ)
        end
    end
    return R, ψ
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

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.mean_and_cov), x::DenseMatrix{<:Real}, w::StatsBase.AbstractWeights,
        dims::Int = 1; corrected = nothing
    )
    m, C = StatsBase.mean_and_cov(x, w, dims; corrected)
    return (m, C), (x .- m, _cov_influences(C, x, w, dims, corrected))
end
