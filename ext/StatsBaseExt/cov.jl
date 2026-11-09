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
    # StatsBase takes `corrected = nothing` as false, with a deprecation warning.
    ψ = InfluenceFunctions._with_variables(x, dims) do vars
        return InfluenceFunctions._cov_influences(C, vars, vars, something(corrected, false), w)
    end
    return C, ψ
end

function InfluenceFunctions.value_and_influence(
        ::typeof(Statistics.cor), x::DenseMatrix{<:Real}, w::StatsBase.AbstractWeights, dims::Int = 1
    )
    R = Statistics.cor(x, w, dims)
    ψ = InfluenceFunctions._with_variables(x, dims) do vars
        return InfluenceFunctions._cor_influences(R, vars, vars, w)
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
    m, ψm = InfluenceFunctions.value_and_influence(Statistics.mean, x, w; dims)
    C, ψC = InfluenceFunctions.value_and_influence(Statistics.cov, x, w, dims; corrected)
    return (m, C), (ψm, ψC)
end
