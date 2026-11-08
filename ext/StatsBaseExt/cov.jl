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

######################
# weighted cov and cor
######################

# Weights are unsupported, and would otherwise be taken for a second variable.
function InfluenceFunctions.value_and_influence(
        f::typeof(Statistics.cov), x::DenseMatrix{<:Real}, w::StatsBase.AbstractWeights; kwargs...
    )
    throw(MethodError(InfluenceFunctions.value_and_influence, (f, x, w)))
end

function InfluenceFunctions.value_and_influence(
        f::typeof(Statistics.cor), x::DenseMatrix{<:Real}, w::StatsBase.AbstractWeights; kwargs...
    )
    throw(MethodError(InfluenceFunctions.value_and_influence, (f, x, w)))
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
