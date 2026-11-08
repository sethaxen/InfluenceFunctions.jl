###############
# weighted mean
###############

# Weights are unsupported. They would otherwise be taken for the `x` of `mean(f, x)`, with the
# data as `f`, which StatsBase computes as a weighted mean before `f` fails to be called.
function InfluenceFunctions.value_and_influence(
        f::typeof(Statistics.mean), x::AbstractArray, w::StatsBase.AbstractWeights; kwargs...
    )
    throw(MethodError(InfluenceFunctions.value_and_influence, (f, x, w)))
end
