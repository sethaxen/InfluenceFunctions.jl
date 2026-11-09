#############
# proportions
#############

# Each proportion is the mean of an indicator, with or without weights, so its influence values
# are that indicator less the proportion. The methods with weights mirror StatsBase's, which
# take the weights last. Integer weights are integer arrays too, but as in StatsBase, a method
# for weights is more specific than one for a second variable.

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.proportions), x::AbstractArray{<:Integer},
        levels::UnitRange{<:Integer} = StatsBase.span(x)
    )
    p = StatsBase.proportions(x, levels)
    return p, _proportion_influences(p, x, levels)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, levels::UnitRange{<:Integer},
        w::StatsBase.AbstractWeights
    )
    p = StatsBase.proportions(x, levels, w)
    return p, _proportion_influences(p, x, levels)
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, k::Integer
    )
    return InfluenceFunctions.value_and_influence(f, x, 1:k)
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, k::Integer,
        w::StatsBase.AbstractWeights
    )
    return InfluenceFunctions.value_and_influence(f, x, 1:k, w)
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, w::StatsBase.AbstractWeights
    )
    return InfluenceFunctions.value_and_influence(f, x, StatsBase.span(x), w)
end

_proportion_influences(p, x, levels) = map((pₗ, l) -> (x .== l) .- pₗ, p, levels)

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer},
        levels::NTuple{2, UnitRange{<:Integer}} = (StatsBase.span(x), StatsBase.span(y))
    )
    p = StatsBase.proportions(x, y, levels)
    return p, _proportion_influences(p, x, y, levels)
end

function InfluenceFunctions.value_and_influence(
        ::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer},
        levels::NTuple{2, UnitRange{<:Integer}}, w::StatsBase.AbstractWeights
    )
    p = StatsBase.proportions(x, y, levels, w)
    return p, _proportion_influences(p, x, y, levels)
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer},
        ks::NTuple{2, Integer}
    )
    return InfluenceFunctions.value_and_influence(f, x, y, (1:ks[1], 1:ks[2]))
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer},
        ks::NTuple{2, Integer}, w::StatsBase.AbstractWeights
    )
    return InfluenceFunctions.value_and_influence(f, x, y, (1:ks[1], 1:ks[2]), w)
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer},
        k::Integer
    )
    return InfluenceFunctions.value_and_influence(f, x, y, (1:k, 1:k))
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer},
        k::Integer, w::StatsBase.AbstractWeights
    )
    return InfluenceFunctions.value_and_influence(f, x, y, (1:k, 1:k), w)
end

function InfluenceFunctions.value_and_influence(
        f::typeof(StatsBase.proportions), x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer},
        w::StatsBase.AbstractWeights
    )
    return InfluenceFunctions.value_and_influence(f, x, y, (StatsBase.span(x), StatsBase.span(y)), w)
end

function _proportion_influences(p, x, y, levels)
    return map(p, CartesianIndices(p)) do pₗ, I
        a, b = levels[1][I[1]], levels[2][I[2]]
        return ((x .== a) .& (y .== b)) .- pₗ
    end
end
