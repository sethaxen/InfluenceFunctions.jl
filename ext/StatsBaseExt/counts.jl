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
