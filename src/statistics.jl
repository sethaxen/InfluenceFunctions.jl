function value_and_influence(::typeof(Statistics.mean), x::AbstractArray{<:Real}; dims = :)
    μ = Statistics.mean(x; dims)
    return μ, x .- μ
end

function value_and_influence(::typeof(Statistics.mean), f, x::AbstractArray; dims = :)
    μ = Statistics.mean(f, x; dims)
    return μ, f.(x) .- μ
end
