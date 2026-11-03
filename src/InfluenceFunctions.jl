module InfluenceFunctions

using Statistics: Statistics

export influence, value_and_influence

"""
    value_and_influence(f, args...; kwargs...) -> (value, ψ)

Compute the estimate `value = f(args...; kwargs...)` together with its influence values `ψ`.

`ψ[i]` estimates the influence function of the estimator `f`, evaluated at the observation
`x[i]`, where `x` is the data argument (the first array in `args`). `ψ` has the array type
and axes of `x`; its element type is whatever the influence computation produces.

Two conventions are used:

  - **Smooth estimators** (means, moments, variances, correlations): `ψ[i]` is the rate of
    change of the estimate as probability mass moves onto observation `i`, with any
    finite-sample constants (such as the `n/(n-1)` of `corrected=true`) held fixed. So
    `sum(ψ) ≈ 0`.
  - **Quantile-type estimators** (quantiles, medians, `iqr`, `mad`): `ψ` is the plug-in
    estimate of the population influence function, which needs a density estimate at the
    estimated quantiles. These methods accept one keyword that `f` does not: `density`.

Every other keyword must be one that `f` accepts for that call; anything else, like an
estimator without a method, throws a `MethodError`.

To support a new estimator `g`, add a method
`InfluenceFunctions.value_and_influence(::typeof(g), args...; kwargs...)`. If `ψ` can be
computed more cheaply on its own, also add a method to [`influence`](@ref).

# Examples

```jldoctest
julia> using InfluenceFunctions, Statistics

julia> value_and_influence(mean, [1.0, 2.0, 6.0])
(3.0, [-2.0, -1.0, 3.0])
```
"""
function value_and_influence end

"""
    influence(f, args...; kwargs...) -> ψ

Influence values of the estimator `f` on its data. See [`value_and_influence`](@ref).

The default is `last(value_and_influence(f, args...; kwargs...))`. Estimators whose
influence values can be computed more cheaply on their own overload this function as well.

# Examples

```jldoctest
julia> using InfluenceFunctions, Statistics

julia> influence(mean, [1.0, 2.0, 6.0])
3-element Vector{Float64}:
 -2.0
 -1.0
  3.0
```
"""
influence(f, args...; kwargs...) = last(value_and_influence(f, args...; kwargs...))

include("statistics.jl")

end
