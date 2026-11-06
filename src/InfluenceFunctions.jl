module InfluenceFunctions

using DensityInterface: DensityInterface
using Statistics: Statistics
using StatsFuns: StatsFuns

export influence, value_and_influence, QuantileSpacing
@static if VERSION >= v"1.11.0-DEV.469"
    eval(Meta.parse("public fit_density, HallSheatherBandwidth"))
end

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

When `f` returns several estimates, the influence values of an estimate do not depend on
which function returned it:

  - **A tuple** (`mean_and_std(x)`, `quantile(x, (0.25, 0.75))`): `ψ` is a tuple holding what
    the estimator of each element gives on its own.
  - **An array of estimates from one sample** (`quantile(x, ps)`, `cov(X)`, `proportions(x)`):
    `ψ` is an array with the axes of the value, and each element holds the influence values
    of that estimate, one per observation. For `cov`, `cor` and `scattermat` of a matrix, the
    observations are the slices of the matrix along `dims`.
  - **A reduction over `dims`** (`mean(x; dims)`): every data element belongs to one slice,
    so `ψ` stays a single array with the axes of `x`, and each slice holds the influence
    values of the estimate for that slice alone.

Influence values combine by the delta method. For a differentiable function `h` of the
estimates, let `∇h` be its gradient at `value`, for example from an automatic differentiation
package. The influence values of `h(value)` are `sum(map((∂ₖ, ψₖ) -> ∂ₖ .* ψₖ, ∇h, ψ))` for a
tuple, and `sum(∇h .* ψ)` for an array of estimates. For a reduction over `dims`, `ψ` cannot
say how the slices relate: with `dims = 1`, use `∇h .* ψ` if each column is its own sample,
and `sum(∇h .* ψ; dims = 2)` if the columns are variables measured on the same rows.

To support a new estimator `g`, add a method
`InfluenceFunctions.value_and_influence(::typeof(g), args...; kwargs...)`. If `ψ` can be
computed more cheaply on its own, also add a method to [`influence`](@ref).

# Examples

```jldoctest
julia> using InfluenceFunctions, Statistics

julia> value_and_influence(mean, [1.0, 2.0, 6.0])
(3.0, [-2.0, -1.0, 3.0])

julia> C, ψ = value_and_influence(cov, [1.0 2.0; 3.0 2.0; 5.0 8.0]);

julia> C[1, 2], ψ[1, 2]
(6.0, [0.0, -6.0, 6.0])
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

include("density.jl")
include("statistics.jl")

end
