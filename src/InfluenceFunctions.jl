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

An influence value estimates the influence function of one estimate, evaluated at one
observation of the data argument `x` (the first array in `args`). Its type is whatever the
influence computation produces. `ψ` mirrors `value`, and has the array type and axes of `x`
exactly when the observations are the elements of `x` and each contributes to only one
estimate:

  1. **`value` is a number:** `ψ` is one array, with one influence value per observation.
  2. **`value` reduces over `dims`** (`mean(x; dims)`): `ψ` is still one array with the
     axes of `x`, and each slice holds the influence values of its own estimate.
  3. **`value` is any other array** (`quantile(x, ps)`, `cov(X)`, `proportions(x)`): `ψ` has
     the axes of `value`, and each element holds one estimate's influence values, one per
     observation.
  4. **`value` is a tuple or named tuple** (`mean_and_std(x)`, `quantile(x, (0.25, 0.75))`):
     `ψ` has the same structure, and each element follows rules 1–3 as its estimator would
     on its own.

Observations are the elements of `x`, except for a matrix passed to `cov`, `cor` or
`scattermat`, where they are its slices along `dims`, or to `genvar` or `totalvar`, where they
are its rows. An array of influence values then has the axes of `x` along that dimension. For
forms with several data arguments, such as `cov(x, y)`, `proportions(x, y)` and
`partialcor(x, y, Z)`, observations are pairs or triples, with the axes of `x`.

Two conventions are used:

  - **Smooth estimators** (means, moments, variances, correlations): an influence value is
    the rate of change of the estimate as probability mass moves onto its observation, with
    any finite-sample constants (such as the `n/(n-1)` of `corrected=true`) held fixed. So
    each estimate's influence values sum to about 0.
  - **Quantile-type estimators** (quantiles, medians, `iqr`, `mad`): an influence value is
    the plug-in estimate of the population influence function, which needs a density
    estimate at the estimated quantiles. These methods accept one keyword that `f` does
    not: `density`.

Where StatsBase computes a smooth estimator with weights `w` (any `AbstractWeights`), they
are accepted in the same position. Observation `i` then carries probability mass
`w[i] / sum(w)`, and an influence value is the rate of change of the estimate as mass moves
onto its observation with `sum(w)` held fixed. The bias correction that `corrected` selects
for the type of weights is held fixed too, even for analytic weights, whose correction
depends on how the weights are spread. Influence values are not multiplied by the weights:
`sum(w .* ψ)` is about 0, and where StatsBase defines a weighted standard error of the mean
(frequency, probability and unit weights), `StatsBase.sem(ψ, w)` estimates the standard
error of the estimate, just as `sem(ψ)` does without weights.

Every other keyword must be one that `f` accepts for that call; anything else, like an
estimator without a method, throws a `MethodError`.

Influence values combine by the delta method. For a differentiable function `h` of the
estimates, let `∇h` be its gradient at `value`, for example from an automatic differentiation
package. The influence values of `h(value)` are `sum(∇h .* ψ)` under rule 3, and
`sum(map((∂ₖ, ψₖ) -> ∂ₖ .* ψₖ, ∇h, ψ))` under rule 4. Under rule 2, `ψ` cannot say how the
slices relate: with `dims = 1`, use `∇h .* ψ` if each column is its own sample, and
`sum(∇h .* ψ; dims = 2)` if the columns are variables measured on the same rows.

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
