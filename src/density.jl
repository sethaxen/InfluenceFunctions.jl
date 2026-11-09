"""
    fit_density(density, x[, w]) -> g

Fit `density` to the data `x`, returning a callable `g` with `g(t)` ≈ the density of the
data at `t`.

Quantile-type estimators call this once per call, on their own data, and evaluate `g`
wherever their influence function needs a density. Weighted estimators pass their weights
`w` too, and `g` is then the density of the distribution that puts mass
`w[i] / sum(w)` on `x[i]`.

`density` may be:

  - a density estimator, such as [`QuantileSpacing`](@ref), the default;
  - an object implementing DensityInterface.jl, such as a Distributions.jl distribution,
    evaluated with `DensityInterface.densityof`. Its density must be with respect to
    Lebesgue measure;
  - a `Function` `t -> f(t)` giving a known density.

A known density ignores `w` and represents the *target* distribution.

To support your own density estimator, define one or both of the following methods:
- `InfluenceFunctions.fit_density(est::MyEstimator, x)`: used in unweighted quantile-based
    estimators.
- `InfluenceFunctions.fit_density(est::MyEstimator, x, w)`: used in weighted quantile-based
    estimators.
"""
function fit_density(density, x, w...)
    if DensityInterface.DensityKind(density) !== DensityInterface.NoDensity()
        return Base.Fix1(DensityInterface.densityof, density)
    elseif density isa Function
        return density
    end
    weighted = isempty(w) ? "" : " for weights"
    throw(
        ArgumentError(
            "`density` must be a density estimator with a `fit_density` method$weighted " *
                "(e.g. `QuantileSpacing()`), an object implementing DensityInterface.jl, " *
                "or a `Function`; got $(repr(density))",
        ),
    )
end

# Density at an estimator's own quantile `q`, which has nominal level `τ`. A fitted density
# is evaluated at `q`; `QuantileSpacing` uses the level directly.
_density_at_quantile(g, q, τ) = g(q)

"""
    HallSheatherBandwidth(; alpha = 0.05)

The Hall–Sheather (1988) bandwidth rule for [`QuantileSpacing`](@ref), with the Gaussian
plug-in as given in Koenker (2005):

```math
h = n^{-1/3} z_{1-\\alpha/2}^{2/3}
    \\left[\\frac{1.5\\,\\phi(\\Phi^{-1}(\\tau))^2}{2\\,\\Phi^{-1}(\\tau)^2 + 1}\\right]^{1/3},
```

where `τ` is the probability level and `n` the (effective) sample size
(see [`QuantileSpacing`](@ref)). `alpha` is the nominal size of the confidence intervals
the rule targets; it must be in ``(0, 1)``.

Hall and Sheather derived this rule from Edgeworth expansions for studentized quantiles,
i.e. for the coverage of confidence intervals, which is what influence values are mostly
used for. Bofinger's ``n^{-1/5}`` rule instead minimizes the mean squared error of the
density estimate itself; Koenker (2005) recommends Hall–Sheather for inference.

# References

  - Hall, P. and Sheather, S. J. (1988). On the distribution of a studentized quantile.
    *Journal of the Royal Statistical Society B* 50(3), 381–391.
  - Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
"""
struct HallSheatherBandwidth{T <: Real}
    alpha::T
    function HallSheatherBandwidth(alpha::T) where {T <: Real}
        0 < alpha < 1 || throw(DomainError(alpha, "alpha must be in (0, 1)"))
        return new{T}(alpha)
    end
end
HallSheatherBandwidth(; alpha::Real = 0.05) = HallSheatherBandwidth(alpha)

function _bandwidth(b::HallSheatherBandwidth, τ, n)
    α = oftype(float(τ), b.alpha)
    z = StatsFuns.norminvcdf(τ)
    zα = StatsFuns.norminvcdf(1 - α / 2)
    return cbrt(3 * StatsFuns.normpdf(z)^2 * zα^2 / (2z^2 + 1) / 2n)
end
_bandwidth(h::Real, τ, n) = h

"""
    QuantileSpacing(; bandwidth = HallSheatherBandwidth())

Siddiqui's (1960) sparsity estimate of a density, as presented in Koenker (2005). At a
probability level `τ`, the density is estimated as

```math
\\hat f = \\frac{2h}{\\tilde F_n^{-1}(\\tau + h) - \\tilde F_n^{-1}(\\tau - h)},
```

where ``\\tilde F_n^{-1}`` is Koenker's piecewise-linear empirical quantile function, which
interpolates the order statistics at the levels ``(2j - 1)/(2n)``. `bandwidth` is a
[`HallSheatherBandwidth`](@ref) or a fixed half-width `h` in ``(0, 0.5)`` on the
probability scale.

The level `τ` is:

  - for a quantile the estimator computes itself, that quantile's nominal level (as in
    Koenker), e.g. `p` for `quantile(x, p)`;
  - for any other point `t`, the mid-rank empirical CDF level
    ``(\\hat F_n(t^-) + \\hat F_n(t))/2``. This choice is the package's own: it equals
    ``\\hat F_n(t)`` away from ties and treats tied minima and maxima alike.

Also the package's own choices: the density is 0 outside the range of the data, levels are
clamped to ``[1/(2n), 1 - 1/(2n)]``, where ``\\tilde F_n^{-1}`` is not flat, and a window that
lies inside an atom of the data gives an infinite density, so a quantile inside an atom has
zero influence. Near the extremes, `h` is halved until
``[\\tau - h, \\tau + h] \\subseteq [0, 1]``, as in R's quantreg.

It is the default because it needs only the sorted data and a closed-form bandwidth, and
unlike a kernel density estimate it stays accurate near the boundary of the support and on
skewed data. Fitting sorts a copy of unweighted data; with weights, it sorts indices and
retains a view of the data.

# Weights

With a vector of weights `w` (see [`fit_density`](@ref)), the estimate is of the
distribution that puts mass ``\\tilde w_i = w_i / \\sum_j w_j`` on ``x_i``. Rows with zero
weight are dropped. Sorted, row ``j`` spans the levels from ``S_{j-1}`` to ``S_j``, the
cumulative masses. The package's own choices, which all reduce to the above for uniform
weights:

  - ``\\tilde F_n^{-1}`` interpolates the sorted data at the middles of their spans,
    ``(S_{j-1} + S_j)/2``, and levels are clamped to the first and last of these;
  - the mid-rank level of a point uses the weighted empirical CDF;
  - `n` in [`HallSheatherBandwidth`](@ref) is Korn and Graubard's (1998) effective sample
    size for the mass below `τ`, ``\\tau(1 - \\tau)/V``, where
    ``V = (1 - \\tau)^2 A + \\tau^2 (B - A)``, ``A`` is the sum of ``\\tilde w_i^2`` over the
    rows below `τ` and ``B`` over all rows. The row that spans `τ` counts in proportion to
    its mass below `τ`. Unlike Kish's ``1/B``, this adapts to where the weight lies: at
    the median it equals Kish's, and in a tail it reflects the weights in that tail.

`StatsBase.FrequencyWeights` count repeats instead, and the estimate equals the one for
the data with each row repeated `w[i]` times.

# References

  - Siddiqui, M. M. (1960). Distribution of quantiles in samples from a bivariate
    population. *Journal of Research of the National Bureau of Standards B* 64, 145–150.
  - Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
  - Korn, E. L. and Graubard, B. I. (1998). Confidence intervals for proportions with small
    expected number of positive counts estimated from survey data. *Survey Methodology*
    24(2), 193–201.
"""
struct QuantileSpacing{B}
    bandwidth::B
    function QuantileSpacing(bandwidth::B) where {B}
        if bandwidth isa Real && !(0 < bandwidth < 0.5)
            throw(DomainError(bandwidth, "a fixed bandwidth must be in (0, 0.5)"))
        end
        return new{B}(bandwidth)
    end
end
QuantileSpacing(; bandwidth = HallSheatherBandwidth()) = QuantileSpacing(bandwidth)

fit_density(est::QuantileSpacing, x) = _SpacingDensity(sort(vec(x)), est.bandwidth, nothing)

struct _SpacingDensity{V <: AbstractVector, B, D}
    xs::V    # sorted, 1-based
    bandwidth::B
    data::D  # nothing for unweighted data; cumulative masses and knots for weighted data
end

function (g::_SpacingDensity)(t)
    (; xs, data) = g
    T = float(_density_result_type(data, xs))
    (t < first(xs) || t > last(xs)) && return zero(T)    # no data out here
    τ = T(_midrank(data, xs, t))
    return _density_at_level(g, τ)
end

function _midrank(::Nothing, xs, t)
    i1 = firstindex(xs)
    ixmin = searchsortedfirst(xs, t) - i1 + 1
    ixmax = searchsortedlast(xs, t) - i1 + 1
    return (ixmin - 1 + ixmax) // (2 * length(xs))
end

_density_at_quantile(g::_SpacingDensity, q, τ) = _density_at_level(g, τ)

function _density_at_level(g::_SpacingDensity, τ)
    (; xs, bandwidth, data) = g
    n = length(xs)
    T = float(promote_type(typeof(τ), _density_result_type(data, xs)))
    isnan(last(xs)) && return T(NaN)    # sorting puts NaNs last
    # In at least Float64: in a narrower type 1 - 1/(2n) can round to 1, where the bandwidth
    # is 0, and Float16 `norminvcdf` is inaccurate.
    S = promote_type(T, Float64)
    τ = _clamp_level(data, S(τ), n)
    h = S(_bandwidth(bandwidth, τ, _effective_size(data, τ, n)))
    isinf(h) && (h = one(h))    # halving would never shrink it
    while τ - h < 0 || τ + h > 1
        h /= 2
    end
    return T(2h / (_spacing_quantile(data, xs, τ + h) - _spacing_quantile(data, xs, τ - h)))
end

_clamp_level(::Nothing, τ, n) = clamp(τ, 1 // (2n), 1 - 1 // (2n))
_density_result_type(::Nothing, xs) = eltype(xs)
_effective_size(::Nothing, τ, n) = n
function _spacing_quantile(::Nothing, xs, u)  # Koenker's F̃⁻¹
    return Statistics.quantile(xs, u; sorted = true, alpha = 0.5, beta = 0.5)
end

# Sorted weighted rows span cumulative mass intervals. The quantile function has a knot
# at each interval's midpoint; squared masses supply the level-specific effective size.
struct _WeightedSpacingData{V <: AbstractVector, M <: AbstractVector, T}
    cmass::M
    csqmass::M
    levels::M
    knots::V
    result_type::Type{T}
end

_density_result_type(data::_WeightedSpacingData, xs) = data.result_type

function fit_density(est::QuantileSpacing, x, w)
    length(x) == length(w) || throw(DimensionMismatch("data and weights must have the same length"))
    V = promote_type(eltype(x), eltype(w))
    M = float(eltype(w))
    perm = similar(x, Int)
    sortperm!(perm, eachindex(x, w); by = i -> (x[i], w[i]))
    filter!(i -> !iszero(w[i]), perm)
    xs = view(x, perm)
    ws = view(w, perm)
    W = M(sum(w))
    cmass = similar(ws, M, length(ws) + 1)
    csqmass = similar(cmass)
    cmass[begin] = csqmass[begin] = 0
    for (j, wⱼ) in enumerate(ws)
        mⱼ = M(wⱼ / W)
        cmass[j + 1] = cmass[j] + mⱼ
        csqmass[j + 1] = csqmass[j] + M(_squared_mass(w, mⱼ, W))
    end
    levels, knots = _spacing_knots(w, xs, ws, cmass, W)
    data = _WeightedSpacingData(cmass, csqmass, levels, knots, V)
    return _SpacingDensity(xs, est.bandwidth, data)
end

_squared_mass(w, m, W) = m^2
_spacing_knots(w, xs, ws, cmass, W) = (_midpoints(cmass), xs)
_midpoints(x) = @views (x[begin:(end - 1)] .+ x[(begin + 1):end]) ./ 2

function _midrank(data::_WeightedSpacingData, xs::AbstractVector, t)
    (; cmass) = data
    i1 = firstindex(xs)
    below = cmass[searchsortedfirst(xs, t) - i1 + 1]
    upto = cmass[searchsortedlast(xs, t) - i1 + 2]
    return (below + upto) / 2
end

function _clamp_level(data::_WeightedSpacingData, τ, n)
    return clamp(τ, first(data.levels), last(data.levels))
end

function _effective_size(data::_WeightedSpacingData, τ, n)
    (; cmass, csqmass) = data
    A = _interpolate(cmass, csqmass, τ)
    return τ * (1 - τ) / ((1 - τ)^2 * A + τ^2 * (last(csqmass) - A))
end

function _spacing_quantile(data::_WeightedSpacingData, xs, u)
    return _interpolate(data.levels, data.knots, u)
end

# Piecewise-linear interpolation, constant outside the knots.
function _interpolate(us::AbstractVector, vs::AbstractVector, u)
    u <= first(us) && return first(vs) + zero(u)
    u >= last(us) && return last(vs) + zero(u)
    j = searchsortedlast(us, u)
    return vs[j] + (u - us[j]) / (us[j + 1] - us[j]) * (vs[j + 1] - vs[j])
end
