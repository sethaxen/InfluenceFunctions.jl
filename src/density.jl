"""
    fit_density(density, x) -> g

Fit `density` to the data `x`, returning a callable `g` with `g(t)` ≈ the density of the
data at `t`. Quantile-type estimators call this once per call, on their own data, and
evaluate `g` wherever their influence function needs a density.

`density` may be:

  - a density estimator, such as [`QuantileSpacing`](@ref), the default;
  - an object implementing DensityInterface.jl, such as a Distributions.jl distribution,
    evaluated with `DensityInterface.densityof`. Its density must be with respect to
    Lebesgue measure;
  - a `Function` `t -> f(t)` giving a known density.

Anything else throws an `ArgumentError`.

To add a density estimator, define a type and a method
`InfluenceFunctions.fit_density(est::MyEstimator, x)` that returns such a callable. No
supertype is needed.
"""
function fit_density(density, x)
    if DensityInterface.DensityKind(density) !== DensityInterface.NoDensity()
        return Base.Fix1(DensityInterface.densityof, density)
    elseif density isa Function
        return density
    end
    throw(
        ArgumentError(
            "`density` must be a density estimator with a `fit_density` method (e.g. " *
                "`QuantileSpacing()`), an object implementing DensityInterface.jl, or a " *
                "`Function`; got $(repr(density))",
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

where `τ` is the probability level and `n` the number of observations. `alpha` is the
nominal size of the confidence intervals the rule targets; it must be in ``(0, 1)``.

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
skewed data. Fitting sorts a copy of the data.

# References

  - Siddiqui, M. M. (1960). Distribution of quantiles in samples from a bivariate
    population. *Journal of Research of the National Bureau of Standards B* 64, 145–150.
  - Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
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

fit_density(est::QuantileSpacing, x) = _SpacingDensity(sort!(vec(collect(x))), est.bandwidth)

struct _SpacingDensity{V <: AbstractVector, B}
    xs::V    # sorted, 1-based
    bandwidth::B
end

function (g::_SpacingDensity)(t)
    (; xs) = g
    T = float(eltype(xs))
    (t < first(xs) || t > last(xs)) && return zero(T)    # no data out here
    ixmin = searchsortedfirst(xs, t)
    ixmax = searchsortedlast(xs, t)
    τ = T((ixmin - 1 + ixmax) // (2 * length(xs)))    # mid-rank level
    return _density_at_level(g, τ)
end

_density_at_quantile(g::_SpacingDensity, q, τ) = _density_at_level(g, τ)

function _density_at_level(g::_SpacingDensity, τ)
    (; xs, bandwidth) = g
    n = length(xs)
    T = float(promote_type(typeof(τ), eltype(xs)))
    isnan(last(xs)) && return T(NaN)    # sorting puts NaNs last
    # In at least Float64: in a narrower type 1 - 1/(2n) can round to 1, where the bandwidth
    # is 0, and Float16 `norminvcdf` is inaccurate.
    S = promote_type(T, Float64)
    τ = clamp(S(τ), 1 // (2n), 1 - 1 // (2n))
    h = S(_bandwidth(bandwidth, τ, n))
    isinf(h) && (h = one(h))    # halving would never shrink it
    while τ - h < 0 || τ + h > 1
        h /= 2
    end
    Q(u) = Statistics.quantile(xs, u; sorted = true, alpha = 0.5, beta = 0.5)    # Koenker's F̃⁻¹
    return T(2h / (Q(τ + h) - Q(τ - h)))
end
