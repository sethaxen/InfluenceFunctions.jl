using Statistics, ForwardDiff, Distributions, OffsetArrays, StableRNGs
using DimensionalData: DimArray, Dim, dims   # not `using DimensionalData`: it exports `X`

const N = 36                                   # divisible by 4 and 12 for reshape tests
const X = rand(StableRNG(1), Gamma(3, 2), N)   # positive, skewed
const C = 5.0                                  # fixed center for fixed-center estimators
const D = Gamma(3, 2)                          # known density for quantile-type tests
const M = [X (X .+ randn(StableRNG(2), N)) randn(StableRNG(3), N)]   # variables as columns

wmean(p, v) = sum(p .* v)

# Covariance matrix of the columns of M under probability weights p, scaled by b.
function wcov(p, b = 1.0)
    Z = M .- sum(p .* M; dims = 1)
    return b .* (Z' * (p .* Z))
end

# Influence values by forward-mode AD: the derivative of T, a function of probability
# weights p, as weight ε moves onto observation i.
function ad_influence(T, n)
    p0 = fill(1 / n, n)
    return [ForwardDiff.derivative(ε -> T((1 - ε) .* p0 .+ ε .* ((1:n) .== i)), 0.0) for i in 1:n]
end

# Smooth estimators: value, T at uniform weights, match with AD, sum-to-zero.
function check_against_ad(T, (v, ψ), v_ref)
    n = length(ψ)
    @test v ≈ v_ref
    @test T(fill(1 / n, n)) ≈ v_ref
    @test ψ ≈ ad_influence(T, n) rtol = 1.0e-10
    @test abs(sum(ψ)) ≤ 1.0e-10 * sum(abs, ψ)
    return nothing
end

# `@inferred` reports `Union{}` when empty keyword arguments are splatted into a method
# that accepts none, so pass keywords only when there are some.
function inferred_value_and_influence(f, x, args...; kwargs...)
    isempty(kwargs) && return @inferred value_and_influence(f, x, args...)
    return @inferred value_and_influence(f, x, args...; kwargs...)
end

# Shapes and types: ψ keeps the data's array type and axes, equals the flattened
# result, and has the floating-point type of the estimator's value. With `vector_only`
# (for estimators that, like `Statistics.quantile`, accept only vectors), only vector
# inputs are checked.
function check_shapes(f, x::AbstractVector, args...; vector_only = false, kwargs...)
    v, ψ = inferred_value_and_influence(f, x, args...; kwargs...)
    @test eltype(ψ) === typeof(v)
    if vector_only
        xd = DimArray(x, Dim{:draw}(1:length(x)))
        ψd = influence(f, xd, args...; kwargs...)
        @test ψd isa DimArray
        @test dims(ψd) == dims(xd)
        @test parent(ψd) ≈ ψ
    else
        for sz in ((N ÷ 4, 4), (N ÷ 12, 4, 3))
            vX, ψX = value_and_influence(f, reshape(x, sz), args...; kwargs...)
            @test vX ≈ v
            @test size(ψX) == sz
            @test vec(ψX) ≈ ψ
        end
        xo = OffsetArray(reshape(x, N ÷ 4, 4), -2, 5)
        ψo = influence(f, xo, args...; kwargs...)
        @test ψo isa OffsetArray
        @test axes(ψo) == axes(xo)
        @test vec(parent(ψo)) ≈ ψ
        xd = DimArray(reshape(x, N ÷ 4, 4), (Dim{:draw}(1:(N ÷ 4)), Dim{:chain}(1:4)))
        ψd = influence(f, xd, args...; kwargs...)
        @test ψd isa DimArray
        @test dims(ψd) == dims(xd)
        @test vec(parent(ψd)) ≈ ψ
    end
    # All-Float32 inputs. With mixed precision (Float32 data, Float64 center) Statistics.var
    # itself infers a Union, so inference is only checked for matching precisions.
    f32(a) = a isa AbstractFloat ? Float32(a) : a
    v32, ψ32 = inferred_value_and_influence(f, Float32.(x), map(f32, args)...; map(f32, values(kwargs))...)
    @test eltype(ψ32) === typeof(v32)
    @test isapprox(ψ32, ψ; rtol = 1.0e-3)
    return nothing
end

# `dims`: the value is `f(x; dims)`, ψ has the axes of `x`, and the influence values of each
# slice equal those of `f` applied to that slice alone. `extra` holds keywords that only
# `value_and_influence` takes (e.g. `density`).
function check_dims(f, x::AbstractArray; dims, extra = (;), kwargs...)
    v, ψ = value_and_influence(f, x; dims, extra..., kwargs...)
    @test v ≈ f(x; dims, kwargs...)
    @test axes(ψ) == axes(x)
    reduced = dims isa Colon ? (1:ndims(x)) : dims
    for I in CartesianIndices(axes(v))   # a single empty index when `v` is a scalar
        idx = ntuple(d -> d in reduced ? Colon() : I[d], ndims(x))
        @test vec(ψ[idx...]) ≈ influence(f, vec(x[idx...]); extra..., kwargs...)
    end
    return nothing
end

# A user-defined density estimator: any type plus a fit_density method, no supertype.
struct ConstantDensity
    c::Float64
end
InfluenceFunctions.fit_density(d::ConstantDensity, x) = Returns(d.c)

# A user-defined density estimator that counts how many times it is fit.
struct CountingDensity
    fits::Base.RefValue{Int}
end
function InfluenceFunctions.fit_density(d::CountingDensity, x)
    d.fits[] += 1
    return Returns(0.5)
end
