using StatsBase

@testset "proportions" begin
    k = rand(StableRNG(6), 1:4, N)
    l = rand(StableRNG(7), 1:3, N)
    v, ψ = value_and_influence(proportions, k, 1:4)
    @test v == proportions(k, 1:4)
    @test size(ψ) == size(v)
    for c in 1:4
        check_against_ad(p -> wmean(p, k .== c), (v[c], ψ[c]), v[c])
    end
    @test value_and_influence(proportions, k) == (v, ψ)
    @test value_and_influence(proportions, k, 4) == (v, ψ)
    @test first(influence(proportions, k .+ 10)) == ψ[1]    # levels from the data
    v, ψ = value_and_influence(proportions, k, 2:3)         # a level is a level, wherever it is
    @test v == proportions(k, 2:3)
    @test ψ[1] == (k .== 2) .- v[1]

    v, ψ = value_and_influence(proportions, k, l, (1:4, 1:3))
    @test v == proportions(k, l, (1:4, 1:3))
    @test size(ψ) == size(v)
    for I in CartesianIndices(v)
        check_against_ad(p -> wmean(p, (k .== I[1]) .& (l .== I[2])), (v[I], ψ[I]), v[I])
    end
    @test value_and_influence(proportions, k, l) == (v, ψ)
    @test value_and_influence(proportions, k, l, (4, 3)) == (v, ψ)
    @test influence(proportions, k, l, 3) == ψ[1:3, 1:3]
    odds(q) = (q[1, 1] * q[2, 2]) / (q[1, 2] * q[2, 1])
    T(p) = odds([wmean(p, (k .== a) .& (l .== b)) for a in 1:4, b in 1:3])
    @test sum(ForwardDiff.gradient(odds, v) .* ψ) ≈ ad_influence(T, N) rtol = 1.0e-10

    km = reshape(k, 9, 4)                                   # the data's axes on the inside
    @test size(influence(proportions, km)[1]) == size(km)
    ko = OffsetArray(km, -2, 5)
    @test axes(influence(proportions, ko)[1]) == axes(ko)
    kd = DimArray(km, (Dim{:draw}(1:9), Dim{:chain}(1:4)))
    @test dims(influence(proportions, kd)[1]) == dims(kd)
    @inferred value_and_influence(proportions, k)
    @inferred value_and_influence(proportions, k, 2:3)
    @inferred value_and_influence(proportions, k, l)
    @inferred value_and_influence(proportions, k, l, (4, 3))
    @test_throws MethodError influence(proportions, float.(k))      # as in StatsBase
    @test_throws MethodError influence(proportions, k, l, 1:3)      # as in StatsBase
    @test_throws DimensionMismatch influence(proportions, k, l[1:3])
end

@testset "weighted proportions" begin
    k = rand(StableRNG(6), 1:4, N)
    l = rand(StableRNG(7), 1:3, N)
    for W in (fweights, pweights, aweights, weights)
        w = W(WV)
        v, ψ = value_and_influence(proportions, k, 1:4, w)
        @test v == proportions(k, 1:4, w)
        @test size(ψ) == size(v)
        for c in 1:4
            check_weighted_against_ad(u -> proportions(k, 1:4, W(u))[c], (v[c], ψ[c]), WV)
        end
        # Integer weights are integer arrays too, but they are weights, not a second variable.
        @test value_and_influence(proportions, k, w) == (v, ψ)
        @test value_and_influence(proportions, k, 4, w) == (v, ψ)

        v, ψ = value_and_influence(proportions, k, l, (1:4, 1:3), w)
        @test v == proportions(k, l, (1:4, 1:3), w)
        @test size(ψ) == size(v)
        for I in CartesianIndices(v)
            check_weighted_against_ad(u -> proportions(k, l, (1:4, 1:3), W(u))[I], (v[I], ψ[I]), WV)
        end
        @test value_and_influence(proportions, k, l, w) == (v, ψ)
        @test value_and_influence(proportions, k, l, (4, 3), w) == (v, ψ)
        @test influence(proportions, k, l, 3, w) == ψ[1:3, 1:3]
    end
    w = pweights(WV)
    @test value_and_influence(proportions, k, uweights(N)) == value_and_influence(proportions, k)
    @test value_and_influence(proportions, k, l, uweights(N)) == value_and_influence(proportions, k, l)
    km = reshape(k, 9, 4)                                   # weights over all elements
    @test vec(influence(proportions, km, w)[1]) == influence(proportions, k, w)[1]
    @inferred value_and_influence(proportions, k, w)
    @inferred value_and_influence(proportions, k, l, w)
    @test_throws DimensionMismatch influence(proportions, k, w[1:5])
end
