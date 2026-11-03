@testset "mean(x)" begin
    check_against_ad(p -> wmean(p, X), value_and_influence(mean, X), mean(X))
    check_shapes(mean, X)
    @test influence(mean, [1, 2, 6]) == [-2.0, -1.0, 3.0]       # Int data → Float64
    @test influence(mean, 1.0:3.0) == [-1.0, 0.0, 1.0]          # ranges work
    v, ψ = value_and_influence(mean, Float64[])
    @test isnan(v) && ψ == Float64[]
    A, B = reshape(X, 9, 4), reshape(X, 3, 4, 3)
    for (x, dims) in ((A, 1), (A, 2), (A, (1, 2)), (A, :), (B, (1, 3)), (B, 2))
        check_dims(mean, x; dims)
    end
    @inferred value_and_influence(mean, A; dims = 1)
    @test_throws MethodError influence(mean, X; density = nothing)   # strict keywords
    @test_throws MethodError influence(mean, [1.0, missing])         # missing is unsupported
end

@testset "mean(f, x)" begin
    check_against_ad(p -> wmean(p, sqrt.(X)), value_and_influence(mean, sqrt, X), mean(sqrt, X))
    @inferred value_and_influence(mean, sqrt, X)
    @test size(influence(mean, abs2, reshape(X, 9, 4))) == (9, 4)
    @test eltype(influence(mean, t -> t // 1, [1, 2, 6])) === Rational{Int}   # no forced float
    A = reshape(X, 9, 4)
    v, ψ = value_and_influence(mean, sqrt, A; dims = 1)
    @test v ≈ mean(sqrt, A; dims = 1)
    @test size(ψ) == size(A)
    @test ψ[:, 2] ≈ influence(mean, sqrt, A[:, 2])
end
