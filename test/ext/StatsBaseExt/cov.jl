using StatsBase

@testset "scattermat" begin
    function T(p, c)
        Z = M .- c
        return N .* (Z' * (p .* Z))
    end
    S, ψ = value_and_influence(scattermat, M)
    @test S == scattermat(M)
    @test size(ψ) == size(S)
    for I in CartesianIndices(S)
        check_against_ad(p -> T(p, sum(p .* M; dims = 1))[I], (S[I], ψ[I]), S[I])
    end
    @test all(ψ .≈ (N - 1) .* influence(cov, M))
    for c in (0, 1.5, [1.0 2.0 3.0])                        # a known center
        Sc, ψc = value_and_influence(scattermat, M; mean = c)
        @test Sc == scattermat(M; mean = c)
        for I in CartesianIndices(Sc)
            check_against_ad(p -> T(p, c)[I], (Sc[I], ψc[I]), Sc[I])
        end
    end
    St, ψt = value_and_influence(scattermat, permutedims(M); dims = 2)   # variables as rows
    @test St ≈ S
    @test all(ψt .≈ ψ)
    @test all(influence(scattermat, permutedims(M); dims = 2, mean = [1.0, 2.0, 3.0]) .≈ influence(scattermat, M; mean = [1.0 2.0 3.0]))
    @test eltype(influence(scattermat, Float32.(M))[1, 2]) === Float32
    @inferred value_and_influence(scattermat, M)
    @inferred value_and_influence(scattermat, M; dims = 2)
    @inferred value_and_influence(scattermat, M; mean = 0)
    @test_throws ArgumentError influence(scattermat, M; dims = 3)   # accepted by StatsBase.scattermat
    # StatsBase.scattermat takes only a DenseMatrix.
    @test_throws MethodError influence(scattermat, DimArray(M, (Dim{:draw}(1:N), Dim{:param}(1:3))))
    @test_throws MethodError influence(scattermat, M, pweights(ones(N)))
    @test_throws MethodError influence(scattermat, M; density = nothing)
end

@testset "weighted cov and cor" begin
    w = fweights(ones(Int, N))
    @test_throws MethodError influence(cov, M, w)
    @test_throws MethodError influence(cor, M, w)
end

@testset "mean_and_cov" begin
    for dims in (1, 2), corrected in (true, false)
        x = dims == 1 ? M : permutedims(M)
        v, ψ = value_and_influence(mean_and_cov, x, dims; corrected)
        @test v == mean_and_cov(x, dims; corrected)
        @test ψ[1] == influence(mean, x; dims)
        @test ψ[2] == influence(cov, x; dims, corrected)
    end
    @test value_and_influence(mean_and_cov, M) == value_and_influence(mean_and_cov, M, 1)
    # Rows are the observations, so the mean's influence values are summed across columns.
    (m, S), (ψm, ψS) = value_and_influence(mean_and_cov, M)
    gm = ForwardDiff.gradient(w -> only(w * (S \ w')), m)
    gS = ForwardDiff.gradient(W -> only(m * (W \ m')), S)
    function T(p)
        c = sum(p .* M; dims = 1)
        return only(c * (wcov(p, N / (N - 1)) \ c'))
    end
    @test vec(sum(gm .* ψm; dims = 2)) .+ sum(gS .* ψS) ≈ ad_influence(T, N) rtol = 1.0e-10
    @inferred value_and_influence(mean_and_cov, M)
    # StatsBase.mean_and_cov takes only a DenseMatrix, with the dimension given positionally.
    @test_throws MethodError influence(mean_and_cov, DimArray(M, (Dim{:draw}(1:N), Dim{:param}(1:3))))
    @test_throws MethodError influence(mean_and_cov, M; dims = 1)
    @test_throws MethodError influence(mean_and_cov, M, pweights(ones(N)))
end
