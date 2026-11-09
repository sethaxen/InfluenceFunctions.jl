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
    @test_throws MethodError influence(scattermat, M; density = nothing)
end

@testset "weighted scattermat" begin
    for W in (fweights, pweights, aweights, weights), c in (nothing, 0, [1.0 2.0 3.0])
        S, ψ = value_and_influence(scattermat, M, W(WV); mean = c)
        @test S == scattermat(M, W(WV); mean = c)
        @test size(ψ) == size(S)
        for I in CartesianIndices(S)
            check_weighted_against_ad(u -> scattermat(M, W(u); mean = c)[I], (S[I], ψ[I]), WV)
        end
    end
    w = pweights(WV)
    S, ψ = value_and_influence(scattermat, M, w)
    St, ψt = value_and_influence(scattermat, permutedims(M), w; dims = 2)   # variables as rows
    @test St ≈ S
    @test all(ψt .≈ ψ)
    @test all(ψ .≈ (sum(WV) - 1) .* influence(cov, M, fweights(WV); corrected = true))
    @test all(influence(scattermat, M, uweights(N)) .≈ influence(scattermat, M))
    @inferred value_and_influence(scattermat, M, w)
    @inferred value_and_influence(scattermat, permutedims(M), w; mean = 0, dims = 2)
    @test_throws DimensionMismatch influence(scattermat, M, w[1:5])
end

@testset "weighted cov" begin
    for W in (fweights, pweights, aweights, weights), corrected in (false, true), dims in (1, 2)
        corrected && (W === aweights || W === weights) && continue   # see "weighted var"
        x = dims == 1 ? M : permutedims(M)
        C, ψ = value_and_influence(cov, x, W(WV), dims; corrected)
        @test C == cov(x, W(WV), dims; corrected)
        @test size(ψ) == size(C)
        for I in CartesianIndices(C)
            check_weighted_against_ad(u -> cov(x, W(u), dims; corrected)[I], (C[I], ψ[I]), WV)
        end
    end
    aw = aweights(WV)
    b = sum(WV) / (sum(WV) - sum(abs2, WV) / sum(WV))
    @test all(influence(cov, M, aw; corrected = true) .≈ b .* influence(cov, M, aw; corrected = false))
    ψ = influence(cov, M, fweights(WV); corrected = true)  # repeats of an observation
    @test inverse_rle(ψ[1, 2], WV) ≈ influence(cov, M[inverse_rle(1:N, WV), :]; corrected = true)[1, 2]
    w = pweights(WV)
    @test influence(cov, M, w; corrected = true)[2, 2] ≈ influence(var, M[:, 2], w; corrected = true)
    @test all(influence(cov, M, uweights(N); corrected = true) .≈ influence(cov, M; corrected = true))
    @test (@test_deprecated r"default to corrected=true" influence(cov, M, w)) == influence(cov, M, w; corrected = false)
    @inferred value_and_influence(cov, M, w; corrected = true)
    @inferred value_and_influence(cov, permutedims(M), w, 2; corrected = true)
    @test_throws ArgumentError influence(cov, M, weights(WV); corrected = true)   # as in StatsBase
    @test_throws DimensionMismatch influence(cov, M, w[1:5])
end

@testset "weighted cor" begin
    for W in (fweights, pweights, aweights, weights), dims in (1, 2)
        x = dims == 1 ? M : permutedims(M)
        R, ψ = value_and_influence(cor, x, W(WV), dims)
        @test R == cor(x, W(WV), dims)
        @test size(ψ) == size(R)
        for I in CartesianIndices(R)
            if I[1] == I[2]                                 # 1 for any data
                @test maximum(abs, ψ[I]) < 1.0e-12
            else
                G = u -> cov2cor(cov(x, W(u), dims; corrected = false))[I]
                check_weighted_against_ad(G, (R[I], ψ[I]), WV)
            end
        end
    end
    @test all(influence(cor, M, uweights(N)) .≈ influence(cor, M))
    @inferred value_and_influence(cor, M, pweights(WV))
    @inferred value_and_influence(cor, permutedims(M), pweights(WV), 2)
    @test_throws DimensionMismatch influence(cor, M, pweights(WV[1:5]))
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
end

@testset "weighted mean_and_cov" begin
    w = pweights(WV)
    for dims in (1, 2), corrected in (true, false)
        x = dims == 1 ? M : permutedims(M)
        v, ψ = value_and_influence(mean_and_cov, x, w, dims; corrected)
        @test v == mean_and_cov(x, w, dims; corrected)
        @test ψ[1] == influence(mean, x, w; dims)
        @test ψ[2] == influence(cov, x, w, dims; corrected)
    end
    @test value_and_influence(mean_and_cov, M, w; corrected = true) == value_and_influence(mean_and_cov, M, w, 1; corrected = true)
    @test (@test_deprecated r"default to corrected=true" value_and_influence(mean_and_cov, M, w)) == value_and_influence(mean_and_cov, M, w; corrected = false)
    @inferred value_and_influence(mean_and_cov, M, w; corrected = true)
    @test_throws MethodError influence(mean_and_cov, M, w; dims = 1)
end
