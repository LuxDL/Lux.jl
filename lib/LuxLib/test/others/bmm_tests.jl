# Most of the tests in this file were derived from https://github.com/FluxML/NNlib.jl/blob/master/test/batchedmul.jl
@testsetup module BatchedMMSetup

using NNlib

function bmm_test(a, b; transA=false, transB=false)
    bs = size(a, 3)
    transA && (a = permutedims(a, [2, 1, 3]))
    transB && (b = permutedims(b, [2, 1, 3]))
    c = []
    for i in 1:bs
        push!(c, a[:, :, i] * b[:, :, i])
    end

    return cat(c...; dims=3)
end

function bmm_adjtest(a, b; adjA=false, adjB=false)
    bs = size(a, 3)
    c = []
    for i in 1:bs
        ai = adjA ? adjoint(a[:, :, i]) : a[:, :, i]
        bi = adjB ? adjoint(b[:, :, i]) : b[:, :, i]
        push!(c, ai * bi)
    end

    return cat(c...; dims=3)
end

function half_batched_mul(x, y)
    @assert size(y, 3) == 1
    d = size(x, 2)
    x_mat = reshape(permutedims(x, (1, 3, 2)), :, d)
    y_mat = reshape(y, d, :)
    z_mat = x_mat * y_mat
    return permutedims(reshape(z_mat, size(x, 1), size(x, 3), :), (1, 3, 2))
end

perm_12(A) = PermutedDimsArray(A, (2, 1, 3))
perm_23(A) = PermutedDimsArray(A, (1, 3, 2))

export bmm_test, bmm_adjtest, half_batched_mul, perm_12, perm_23

end

@testitem "batched_mul" tags=[:batched_ops] setup=[SharedTestSetup, BatchedMMSetup] begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "batched_mul: Float64 × $(TB)" for TB in [Float64, Float32]
            @testset "real" begin
                A = randn(rng, 7, 5, 3) |> aType
                B = randn(rng, TB, 5, 7, 3) |> aType
                C = randn(rng, 7, 6, 3) |> aType

                @test batched_matmul(A, B) ≈ bmm_test(A, B)
                @test batched_matmul(batched_transpose(A), batched_transpose(B)) ≈
                      bmm_test(A, B; transA=true, transB=true)
                @test batched_matmul(batched_transpose(A), C) ≈ bmm_test(A, C; transA=true)
                @test batched_matmul(A, batched_transpose(A)) ≈ bmm_test(A, A; transB=true)
            end

            @testset "complex" begin
                cA = randn(rng, Complex{Float64}, 7, 5, 3) |> aType
                cB = randn(rng, Complex{TB}, 5, 7, 3) |> aType
                cC = randn(rng, Complex{Float64}, 7, 6, 3) |> aType

                @test batched_matmul(cA, cB) ≈ bmm_adjtest(cA, cB)
                @test batched_matmul(batched_adjoint(cA), batched_adjoint(cB)) ≈
                      bmm_adjtest(cA, cB; adjA=true, adjB=true)
                @test batched_matmul(batched_adjoint(cA), cC) ≈
                      bmm_adjtest(cA, cC; adjA=true)
                @test batched_matmul(cA, batched_adjoint(cA)) ≈
                      bmm_adjtest(cA, cA; adjB=true)

                @testset "Integers" begin
                    TBi = TB == Float64 ? Int64 : Int32
                    iA = rand(rng, 1:99, 7, 5, 3) |> aType
                    iB = TB.(rand(rng, 1:99, 5, 7, 3)) |> aType
                    iC = zeros(Int, 7, 6, 3) |> aType

                    @test batched_matmul(iA, iB) == bmm_adjtest(iA, iB)
                    @test batched_matmul(cA, iB) ≈ bmm_adjtest(cA, iB)
                end
            end

            @testset "Errors" begin
                @test_throws DimensionMismatch batched_matmul(
                    aType(rand(rng, 2, 2, 2)), aType(rand(rng, TB, 2, 2, 10)))
                @test_throws DimensionMismatch batched_matmul(
                    aType(rand(rng, 2, 2, 2)), aType(rand(rng, TB, 10, 2, 2)))
                @test_throws Exception batched_mul!(
                    aType(zeros(2, 2, 10)), aType(rand(rng, 2, 2, 2)),
                    aType(rand(rng, TB, 2, 2, 2)))
            end

            @testset "PermutedDimsArrays" begin
                if !ongpu
                    for perm in [(1, 3, 2), (2, 1, 3), (3, 2, 1)],
                        fun in [identity, batched_adjoint],
                        ty in [identity, complex]

                        A = randn(rng, ty(Float64), 4, 4, 4) |> aType
                        B = randn(rng, ty(TB), 4, 4, 4) |> aType

                        @test batched_matmul(fun(A), PermutedDimsArray(B, perm)) ≈
                              batched_matmul(fun(A), permutedims(B, perm))
                        @test batched_matmul(fun(PermutedDimsArray(A, perm)), B) ≈
                              batched_matmul(fun(permutedims(A, perm)), B)
                    end
                end
            end

            @testset "PermutedDimsArray output" begin
                A′ = randn(rng, 4, 3, 2) |> aType
                B′ = batched_adjoint(randn(rng, TB, 5, 3, 2)) |> aType
                C1 = batched_matmul(A′, B′) # size 4,5,2
                C2 = PermutedDimsArray(zeros(5, 2, 4), (3, 1, 2)) |> aType # size 4,5,2

                @test C1 ≈ batched_mul!(C2, A′, B′) # Float64: "Debug: transposing C = A * B into Cᵀ = Bᵀ * Aᵀ"
                @test C1 ≈ C2

                @testset "Trivial batches for B" begin
                    D′ = randn(rng, TB, 3, 5, 1) |> aType
                    @test size(batched_matmul(A′, D′)) == (4, 5, 2)
                    @test batched_matmul(A′, D′) ≈ half_batched_mul(A′, D′)
                end
            end

            @testset "Large output, multi-threaded path" begin
                if TB == Float64
                    N = 50
                    A = rand(rng, N, N, N) |> aType
                    B = rand(rng, N, N, N) |> aType
                    C = reshape(
                        reduce(hcat, [vec(A[:, :, k] * B[:, :, k]) for k in 1:N]), N, N, N)
                    @test C ≈ A ⊠ B

                    D = rand(rng, N, N, 1) |> aType
                    E = reshape(
                        reduce(hcat, [vec(A[:, :, k] * D[:, :, 1]) for k in 1:N]), N, N, N)
                    @test E ≈ A ⊠ D
                end
            end
        end
    end
end

@testitem "batched_mul: trivial dimensions & unit strides" tags=[:batched_ops] setup=[
    SharedTestSetup, BatchedMMSetup] begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "Float64 × $(TB)" for TB in [Float64, ComplexF64]
            @testset "trivial dimensions & unit strides" begin
                @testset "$tA(rand$((sA...,3))) ⊠ $tB(rand$((sB...,3)))" for tA in [
                        identity, batched_adjoint, batched_transpose, perm_12, perm_23],
                    sA in [(1, 1), (1, 3), (3, 1), (3, 3)],
                    tB in [identity, batched_adjoint, batched_transpose, perm_12, perm_23],
                    sB in [(1, 1), (1, 3), (3, 1), (3, 3)]

                    A = tA(rand(rng, TB, sA..., 3)) |> aType
                    B = tB(rand(rng, TB, sB..., 3)) |> aType
                    size(A, 2) == size(B, 1) && size(A, 3) == size(B, 3) == 3 || continue

                    C = cat(A[:, :, 1] * B[:, :, 1], A[:, :, 2] * B[:, :, 2],
                        A[:, :, 3] * B[:, :, 3]; dims=3)
                    @test batched_matmul(A, B) ≈ C

                    α, β = rand(rng, TB), rand(rng, TB)
                    D = rand(rng, TB, size(C)) |> aType
                    @test batched_mul!(copy(D), A, B, α, β) ≈ α .* C .+ β .* D
                    @test NNlib.batched_mul_generic!(copy(D), A, B, α, β) ≈ α .* C .+ β .* D

                    C2 = batched_transpose(permutedims(C, (2, 1, 3)))
                    C3 = batched_adjoint(permutedims(conj(C), (2, 1, 3)))
                    @test Array(C2) == Array(C3) == Array(C)

                    if !ongpu
                        C2 .= D
                        C3 .= D
                        @test batched_mul!(C2, A, B, α, β) ≈ α .* C .+ β .* D
                        @test C2 ≈ α .* C .+ β .* D
                        @test batched_mul!(C3, A, B, α, β) ≈ α .* C .+ β .* D
                        @test C3 ≈ α .* C .+ β .* D
                    end
                end
            end
        end
    end
end

@testitem "BatchedAdjOrTrans interface" tags=[:batched_ops] setup=[
    SharedTestSetup, BatchedMMSetup] begin
    rng = StableRNG(1234)

    @testset "Float64 × $(TB)" for TB in [Float64, Float32]
        A = randn(rng, 7, 5, 3)
        B = randn(rng, TB, 5, 7, 3)
        C = randn(rng, 7, 6, 3)

        function interface_tests(X, _X)
            @test length(_X) == length(X)
            @test size(_X) == (size(X, 2), size(X, 1), size(X, 3))
            @test axes(_X) == (axes(X, 2), axes(X, 1), axes(X, 3))

            @test getindex(_X, 2, 3, 3) == getindex(X, 3, 2, 3)
            @test getindex(_X, 5, 4, 1) == getindex(X, 4, 5, 1)

            setindex!(_X, 2.0, 2, 4, 1)
            @test getindex(_X, 2, 4, 1) == 2.0
            setindex!(_X, 3.0, 1, 2, 2)
            @test getindex(_X, 1, 2, 2) == 3.0

            _sim = similar(_X, TB, (2, 3))
            @test size(_sim) == (2, 3)
            @test typeof(_sim) == Array{TB, 2}

            _sim = similar(_X, TB)
            @test length(_sim) == length(_X)
            @test typeof(_sim) == Array{TB, 3}

            _sim = similar(_X, (2, 3))
            @test size(_sim) == (2, 3)
            @test typeof(_sim) == Array{Float64, 2}

            _sim = similar(_X)
            @test length(_sim) == length(_X)
            @test typeof(_sim) == Array{Float64, 3}

            @test parent(_X) == _X.parent
        end

        for (X, _X) in zip([A, B, C], map(batched_adjoint, [A, B, C]))
            interface_tests(X, _X)

            @test -_X == NNlib.BatchedAdjoint(-_X.parent)

            _copyX = copy(_X)
            @test _X == _copyX

            setindex!(_copyX, 2.0, 1, 2, 1)
            @test _X != _copyX
        end

        for (X, _X) in zip([A, B, C], map(batched_transpose, [A, B, C]))
            interface_tests(X, _X)

            @test -_X == NNlib.BatchedTranspose(-_X.parent)

            _copyX = copy(_X)
            @test _X == _copyX

            setindex!(_copyX, 2.0, 1, 2, 1)
            @test _X != _copyX
        end
    end
end

@testitem "batched_matmul(ndims < 3)" tags=[:batched_ops] setup=[
    SharedTestSetup, BatchedMMSetup] begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "Float64 × $(TB)" for TB in [Float64, ComplexF64]
            A = randn(rng, 3, 3, 3) |> aType
            M = aType(rand(rng, TB, 3, 3)) .+ im
            V = aType(rand(rng, TB, 3))

            # These are all reshaped and sent to batched_matmul(3-array, 3-array)
            @test batched_matmul(A, M) ≈ cat([A[:, :, k] * M for k in 1:3]...; dims=3)
            @test batched_matmul(A, M') ≈ cat([A[:, :, k] * M' for k in 1:3]...; dims=3)
            @test A ⊠ transpose(M) ≈
                  cat([A[:, :, k] * transpose(M) for k in 1:3]...; dims=3)

            @test batched_matmul(M, A) ≈ cat([M * A[:, :, k] for k in 1:3]...; dims=3)
            @test batched_matmul(M', A) ≈ cat([M' * A[:, :, k] for k in 1:3]...; dims=3)
            @test transpose(M) ⊠ A ≈
                  cat([transpose(M) * A[:, :, k] for k in 1:3]...; dims=3)

            # batched_vec
            @test batched_vec(A, M) ≈ hcat([A[:, :, k] * M[:, k] for k in 1:3]...)
            @test batched_vec(A, M') ≈ hcat([A[:, :, k] * (M')[:, k] for k in 1:3]...)
            @test batched_vec(A, V) ≈ hcat([A[:, :, k] * V for k in 1:3]...)
        end
    end
end

@testitem "BMM AutoDiff" tags=[:batched_ops] setup=[SharedTestSetup, BatchedMMSetup] begin
    rng = StableRNG(1234)

    fn(A, B) = sum(batched_matmul(A, B))
    fn_vec(A, B) = sum(batched_vec(A, B))

    @testset "$mode" for (mode, aType, ongpu) in MODES
        M, P, Q = 13, 7, 11
        B = 3

        @testset "Two 3-arrays" begin
            test_gradients(fn, aType(randn(rng, M, P, B)),
                aType(randn(rng, P, Q, B)); atol=1e-3, rtol=1e-3)
            test_gradients(fn, batched_adjoint(aType(randn(rng, P, M, B))),
                aType(randn(rng, P, Q, B)); atol=1e-3, rtol=1e-3)
            test_gradients(fn, aType(randn(rng, M, P, B)),
                batched_transpose(aType(randn(rng, Q, P, B))); atol=1e-3, rtol=1e-3)
        end

        @testset "One a matrix..." begin
            test_gradients(fn, aType(randn(rng, M, P)),
                aType(randn(rng, P, Q, B)); atol=1e-3, rtol=1e-3)
            test_gradients(fn, adjoint(aType(randn(rng, P, M))),
                aType(randn(rng, P, Q, B)); atol=1e-3, rtol=1e-3)
            test_gradients(fn, aType(randn(rng, M, P)),
                batched_adjoint(aType(randn(rng, Q, P, B))); atol=1e-3, rtol=1e-3)

            test_gradients(fn, aType(randn(rng, M, P)),
                aType(randn(rng, P, Q, B)); atol=1e-3, rtol=1e-3)
            test_gradients(fn, adjoint(aType(randn(rng, P, M))),
                aType(randn(rng, P, Q, B)); atol=1e-3, rtol=1e-3)
            test_gradients(fn, aType(randn(rng, M, P)),
                batched_adjoint(aType(randn(rng, Q, P, B))); atol=1e-3, rtol=1e-3)
        end

        @testset "... or equivalent to a matrix" begin
            test_gradients(fn, aType(randn(rng, M, P, 1)),
                aType(randn(rng, P, Q, B)); atol=1e-3, rtol=1e-3)
            test_gradients(fn, batched_transpose(aType(randn(rng, P, M, 1))),
                aType(randn(rng, P, Q, B)); atol=1e-3, rtol=1e-3)
            test_gradients(fn, aType(randn(rng, M, P, 1)),
                batched_transpose(aType(randn(rng, Q, P, B))); atol=1e-3, rtol=1e-3)
        end

        @testset "batched_vec" begin
            test_gradients(fn_vec, aType(randn(rng, M, P, B)),
                aType(randn(rng, P, B)); atol=1e-3, rtol=1e-3)
            test_gradients(fn_vec, aType(randn(rng, M, P, B)),
                transpose(aType(randn(rng, B, P))); atol=1e-3, rtol=1e-3)

            test_gradients(fn_vec, aType(randn(rng, M, P, B)),
                aType(randn(rng, P)); atol=1e-3, rtol=1e-3)
        end
    end
end
