using Statistics, WeightInitializers, Test

include("common.jl")

@testset "Sparse Initialization" begin
    @testset "rng = $(typeof(rng)) & arrtype = $arrtype" for (
        rng, arrtype, supports_fp64, backend
    ) in RNGS_ARRTYPES
        # sparse_init should yield an error for non 2-d dimensions
        # sparse_init should yield no zero elements if sparsity < 0
        # sparse_init should yield all zero elements if sparsity > 1
        # sparse_init should yield exactly ceil(n_in * sparsity) elements in each column for
        # other sparsity values
        # sparse_init should yield a kernel in its non-zero elements consistent with the std
        # parameter

        @test_throws ArgumentError sparse_init(3, 4, 5, sparsity=0.1)
        @test_throws ArgumentError sparse_init(3, sparsity=0.1)
        v = sparse_init(100, 100; sparsity=-0.1)
        @test sum(v .== 0) == 0
        v = sparse_init(100, 100; sparsity=1.1)
        @test sum(v .== 0) == length(v)

        for (n_in, n_out, sparsity, σ) in [(100, 100, 0.25, 0.1), (100, 400, 0.75, 0.01)]
            expected_zeros = ceil(Integer, n_in * sparsity)
            v = sparse_init(n_in, n_out; sparsity=sparsity, std=σ)
            @test all([sum(v[:, col] .== 0) == expected_zeros for col in 1:n_out])
            @test 0.9 * σ < std(v[v .!= 0]) < 1.1 * σ
        end

        @testset "sparse_init Type $T" for T in (Float16, Float32, Float64)
            !supports_fp64 && T == Float64 && continue

            @test eltype(sparse_init(rng, T, 3, 4; sparsity=0.5)) == T
        end

        @testset "sparse_init AbstractArray Type $T" for T in (Float16, Float32, Float64)
            !supports_fp64 && T == Float64 && continue

            @test sparse_init(T, 3, 5; sparsity=0.5) isa AbstractArray{T,2}
            @test sparse_init(rng, T, 3, 5; sparsity=0.5) isa arrtype{T,2}

            cl = sparse_init(rng; sparsity=0.5)
            display(cl)
            @test cl(T, 3, 5) isa arrtype{T,2}

            cl = sparse_init(rng, T; sparsity=0.5)
            display(cl)
            @test cl(3, 5) isa arrtype{T,2}
        end

        @testset "sparse_init Closure" begin
            cl = sparse_init(; sparsity=0.5)
            display(cl)

            # Sizes
            @test size(cl(3, 4)) == (3, 4)
            @test size(cl(rng, 3, 4)) == (3, 4)

            # Type
            @test eltype(cl(4, 2)) == Float32
            @test eltype(cl(rng, 4, 2)) == Float32
        end
    end
end
