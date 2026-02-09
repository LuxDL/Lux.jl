using LinearAlgebra, WeightInitializers, Test
using GPUArraysCore

include("common.jl")

@testset "Orthogonal Initialization" begin
    @testset "rng = $(typeof(rng)) & arrtype = $arrtype" for (
        rng, arrtype, supports_fp64, backend
    ) in RNGS_ARRTYPES
        # A matrix of dim = (m,n) with m > n should produce a QR decomposition.
        # In the other case, the transpose should be taken to compute the QR decomposition.
        if backend == "oneapi" || backend == "metal"  # `qr` not implemented
            @test_broken orthogonal(rng, 3, 5) isa arrtype{Float32,2}
            continue
        end

        for (rows, cols) in [(5, 3), (3, 5)]
            v = orthogonal(rng, rows, cols)
            GPUArraysCore.@allowscalar if rows < cols
                (@test v * v' ≈ I(rows))
            else
                (@test v' * v ≈ I(cols))
            end
        end

        for mat in [(3, 4, 5), (2, 2, 5)]
            v = orthogonal(rng, mat...)
            cols = mat[end]
            rows = div(prod(mat), cols)
            v = reshape(v, (rows, cols))
            GPUArraysCore.@allowscalar if rows < cols
                (@test v * v' ≈ I(rows))
            else
                (@test v' * v ≈ I(cols))
            end
        end

        @testset "Orthogonal Types $T" for T in (Float32, Float64)
            !supports_fp64 && T == Float64 && continue

            @test eltype(orthogonal(rng, T, 3, 4; gain=1.5)) == T
            @test eltype(orthogonal(rng, T, 3, 4, 5; gain=1.5)) == T
        end

        @testset "Orthogonal AbstractArray Type $T" for T in (Float32, Float64)
            !supports_fp64 && T == Float64 && continue

            @test orthogonal(rng, T, 3, 5) isa AbstractArray{T,2}
            @test orthogonal(rng, T, 3, 5) isa arrtype{T,2}

            cl = orthogonal(rng)
            display(cl)
            @test cl(T, 3, 5) isa arrtype{T,2}

            cl = orthogonal(rng, T)
            display(cl)
            @test cl(3, 5) isa arrtype{T,2}
        end

        @testset "Orthogonal Closure" begin
            cl = orthogonal()
            display(cl)

            # Sizes
            @test size(cl(3, 4)) == (3, 4)
            @test size(cl(rng, 3, 4)) == (3, 4)
            @test size(cl(3, 4, 5)) == (3, 4, 5)
            @test size(cl(rng, 3, 4, 5)) == (3, 4, 5)

            # Type
            @test eltype(cl(4, 2)) == Float32
            @test eltype(cl(rng, 4, 2)) == Float32
        end
    end
end
