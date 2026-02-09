using LinearAlgebra, Statistics, WeightInitializers, Test

include("common.jl")

@testset "Warning: truncated_normal" begin
    @test_warn "Mean is more than 2 std outside the limits in truncated_normal, so \
        the distribution of values may be inaccurate." truncated_normal(2; mean=-5.0f0)
end

@testset "Basic Initializations" begin
    @testset "rng = $(typeof(rng)) & arrtype = $arrtype" for (
        rng, arrtype, supports_fp64, backend
    ) in RNGS_ARRTYPES
        @testset "Sizes and Types: $init" for init in [
            zeros32,
            ones32,
            rand32,
            randn32,
            kaiming_uniform,
            kaiming_normal,
            glorot_uniform,
            glorot_normal,
            truncated_normal,
            identity_init,
        ]
            !supports_fp64 &&
                (
                    init === zeros32 ||
                    init === ones32 ||
                    init === rand32 ||
                    init === randn32
                ) &&
                continue

            if backend == "oneapi" && init === truncated_normal
                @test_broken size(init(rng, 3)) == (3,)  # `erfinv` not implemented
                continue
            end

            # Sizes
            @test size(init(3)) == (3,)
            @test size(init(rng, 3)) == (3,)
            @test size(init(3, 4)) == (3, 4)
            @test size(init(rng, 3, 4)) == (3, 4)
            @test size(init(3, 4, 5)) == (3, 4, 5)
            @test size(init(rng, 3, 4, 5)) == (3, 4, 5)

            # Type
            @test eltype(init(rng, 4, 2)) == Float32
            @test eltype(init(4, 2)) == Float32

            # RNG Closure
            cl = init(rng)
            display(cl)
            @test cl(3) isa arrtype{Float32,1}
            @test cl(3, 5) isa arrtype{Float32,2}
        end

        @testset "Sizes and Types: $init" for (init, fp) in [
            (zeros16, Float16),
            (zerosC16, ComplexF16),
            (zeros32, Float32),
            (zerosC32, ComplexF32),
            (zeros64, Float64),
            (zerosC64, ComplexF64),
            (ones16, Float16),
            (onesC16, ComplexF16),
            (ones32, Float32),
            (onesC32, ComplexF32),
            (ones64, Float64),
            (onesC64, ComplexF64),
            (rand16, Float16),
            (randC16, ComplexF16),
            (rand32, Float32),
            (randC32, ComplexF32),
            (rand64, Float64),
            (randC64, ComplexF64),
            (randn16, Float16),
            (randnC16, ComplexF16),
            (randn32, Float32),
            (randnC32, ComplexF32),
            (randn64, Float64),
            (randnC64, ComplexF64),
        ]
            !supports_fp64 && (fp == Float64 || fp == ComplexF64) && continue

            # Sizes
            @test size(init(3)) == (3,)
            @test size(init(rng, 3)) == (3,)
            @test size(init(3, 4)) == (3, 4)
            @test size(init(rng, 3, 4)) == (3, 4)
            @test size(init(3, 4, 5)) == (3, 4, 5)
            @test size(init(rng, 3, 4, 5)) == (3, 4, 5)

            # Type
            @test eltype(init(rng, 4, 2)) == fp
            @test eltype(init(4, 2)) == fp

            # RNG Closure
            cl = init(rng)
            display(cl)
            @test cl(3) isa arrtype{fp,1}
            @test cl(3, 5) isa arrtype{fp,2}

            # Kwargs closure
            cl = init()
            display(cl)
            @test cl(rng, 3) isa arrtype{fp,1}
            @test cl(rng, 3, 5) isa arrtype{fp,2}

            # throw error on type as input
            @test_throws ArgumentError init(Float32)
            @test_throws ArgumentError init(Float32, 3, 5)
            @test_throws ArgumentError init(rng, Float32)
            @test_throws ArgumentError init(rng, Float32, 3, 5)
        end

        @testset "AbstractArray Type: $init $T" for init in [
                kaiming_uniform,
                kaiming_normal,
                glorot_uniform,
                glorot_normal,
                truncated_normal,
                identity_init,
            ],
            T in (Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)

            !supports_fp64 && (T == Float64 || T == ComplexF64) && continue

            init === truncated_normal && !(T <: Real) && continue

            if backend == "oneapi" && init === truncated_normal && T == Float32
                @test_broken init(rng, T, 3) isa AbstractArray{T,1}  # `erfinv` not implemented
                continue
            end

            @test init(T, 3) isa AbstractArray{T,1}
            @test init(rng, T, 3) isa arrtype{T,1}
            @test init(T, 3, 5) isa AbstractArray{T,2}
            @test init(rng, T, 3, 5) isa arrtype{T,2}

            cl = init(rng)
            display(cl)
            @test cl(T, 3) isa arrtype{T,1}
            @test cl(T, 3, 5) isa arrtype{T,2}

            cl = init(rng, T)
            display(cl)
            @test cl(3) isa arrtype{T,1}
            @test cl(3, 5) isa arrtype{T,2}

            cl = init(T)
            display(cl)
            @test cl(3) isa Array{T,1}
            @test cl(3, 5) isa Array{T,2}
            @test cl(rng, 3, 5) isa arrtype{T,2}
        end

        @testset "Closure: $init" for init in [
            kaiming_uniform,
            kaiming_normal,
            glorot_uniform,
            glorot_normal,
            truncated_normal,
            identity_init,
        ]
            if backend == "oneapi" && init === truncated_normal
                @test_broken size(init(rng, 3)) == (3,)  # `erfinv` not implemented
                continue
            end

            cl = init()
            display(cl)

            # Sizes
            @test size(cl(3)) == (3,)
            @test size(cl(rng, 3)) == (3,)
            @test size(cl(3, 4)) == (3, 4)
            @test size(cl(rng, 3, 4)) == (3, 4)
            @test size(cl(3, 4, 5)) == (3, 4, 5)
            @test size(cl(rng, 3, 4, 5)) == (3, 4, 5)

            # Type
            @test eltype(cl(4, 2)) == Float32
            @test eltype(cl(rng, 4, 2)) == Float32
        end

        @testset "Kwargs types" for T in (
            Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64
        )
            !supports_fp64 && (T == Float64 || T == ComplexF64) && continue

            if (T <: Real)
                @test eltype(truncated_normal(T, 2, 5; mean=0, std=1, lo=-2, hi=2)) == T
                @test eltype(orthogonal(T, 2, 5; gain=1.0)) == T
            end
            @test eltype(glorot_uniform(T, 2, 5; gain=1.0)) == T
            @test eltype(glorot_normal(T, 2, 5; gain=1.0)) == T
            @test eltype(kaiming_uniform(T, 2, 5; gain=sqrt(2))) == T
            @test eltype(kaiming_normal(T, 2, 5; gain=sqrt(2))) == T
            @test eltype(identity_init(T, 2, 5; gain=1.0)) == T
            @test eltype(sparse_init(T, 2, 5; sparsity=0.5, std=0.01)) == T
        end

        @testset "kaiming" begin
            # kaiming_uniform should yield a kernel in range [-sqrt(6/n_out), sqrt(6/n_out)]
            # and kaiming_normal should yield a kernel with stddev ~= sqrt(2/n_out)
            @testset for (n_in, n_out) in [(100, 100), (100, 400)]
                v = kaiming_uniform(rng, n_in, n_out)
                σ2 = sqrt(6 / n_out)
                @test -1σ2 < minimum(v) < -0.9σ2
                @test 0.9σ2 < maximum(v) < 1σ2

                v = kaiming_normal(rng, n_in, n_out)
                σ2 = sqrt(2 / n_out)
                @test 0.9σ2 < std(v) < 1.1σ2
            end

            # Type
            @test eltype(kaiming_uniform(rng, 3, 4; gain=1.5f0)) == Float32
            @test eltype(kaiming_normal(rng, 3, 4; gain=1.5f0)) == Float32
        end

        @testset "glorot: $init" for init in [glorot_uniform, glorot_normal]
            # glorot_uniform and glorot_normal should both yield a kernel with
            # variance ≈ 2/(fan_in + fan_out)
            for dims in [(1000,), (100, 100), (100, 400), (2, 3, 32, 64), (2, 3, 4, 32, 64)]
                v = init(dims...)
                fan_in, fan_out = WeightInitializers.Utils.nfan(dims...)
                σ2 = 2 / (fan_in + fan_out)
                @test 0.9σ2 < var(v) < 1.1σ2
            end
            @test eltype(init(3, 4; gain=1.5)) == Float32
        end

        @testset "orthogonal" begin
            # A matrix of dim = (m,n) with m > n should produce a QR decomposition. In the other case, the transpose should be taken to compute the QR decomposition.
            for (rows, cols) in [(5, 3), (3, 5)]
                v = orthogonal(rows, cols)
                rows < cols ? (@test v * v' ≈ I(rows)) : (@test v' * v ≈ I(cols))
            end
            for mat in [(3, 4, 5), (2, 2, 5)]
                v = orthogonal(mat...)
                cols = mat[end]
                rows = div(prod(mat), cols)
                v = reshape(v, (rows, cols))
                rows < cols ? (@test v * v' ≈ I(rows)) : (@test v' * v ≈ I(cols))
            end
            @test eltype(orthogonal(3, 4; gain=1.5)) == Float32
        end
    end
end
