using Aqua
using WeightInitializers, Test, Statistics
using StableRNGs, Random, CUDA, LinearAlgebra

CUDA.allowscalar(false)

const GROUP = get(ENV, "GROUP", "All")

@testset "WeightInitializers.jl Tests" begin
    rngs_arrtypes = []

    if GROUP == "All" || GROUP == "CPU"
        append!(rngs_arrtypes,
            [(StableRNG(12345), AbstractArray), (Random.default_rng(), AbstractArray)])
    end

    if GROUP == "All" || GROUP == "CUDA"
        append!(rngs_arrtypes, [(CUDA.default_rng(), CuArray)])
    end

    @testset "_nfan" begin
        # Fallback
        @test WeightInitializers._nfan() == (1, 1)
        # Vector
        @test WeightInitializers._nfan(4) == (1, 4)
        # Matrix
        @test WeightInitializers._nfan(4, 5) == (5, 4)
        # Tuple
        @test WeightInitializers._nfan((4, 5, 6)) == WeightInitializers._nfan(4, 5, 6)
        # Convolution
        @test WeightInitializers._nfan(4, 5, 6) == 4 .* (5, 6)
    end

    @testset "rng = $(typeof(rng)) & arrtype = $arrtype" for (rng, arrtype) in rngs_arrtypes
        @testset "Sizes and Types: $init" for init in [zeros32, ones32, rand32, randn32,
            kaiming_uniform, kaiming_normal, glorot_uniform, glorot_normal,
            truncated_normal, identity_init
        ]
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
            @test cl(3) isa arrtype{Float32, 1}
            @test cl(3, 5) isa arrtype{Float32, 2}
        end

        @testset "Sizes and Types: $init" for (init, fp) in [(zeros16, Float16),
            (zerosC16, ComplexF16), (zeros32, Float32), (zerosC32, ComplexF32),
            (zeros64, Float64), (zerosC64, ComplexF64), (ones16, Float16),
            (onesC16, ComplexF16), (ones32, Float32), (onesC32, ComplexF32),
            (ones64, Float64), (onesC64, ComplexF64), (rand16, Float16),
            (randC16, ComplexF16), (rand32, Float32), (randC32, ComplexF32),
            (rand64, Float64), (randC64, ComplexF64), (randn16, Float16),
            (randnC16, ComplexF16), (randn32, Float32), (randnC32, ComplexF32),
            (randn64, Float64), (randnC64, ComplexF64)]
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
            @test cl(3) isa arrtype{fp, 1}
            @test cl(3, 5) isa arrtype{fp, 2}
        end

        @testset "AbstractArray Type: $init $T" for init in [kaiming_uniform,
                kaiming_normal,
                glorot_uniform, glorot_normal, truncated_normal, identity_init],
            T in (Float16, Float32,
                Float64, ComplexF16, ComplexF32, ComplexF64)

            init === truncated_normal && !(T <: Real) && continue

            @test init(T, 3) isa AbstractArray{T, 1}
            @test init(rng, T, 3) isa arrtype{T, 1}
            @test init(T, 3, 5) isa AbstractArray{T, 2}
            @test init(rng, T, 3, 5) isa arrtype{T, 2}

            cl = init(rng)
            @test cl(T, 3) isa arrtype{T, 1}
            @test cl(T, 3, 5) isa arrtype{T, 2}

            cl = init(rng, T)
            @test cl(3) isa arrtype{T, 1}
            @test cl(3, 5) isa arrtype{T, 2}
        end

        @testset "Closure: $init" for init in [kaiming_uniform, kaiming_normal,
            glorot_uniform, glorot_normal, truncated_normal, identity_init]
            cl = init(;)
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
            Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
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
            for (n_in, n_out) in [(100, 100), (100, 400)]
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
                fan_in, fan_out = WeightInitializers._nfan(dims...)
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

    @testset "Orthogonal rng = $(typeof(rng)) & arrtype = $arrtype" for (rng, arrtype) in rngs_arrtypes
        # A matrix of dim = (m,n) with m > n should produce a QR decomposition.
        # In the other case, the transpose should be taken to compute the QR decomposition.
        for (rows, cols) in [(5, 3), (3, 5)]
            v = orthogonal(rng, rows, cols)
            CUDA.@allowscalar rows < cols ? (@test v * v' ≈ I(rows)) :
                              (@test v' * v ≈ I(cols))
        end
        for mat in [(3, 4, 5), (2, 2, 5)]
            v = orthogonal(rng, mat...)
            cols = mat[end]
            rows = div(prod(mat), cols)
            v = reshape(v, (rows, cols))
            CUDA.@allowscalar rows < cols ? (@test v * v' ≈ I(rows)) :
                              (@test v' * v ≈ I(cols))
        end
        # Type
        @testset "Orthogonal Types $T" for T in (Float32, Float64)#(Float16, Float32, Float64)
            @test eltype(orthogonal(rng, T, 3, 4; gain=1.5)) == T
            @test eltype(orthogonal(rng, T, 3, 4, 5; gain=1.5)) == T
        end
        @testset "Orthogonal AbstractArray Type $T" for T in (Float32, Float64)#(Float16, Float32, Float64)
            @test orthogonal(T, 3, 5) isa AbstractArray{T, 2}
            @test orthogonal(rng, T, 3, 5) isa arrtype{T, 2}

            cl = orthogonal(rng)
            @test cl(T, 3, 5) isa arrtype{T, 2}

            cl = orthogonal(rng, T)
            @test cl(3, 5) isa arrtype{T, 2}
        end
        @testset "Orthogonal Closure" begin
            cl = orthogonal(;)
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

    @testset "sparse_init rng = $(typeof(rng)) & arrtype = $arrtype" for (rng, arrtype) in rngs_arrtypes
        # sparse_init should yield an error for non 2-d dimensions
        # sparse_init should yield no zero elements if sparsity < 0
        # sparse_init should yield all zero elements if sparsity > 1
        # sparse_init should yield exactly ceil(n_in * sparsity) elements in each column for other sparsity values
        # sparse_init should yield a kernel in its non-zero elements consistent with the std parameter

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

        # Type
        @testset "sparse_init Types $T" for T in (Float16, Float32, Float64)
            @test eltype(sparse_init(rng, T, 3, 4; sparsity=0.5)) == T
        end
        @testset "sparse_init AbstractArray Type $T" for T in (Float16, Float32, Float64)
            @test sparse_init(T, 3, 5; sparsity=0.5) isa AbstractArray{T, 2}
            @test sparse_init(rng, T, 3, 5; sparsity=0.5) isa arrtype{T, 2}

            cl = sparse_init(rng; sparsity=0.5)
            @test cl(T, 3, 5) isa arrtype{T, 2}

            cl = sparse_init(rng, T; sparsity=0.5)
            @test cl(3, 5) isa arrtype{T, 2}
        end
        @testset "sparse_init Closure" begin
            cl = sparse_init(; sparsity=0.5)
            # Sizes
            @test size(cl(3, 4)) == (3, 4)
            @test size(cl(rng, 3, 4)) == (3, 4)
            # Type
            @test eltype(cl(4, 2)) == Float32
            @test eltype(cl(rng, 4, 2)) == Float32
        end
    end

    @testset "identity_init" begin
        @testset "Non-identity sizes" begin
            @test identity_init(2, 3)[:, end] == zeros(Float32, 2)
            @test identity_init(3, 2; shift=1)[1, :] == zeros(Float32, 2)
            @test identity_init(1, 1, 3, 4)[:, :, :, end] == zeros(Float32, 1, 1, 3)
            @test identity_init(2, 1, 3, 3)[end, :, :, :] == zeros(Float32, 1, 3, 3)
            @test identity_init(1, 2, 3, 3)[:, end, :, :] == zeros(Float32, 1, 3, 3)
        end
    end

    @testset "Warning: truncated_normal" begin
        @test_warn "Mean is more than 2 std outside the limits in truncated_normal, so \
            the distribution of values may be inaccurate." truncated_normal(2; mean=-5.0f0)
    end

    @testset "Aqua: Quality Assurance" begin
        Aqua.test_all(WeightInitializers; ambiguities=false)
        Aqua.test_ambiguities(WeightInitializers; recursive=false)
    end
end
