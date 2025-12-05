@testitem "Warning: truncated_normal" begin
    @test_warn "Mean is more than 2 std outside the limits in truncated_normal, so \
        the distribution of values may be inaccurate." truncated_normal(2; mean=-5.0f0)
end

@testitem "Identity Initialization" begin
    using LinearAlgebra

    @testset "2D identity matrices" begin
        # Square matrix should be identity
        mat = identity_init(5, 5)
        @test mat ≈ Matrix{Float32}(I, 5, 5)
        @test diag(mat) == ones(Float32, 5)
        # Check off-diagonal elements are zero
        for i in 1:5, j in 1:5
            if i != j
                @test mat[i, j] == 0.0f0
            end
        end

        # Test with gain parameter
        mat_gain = identity_init(4, 4; gain=2.5)
        @test mat_gain ≈ 2.5f0 * Matrix{Float32}(I, 4, 4)
        @test diag(mat_gain) == fill(2.5f0, 4)

        # Non-square matrices
        mat_rect1 = identity_init(3, 5)
        @test size(mat_rect1) == (3, 5)
        @test diag(mat_rect1) == ones(Float32, 3)
        @test mat_rect1[:, 4:5] == zeros(Float32, 3, 2)

        mat_rect2 = identity_init(5, 3)
        @test size(mat_rect2) == (5, 3)
        @test diag(mat_rect2) == ones(Float32, 3)
        @test mat_rect2[4:5, :] == zeros(Float32, 2, 3)
    end

    @testset "Non-identity sizes" begin
        @test identity_init(2, 3)[:, end] == zeros(Float32, 2)
        @test identity_init(3, 2; shift=1)[1, :] == zeros(Float32, 2)
        @test identity_init(1, 1, 3, 4)[:, :, :, end] == zeros(Float32, 1, 1, 3)
        @test identity_init(2, 1, 3, 3)[end, :, :, :] == zeros(Float32, 1, 3, 3)
        @test identity_init(1, 2, 3, 3)[:, end, :, :] == zeros(Float32, 1, 3, 3)
    end
end

@testitem "Orthogonal Initialization" setup = [SharedTestSetup] begin
    using GPUArraysCore, LinearAlgebra

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

@testitem "Sparse Initialization" setup = [SharedTestSetup] begin
    using Statistics

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

@testitem "Basic Initializations" setup = [SharedTestSetup] begin
    using LinearAlgebra, Statistics

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

@testitem "Kaiming Uniform: Complex" begin
    using WeightInitializers, Test

    x = kaiming_uniform(ComplexF32, 1024, 1024)
    @test eltype(x) == ComplexF32
    @test size(x) == (1024, 1024)
    @test minimum(imag.(x)) < 0.0
end

@testitem "Initialization inside compile" begin
    using Reactant, WeightInitializers, Test

    rrng = Reactant.ReactantRNG()

    @testset "Concrete: $(op)" for op in (zeros32, ones32)
        gen_arr = op(rrng, 3, 4)
        @test eltype(gen_arr) == Float32
        @test size(gen_arr) == (3, 4)
        @test gen_arr isa Reactant.ConcreteRArray{Float32,2}
    end

    @testset "Traced: $(op)" for op in (zeros32, ones32, rand32, randn32)
        gen_arr = @jit op(rrng, 3, 4)
        @test eltype(gen_arr) == Float32
        @test size(gen_arr) == (3, 4)
        @test gen_arr isa Reactant.ConcreteRArray{Float32,2}
    end
end
