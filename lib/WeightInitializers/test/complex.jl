using LinearAlgebra, Random, StableRNGs, Statistics, Test, WeightInitializers

@testset "Complex Initializers" begin
    rng = StableRNG(12345)

    @testset "Complex Initializers ($T)" for T in (ComplexF32, ComplexF64)
        fan_in, fan_out = 512, 512
        dims = (fan_out, fan_in)

        # 1. Glorot Uniform
        @testset "Glorot Uniform" begin
            for gain in (1.0, 1.5)
                w = glorot_uniform(rng, T, dims...; gain=gain)
                @test eltype(w) == T
                @test size(w) == dims
                @test abs(mean(w)) < 0.01

                theo_var = 2 * gain^2 / (fan_in + fan_out)
                @test isapprox(var(w), theo_var; rtol=0.05)
                @test isapprox(var(real.(w)), theo_var / 2; rtol=0.08)
                @test isapprox(var(imag.(w)), theo_var / 2; rtol=0.08)

                # Axis-aligned uniform bounds: Re and Im each in [-b, b]
                b = gain * sqrt(3 / (fan_in + fan_out))
                @test maximum(real.(w)) ≤ b + 1e-4
                @test minimum(real.(w)) ≥ -b - 1e-4
                @test maximum(imag.(w)) ≤ b + 1e-4
                @test minimum(imag.(w)) ≥ -b - 1e-4
            end

            # Complex gain conversion without InexactError
            w_cgain = glorot_uniform(rng, T, 16, 16; gain=1.0 + 0.5im)
            @test eltype(w_cgain) == T
            @test size(w_cgain) == (16, 16)
        end

        # 2. Glorot Normal
        @testset "Glorot Normal" begin
            for gain in (1.0, 1.5)
                w = glorot_normal(rng, T, dims...; gain=gain)
                @test eltype(w) == T
                @test size(w) == dims
                @test abs(mean(w)) < 0.01

                theo_var = 2 * gain^2 / (fan_in + fan_out)
                @test isapprox(var(w), theo_var; rtol=0.05)
                @test isapprox(var(real.(w)), theo_var / 2; rtol=0.08)
                @test isapprox(var(imag.(w)), theo_var / 2; rtol=0.08)
            end

            w_cgain = glorot_normal(rng, T, 16, 16; gain=1.0 + 0.5im)
            @test eltype(w_cgain) == T
            @test size(w_cgain) == (16, 16)
        end

        # 3. Kaiming Uniform
        @testset "Kaiming Uniform" begin
            w = kaiming_uniform(rng, T, dims...)
            @test eltype(w) == T
            @test size(w) == dims
            @test abs(mean(w)) < 0.01

            gain = √real(T)(2)
            theo_var = gain^2 / fan_in
            @test isapprox(var(w), theo_var; rtol=0.05)
            @test isapprox(var(real.(w)), theo_var / 2; rtol=0.08)
            @test isapprox(var(imag.(w)), theo_var / 2; rtol=0.08)

            # Axis-aligned uniform bounds: Re and Im each in [-bound, bound]
            bound = √(1.5) * gain / sqrt(fan_in)
            @test maximum(real.(w)) ≤ bound + 1e-4
            @test minimum(real.(w)) ≥ -bound - 1e-4
            @test maximum(imag.(w)) ≤ bound + 1e-4
            @test minimum(imag.(w)) ≥ -bound - 1e-4

            w_cgain = kaiming_uniform(rng, T, 16, 16; gain=1.0 + 0.5im)
            @test eltype(w_cgain) == T
            @test size(w_cgain) == (16, 16)
        end

        # 4. Kaiming Normal
        @testset "Kaiming Normal" begin
            w = kaiming_normal(rng, T, dims...)
            @test eltype(w) == T
            @test size(w) == dims
            @test abs(mean(w)) < 0.01

            gain = √real(T)(2)
            theo_var = gain^2 / fan_in
            @test isapprox(var(w), theo_var; rtol=0.05)
            @test isapprox(var(real.(w)), theo_var / 2; rtol=0.08)
            @test isapprox(var(imag.(w)), theo_var / 2; rtol=0.08)

            w_cgain = kaiming_normal(rng, T, 16, 16; gain=1.0 + 0.5im)
            @test eltype(w_cgain) == T
            @test size(w_cgain) == (16, 16)
        end

        # 5. Orthogonal
        @testset "Orthogonal" begin
            for gain in (1.0, 1.5)
                # 2D case rows >= cols
                w = orthogonal(rng, T, 16, 8; gain=gain)
                @test eltype(w) == T
                @test size(w) == (16, 8)
                @test isapprox(w' * w, (gain^2) * I(8); atol=1e-5)

                # 2D case rows < cols
                w = orthogonal(rng, T, 8, 16; gain=gain)
                @test eltype(w) == T
                @test size(w) == (8, 16)
                @test isapprox(w * w', (gain^2) * I(8); atol=1e-5)

                # N-D cases where prod(dims[1:end-1]) < dims[end]
                for ndims in ((2, 2, 8), (2, 2, 2, 16))
                    w_nd = orthogonal(rng, T, ndims...; gain=gain)
                    @test eltype(w_nd) == T
                    @test size(w_nd) == ndims
                    rows = prod(ndims[1:(end - 1)])
                    cols = ndims[end]
                    w_mat = reshape(w_nd, rows, cols)
                    @test isapprox(w_mat * w_mat', (gain^2) * I(rows); atol=1e-5)
                end
            end
        end
    end
end
