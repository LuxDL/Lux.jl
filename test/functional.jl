@testset "Elementwise Operation Dispatches" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    # For FP32 Inputs
    x = randn(rng, Float32, 10, 5, 2)
    y = randn(rng, Float32, 10, 1, 2)

    # On CPU the fallback should always work
    @test Lux.elementwise_add(x, y) == x .+ y
    @test Lux.elementwise_mul(x, y) == x .* y
    @test Lux.applyactivation(tanh, x) == tanh.(x)

    custom_activation(x) = abs(x)
    @test Lux.applyactivation(custom_activation, x) == custom_activation.(x)

    # On GPU try to use CUDNN
    if CUDA.functional()
        x_g = x |> gpu
        y_g = y |> gpu

        @test Lux.elementwise_add(x_g, y_g) == x_g .+ y_g
        @test Lux.elementwise_mul(x_g, y_g) == x_g .* y_g
        @test Lux.applyactivation(tanh, x_g) == tanh.(x_g)
        # Custom Activation test
        @test Lux.applyactivation(custom_activation, x_g) == custom_activation.(x_g)
    end

    # For Complex Inputs
    x = randn(rng, ComplexF32, 10, 5, 2)
    y = randn(rng, ComplexF32, 10, 1, 2)

    # On CPU the fallback should always work
    @test Lux.elementwise_add(x, y) == x .+ y
    @test Lux.elementwise_mul(x, y) == x .* y
    @test Lux.applyactivation(tanh, x) == tanh.(x)
    @test Lux.applyactivation(custom_activation, x) == custom_activation.(x)

    # On GPU try to use CUDNN but use fallback if CUDNN doesn't support the operation (ComplexFP32 in this case).
    # See https://github.com/avik-pal/Lux.jl/issues/22
    if CUDA.functional()
        x_g = x |> gpu
        y_g = y |> gpu

        @test Lux.elementwise_add(x_g, y_g) == x_g .+ y_g
        @test Lux.elementwise_mul(x_g, y_g) == x_g .* y_g
        ## See https://github.com/FluxML/NNlibCUDA.jl/issues/47
        ## NNlibCUDA changes how broadcasting behaves for CuArrays
        @test_broken Lux.applyactivation(tanh, x_g) == tanh.(x_g)
        # Custom Activation test
        @test Lux.applyactivation(custom_activation, x_g) == custom_activation.(x_g)
    end
end