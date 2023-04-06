using Lux, Random, Test

include("test_utils.jl")

@testset "$mode: Elementwise Operation Dispatches" for (mode, aType, device, ongpu) in MODES
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    custom_activation(x) = abs(x)

    for T in [Float64, Float32, ComplexF64, ComplexF32]
        x = randn(rng, T, 10, 5, 2) |> aType
        y = randn(rng, T, 10, 1, 2) |> aType

        # On CPU the fallback should always work
        @test Lux.elementwise_add(x, y) == x .+ y
        @test Lux.elementwise_mul(x, y) == x .* y
        @test Lux.applyactivation(tanh, x) == tanh.(x)
        @test Lux.applyactivation(custom_activation, x) == custom_activation.(x)

        if T <: Real
            # Gradient for complex outputs are not defined
            @eval @test_gradients $(sum ∘ Lux.elementwise_add) $x $y gpu_testing=$ongpu
        end

        # Deprecated Functionality (Remove in v0.5)
        @test_deprecated Lux.elementwise_add(x, y)
        @test_deprecated Lux.elementwise_mul(x, y)
        @test_deprecated Lux.applyactivation(tanh, x)
    end
end
