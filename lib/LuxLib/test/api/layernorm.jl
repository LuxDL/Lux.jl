using LuxLib, Statistics, Test

include("../test_utils.jl")

function _setup_layernorm(aType, T, x_size, affine_shape)
    x = randn(T, x_size) |> aType
    if affine_shape !== nothing
        scale = randn(T, affine_shape..., 1) |> aType
        bias = randn(T, affine_shape..., 1) |> aType
        return x, scale, bias
    else
        return x, nothing, nothing
    end
end

@testset "$mode: LayerNorm" for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        x_shape in ((3, 3, 2, 1), (2, 2, 2, 1), (2, 3, 2, 2)),
        affine_shape in (nothing, x_shape[1:3], (1, 1, 1), (1, 1, x_shape[3]))

        dims = Colon()
        epsilon = T(1e-5)
        _f = (args...) -> layernorm(args...; dims, epsilon)

        x, scale, bias = _setup_layernorm(aType, T, x_shape, affine_shape)

        @inferred _f(x, scale, bias)
        @jet _f(x, scale, bias)

        y = _f(x, scale, bias)

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape

        if affine_shape === nothing
            @test check_approx(mean(y; dims), 0; atol=1e-3, rtol=1e-3)
            @test check_approx(std(y; dims), 1; atol=1e-1, rtol=1e-1)
        end

        fp16 = T == Float16
        if affine_shape !== nothing
            __f = (args...) -> sum(_f(x, args...))
            @eval @test_gradients $__f $scale $bias soft_fail=$fp16 atol=1.0f-2 rtol=1.0f-2 gpu_testing=$on_gpu
        end
    end
end
