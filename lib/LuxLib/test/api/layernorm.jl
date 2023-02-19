using CUDA, Statistics, Test
using LuxLib

include("../test_utils.jl")

function _setup_layernorm(T, x_size, affine_shape)
    x = randn(T, x_size)
    if affine_shape !== nothing
        scale = randn(T, affine_shape..., 1)
        bias = randn(T, affine_shape..., 1)
        return x, scale, bias
    else
        return x, nothing, nothing
    end
end

@testset "LayerNorm" begin
    if cpu_testing()
        for T in (Float16, Float32, Float64),
            x_shape in ((3, 3, 2, 1), (2, 2, 2, 1), (2, 3, 2, 2)),
            affine_shape in (nothing, x_shape[1:3], (1, 1, 1), (1, 1, x_shape[3]))

            println("LN_CPU: $T $(x_shape) $(affine_shape)")

            dims = Colon()
            epsilon = T(1e-5)
            _f = (args...) -> layernorm(args...; dims, epsilon)

            x, scale, bias = _setup_layernorm(T, x_shape, affine_shape)

            @inferred _f(x, scale, bias)

            y = _f(x, scale, bias)

            @test y isa Array{T, 4}
            @test size(y) == x_shape

            if affine_shape === nothing
                @test isapprox(mean(y; dims), 0; atol=1e-3, rtol=1e-3)
                @test isapprox(std(y; dims), 1; atol=1e-1, rtol=1e-1)
            end

            run_JET_tests(_f, x, scale, bias)

            if T != Float16 # FDM is not ideal with Float16 values
                if affine_shape === nothing
                    test_gradient_correctness_fdm(x -> sum(_f(x, nothing, nothing)), x;
                                                  atol=1.0f-2, rtol=1.0f-2)
                else
                    test_gradient_correctness_fdm(sum ∘ _f, x, scale, bias; atol=1.0f-2,
                                                  rtol=1.0f-2)
                end
            end
        end
    end

    if gpu_testing()
        for T in (Float16, Float32, Float64),
            x_shape in ((3, 3, 2, 1), (2, 2, 2, 1), (2, 3, 2, 2)),
            affine_shape in (nothing, x_shape[1:3], (1, 1, 1), (1, 1, x_shape[3]))

            println("LN_GPU: $T $(x_shape) $(affine_shape)")

            dims = Colon()
            epsilon = T(1e-5)
            _f = (args...) -> layernorm(args...; dims, epsilon)

            x, scale, bias = _setup_layernorm(T, x_shape, affine_shape)

            x = x |> cu .|> T
            if affine_shape !== nothing
                scale = scale |> cu .|> T
                bias = bias |> cu .|> T
            end

            @inferred _f(x, scale, bias)

            y = _f(x, scale, bias)

            @test y isa CuArray{T, 4}
            @test size(y) == x_shape

            if affine_shape === nothing
                @test isapprox(mean(y; dims), 0; atol=1e-3, rtol=1e-3)
                @test isapprox(std(y; dims), 1; atol=1e-1, rtol=1e-1)
            end

            run_JET_tests(_f, x, scale, bias)

            # if T != Float16 # FDM is not ideal with Float16 values
            #     if affine_shape === nothing
            #         test_gradient_correctness_fdm(x -> sum(_f(x, nothing, nothing)), x;
            #                                       atol=1.0f-2, rtol=1.0f-2)
            #     else
            #         test_gradient_correctness_fdm(sum ∘ _f, x, scale, bias; atol=1.0f-2,
            #                                       rtol=1.0f-2)
            #     end
            # end
        end
    end
end
