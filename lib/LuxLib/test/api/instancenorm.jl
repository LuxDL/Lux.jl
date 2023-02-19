using CUDA, Random, Statistics, Test
using LuxLib

include("../test_utils.jl")

rng = MersenneTwister(0)

function _setup_instancenorm(T, sz; affine::Bool=true)
    x = randn(T, sz)
    scale = affine ? ones(T, sz[end - 1]) : nothing
    bias = affine ? zeros(T, sz[end - 1]) : nothing
    return x, scale, bias
end

_istraining(::Val{training}) where {training} = training

@testset "Instance Normalization" begin
    if cpu_testing()
        for T in (Float16, Float32, Float64),
            sz in ((4, 4, 6, 2), (3, 4, 2), (4, 4, 4, 3, 2)),
            training in (Val(true), Val(false)),
            affine in (true, false)

            println("IN_CPU: $T $sz $training $affine")

            _f = (args...) -> instancenorm(args...; epsilon, training)

            epsilon = T(1e-5)
            x, scale, bias = _setup_instancenorm(T, sz; affine)
            @time y, nt = _f(x, scale, bias)

            @inferred instancenorm(x, scale, bias; epsilon, training)
            run_JET_tests(_f, x, scale, bias)
            @test y isa Array{T, length(sz)}
            @test size(y) == sz

            _target_std = ones(ntuple(_ -> 1, length(sz) - 2)..., size(x)[(end - 1):end]...)
            if length(sz) != 3
                @test isapprox(std(y; dims=1:(length(sz) - 2)), _target_std; atol=0.2)
            else
                @test_broken isapprox(std(y; dims=1:(length(sz) - 2)), _target_std;
                                      atol=0.2)
            end
            @test std(y; dims=1:(length(sz) - 2)) != std(x; dims=1:(length(sz) - 2))

            Zygote.gradient(sum ∘ first ∘ _f, x, scale, bias)  # Compile
            @time gs_x, gs_scale, gs_bias, = Zygote.gradient(sum ∘ first ∘ _f, x, scale,
                                                             bias)

            if T != Float16
                if affine
                    __f = (args...) -> sum(first(instancenorm(x, args...; epsilon,
                                                              training)))
                    test_gradient_correctness_fdm(__f, scale, bias; atol=1.0f-2,
                                                  rtol=1.0f-2)
                else
                    __f = (args...) -> sum(first(instancenorm(args..., scale, bias; epsilon,
                                                              training)))
                    test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)
                end
            end
        end
    end

    if gpu_testing()
        for T in (Float16, Float32, Float64),
            sz in ((4, 4, 6, 2), (3, 4, 2), (4, 4, 4, 3, 2)),
            training in (Val(true), Val(false)),
            affine in (true, false)

            println("IN_GPU: $T $sz $training $affine")

            _f = (args...) -> instancenorm(args...; epsilon, training)

            epsilon = T(1e-5)
            x, scale, bias = _setup_instancenorm(T, sz; affine)

            x, scale, bias = (x, scale, bias) .|> cu
            x = x .|> T
            if scale !== nothing
                scale = scale .|> T
                bias = bias .|> T
            end

            CUDA.@time y, nt = _f(x, scale, bias)

            @inferred instancenorm(x, scale, bias; epsilon, training)
            run_JET_tests(_f, x, scale, bias)
            @test y isa CuArray{T, length(sz)}
            @test size(y) == sz

            _target_std = ones(ntuple(_ -> 1, length(sz) - 2)..., size(x)[(end - 1):end]...)
            if length(sz) != 3
                @test isapprox(std(Array(y); dims=1:(length(sz) - 2)), _target_std;
                               atol=0.2)
            else
                @test_broken isapprox(std(Array(y); dims=1:(length(sz) - 2)), _target_std;
                                      atol=0.2)
            end
            @test std(Array(y); dims=1:(length(sz) - 2)) !=
                  std(Array(x); dims=1:(length(sz) - 2))

            Zygote.gradient(sum ∘ first ∘ _f, x, scale, bias)  # Compile
            @time gs_x, gs_scale, gs_bias, = Zygote.gradient(sum ∘ first ∘ _f, x, scale,
                                                             bias)

            # if T != Float16
            #     if affine
            #         __f = (args...) -> sum(first(instancenorm(x, args...; epsilon,
            #                                                   training)))
            #         test_gradient_correctness_fdm(__f, scale, bias; atol=1.0f-2,
            #                                       rtol=1.0f-2)
            #     else
            #         __f = (args...) -> sum(first(instancenorm(args..., scale, bias; epsilon,
            #                                                   training)))
            #         test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)
            #     end
            # end
        end
    end
end
