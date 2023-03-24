using CUDA, Random, Test
using LuxLib

include("../test_utils.jl")

rng = MersenneTwister(0)

function _setup_batchnorm(T, sz; affine::Bool=true, track_stats::Bool)
    x = randn(T, sz)
    scale = affine ? randn(T, sz[end - 1]) : nothing
    bias = affine ? randn(T, sz[end - 1]) : nothing

    if track_stats
        running_mean = randn(T, sz[end - 1])
        running_var = abs2.(randn(T, sz[end - 1]))
        return x, scale, bias, running_mean, running_var
    else
        return x, scale, bias, nothing, nothing
    end
end

@testset "Batch Normalization" begin
    if cpu_testing()
        for T in (Float16, Float32, Float64),
            sz in ((4, 4, 6, 2), (8, 2), (4, 4, 4, 3, 2)),
            training in (Val(true), Val(false)),
            affine in (true, false),
            track_stats in (true, false)

            println("BN_CPU: $T $(sz) $training $affine $track_stats")

            _f = (args...) -> batchnorm(args...; epsilon, training, momentum=T(0.9))

            epsilon = T(1e-5)
            x, scale, bias, rm, rv = _setup_batchnorm(T, sz; track_stats, affine)
            @time y, nt = _f(x, scale, bias, rm, rv)

            @inferred batchnorm(x, scale, bias, rm, rv; epsilon, training, momentum=T(0.9))
            run_JET_tests(_f, x, scale, bias, rm, rv)
            @test y isa Array{T, length(sz)}
            @test size(y) == sz
            if rm !== nothing
                @test size(nt.running_mean) == (size(x, length(sz) - 1),)
                @test size(nt.running_var) == (size(x, length(sz) - 1),)
            end

            Zygote.gradient(sum ∘ first ∘ _f, x, scale, bias, rm, rv)  # Compile
            @time gs_x, gs_scale, gs_bias, _, _ = Zygote.gradient(sum ∘ first ∘ _f, x,
                                                                  scale, bias, rm, rv)

            if T != Float16
                if affine
                    __f = (args...) -> sum(first(batchnorm(x, args..., rm, rv; epsilon,
                                                           training, momentum=T(0.9))))
                    test_gradient_correctness_fdm(__f, scale, bias; atol=1.0f-2,
                                                  rtol=1.0f-2)
                else
                    __f = (args...) -> sum(first(batchnorm(args..., scale, bias, rm, rv;
                                                           epsilon, training,
                                                           momentum=T(0.9))))
                    test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)
                end
            end
        end
    end

    if cuda_testing()
        for T in (Float32, Float64),
            sz in ((4, 4, 6, 2), (8, 2), (4, 4, 4, 3, 2)),
            training in (Val(true), Val(false)),
            affine in (true, false),
            track_stats in (true, false)

            println("BN_GPU: $T $(sz) $training $affine $track_stats")

            _f = (args...) -> batchnorm(args...; epsilon, training, momentum=T(0.9))

            epsilon = T(1e-5)
            x, scale, bias, rm, rv = _setup_batchnorm(T, sz; track_stats, affine)

            x, scale, bias, rm, rv = (x, scale, bias, rm, rv) .|> cu
            x = x .|> T
            if scale !== nothing
                scale = scale .|> T
                bias = bias .|> T
            end
            if rm !== nothing
                rm = rm .|> T
                rv = rv .|> T
            end

            CUDA.@time y, nt = _f(x, scale, bias, rm, rv)

            @inferred batchnorm(x, scale, bias, rm, rv; epsilon, training, momentum=T(0.9))
            run_JET_tests(_f, x, scale, bias, rm, rv)
            @test y isa CuArray{T, length(sz)}
            @test size(y) == sz
            if rm !== nothing
                @test size(nt.running_mean) == (size(x, length(sz) - 1),)
                @test size(nt.running_var) == (size(x, length(sz) - 1),)
            end

            Zygote.gradient(sum ∘ first ∘ _f, x, scale, bias, rm, rv)  # Compile
            CUDA.@time gs_x, gs_scale, gs_bias, _, _ = Zygote.gradient(sum ∘ first ∘ _f, x,
                                                                       scale, bias, rm, rv)

            # if T != Float16
            #     if affine
            #         __f = (args...) -> sum(first(batchnorm(args..., rm, rv; epsilon,
            #                                                training, momentum=T(0.9))))
            #         test_gradient_correctness_fdm(__f, x, scale, bias; atol=1.0f-2,
            #                                       rtol=1.0f-2)
            #     else
            #         __f = (args...) -> sum(first(batchnorm(args..., scale, bias, rm, rv;
            #                                                epsilon, training,
            #                                                momentum=T(0.9))))
            #         test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)
            #     end
            # end
        end
    end
end
