using LuxCUDA, Random, Test
using LuxLib

include("../test_utils.jl")

rng = MersenneTwister(0)

function _setup_batchnorm(aType, T, sz; affine::Bool=true, track_stats::Bool)
    x = randn(T, sz) |> aType
    scale = affine ? aType(randn(T, sz[end - 1])) : nothing
    bias = affine ? aType(randn(T, sz[end - 1])) : nothing

    if track_stats
        running_mean = randn(T, sz[end - 1]) |> aType
        running_var = abs2.(randn(T, sz[end - 1])) |> aType
        return x, scale, bias, running_mean, running_var
    else
        return x, scale, bias, nothing, nothing
    end
end

@testset "Batch Normalization" begin for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        sz in ((4, 4, 6, 2), (8, 2), (4, 4, 4, 3, 2)),
        training in (Val(true), Val(false)),
        affine in (true, false),
        track_stats in (true, false)

        _f = (args...) -> batchnorm(args...; epsilon, training, momentum=T(0.9))

        epsilon = T(1e-5)
        x, scale, bias, rm, rv = _setup_batchnorm(aType, T, sz; track_stats, affine)

        y, nt = batchnorm(x, scale, bias, rm, rv; epsilon, training, momentum=T(0.9))

        @inferred batchnorm(x, scale, bias, rm, rv; epsilon, training, momentum=T(0.9))
        run_JET_tests(_f, x, scale, bias, rm, rv)

        @test y isa aType{T, length(sz)}
        @test size(y) == sz

        if rm !== nothing
            @test size(nt.running_mean) == (size(x, length(sz) - 1),)
            @test size(nt.running_var) == (size(x, length(sz) - 1),)
        end

        if __istraining(training)
            if affine
                __f = (args...) -> sum(first(batchnorm(args..., rm, rv; epsilon, training,
                                                       momentum=T(0.9))))
                test_gradient_correctness(__f, x, scale, bias; gpu_testing=on_gpu,
                                          skip_fdm=T == Float16, atol=1.0f-2, rtol=1.0f-2)
            else
                __f = (args...) -> sum(first(batchnorm(args..., scale, bias, rm, rv;
                                                       epsilon, training, momentum=T(0.9))))
                test_gradient_correctness(__f, x; gpu_testing=on_gpu, skip_fdm=T == Float16,
                                          atol=1.0f-2, rtol=1.0f-2)
            end
        end
    end
end end
