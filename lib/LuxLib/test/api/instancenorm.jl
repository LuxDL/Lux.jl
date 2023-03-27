using LuxCUDA, Random, Statistics, Test
using LuxLib

include("../test_utils.jl")

rng = MersenneTwister(0)

function _setup_instancenorm(aType, T, sz; affine::Bool=true)
    x = randn(T, sz) |> aType
    scale = affine ? aType(ones(T, sz[end - 1])) : nothing
    bias = affine ? aType(zeros(T, sz[end - 1])) : nothing
    return x, scale, bias
end

@testset "Instance Normalization" begin for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        sz in ((4, 4, 6, 2), (3, 4, 2), (4, 4, 4, 3, 2)),
        training in (Val(true), Val(false)),
        affine in (true, false)

        _f = (args...) -> instancenorm(args...; epsilon, training)

        epsilon = T(1e-5)
        x, scale, bias = _setup_instancenorm(aType, T, sz; affine)

        y, nt = instancenorm(x, scale, bias; epsilon, training)

        @inferred instancenorm(x, scale, bias; epsilon, training)
        run_JET_tests(_f, x, scale, bias)
        @test y isa aType{T, length(sz)}
        @test size(y) == sz

        _target_std = ones(ntuple(_ -> 1, length(sz) - 2)..., size(x)[(end - 1):end]...)
        if length(sz) != 3
            @test isapprox(std(Array(y); dims=1:(length(sz) - 2)), _target_std; atol=0.2)
        else
            @test_broken isapprox(std(Array(y); dims=1:(length(sz) - 2)), _target_std;
                                  atol=0.2)
        end
        @test std(y; dims=1:(length(sz) - 2)) != std(x; dims=1:(length(sz) - 2))

        if __istraining(training)
            if affine
                __f = (args...) -> sum(first(instancenorm(args...; epsilon, training)))
                test_gradient_correctness(__f, x, scale, bias; gpu_testing=on_gpu,
                                          skip_fdm=T == Float16, atol=1.0f-2, rtol=1.0f-2,
                                          soft_fail=T == Float16)
            else
                __f = (args...) -> sum(first(instancenorm(args..., scale, bias; epsilon,
                                                          training)))
                test_gradient_correctness(__f, x; gpu_testing=on_gpu, skip_fdm=T == Float16,
                                          atol=1.0f-2, rtol=1.0f-2, soft_fail=T == Float16)
            end
        end
    end
end end
