using LuxCUDA, Statistics, Test
using LuxLib

include("../test_utils.jl")

rng = get_stable_rng(12345)

function _setup_instancenorm(aType, T, sz; affine::Bool=true)
    x = randn(T, sz) |> aType
    scale = affine ? aType(ones(T, sz[end - 1])) : nothing
    bias = affine ? aType(zeros(T, sz[end - 1])) : nothing
    return x, scale, bias
end

@testset "$mode: Instance Norm" for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        sz in ((4, 4, 6, 2), (3, 4, 2), (4, 4, 4, 3, 2)),
        training in (Val(true), Val(false)),
        affine in (true, false)

        _f = (args...) -> instancenorm(args...; epsilon, training)

        epsilon = T(1e-5)
        x, scale, bias = _setup_instancenorm(aType, T, sz; affine)

        y, nt = instancenorm(x, scale, bias; epsilon, training)

        @inferred instancenorm(x, scale, bias; epsilon, training)
        @jet _f(x, scale, bias)
        @test y isa aType{T, length(sz)}
        @test size(y) == sz

        _target_std = ones(ntuple(_ -> 1, length(sz) - 2)..., size(x)[(end - 1):end]...)
        @eval @test check_approx(std(Array($y); dims=1:($(length(sz) - 2))),
            $_target_std;
            atol=0.2,
            rtol=0.2)
        @test std(y; dims=1:(length(sz) - 2)) != std(x; dims=1:(length(sz) - 2))

        if __istraining(training)
            fp16 = T == Float16
            if affine
                __f = (args...) -> sum(first(instancenorm(x, args...; epsilon, training)))
                @eval @test_gradients $__f $scale $bias soft_fail=$fp16 atol=1.0f-2 rtol=1.0f-2 gpu_testing=$on_gpu
            end
        end
    end
end
