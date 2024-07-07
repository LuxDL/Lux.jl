@testitem "Instance Normalization" tags=[:normalization] setup=[SharedTestSetup] timeout=3600 begin
    using Statistics

    rng = get_stable_rng(12345)

    function _setup_instancenorm(aType, T, sz; affine::Bool=true)
        x = __generate_fixed_array(T, sz) |> aType
        scale = affine ? aType(__generate_fixed_array(T, sz[end - 1])) : nothing
        bias = affine ? aType(__generate_fixed_array(T, sz[end - 1])) : nothing
        return x, scale, bias
    end

    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $act" for T in (Float16, Float32, Float64),
            sz in ((4, 4, 6, 2), (3, 4, 2), (4, 4, 4, 3, 2)),
            training in (Val(true), Val(false)),
            affine in (true, false),
            act in (identity, relu, tanh_fast, sigmoid_fast, x -> x^3)

            _f = (args...) -> instancenorm(args..., training, act, epsilon)

            epsilon = T(1e-5)
            x, scale, bias = _setup_instancenorm(aType, T, sz; affine)

            y, nt = instancenorm(x, scale, bias, training, act, epsilon)

            @inferred instancenorm(x, scale, bias, training, act, epsilon)
            @jet instancenorm(x, scale, bias, training, act, epsilon)

            @test y isa aType{T, length(sz)}
            @test size(y) == sz

            if !affine && act === identity
                _target_std = ones(
                    ntuple(_ -> 1, length(sz) - 2)..., size(x)[(end - 1):end]...)
                @test check_approx(
                    std(Array(y); dims=1:(length(sz) - 2)), _target_std; atol=0.2, rtol=0.2)
            end
            @test std(y; dims=1:(length(sz) - 2)) != std(x; dims=1:(length(sz) - 2))

            if __istraining(training) && affine
                fp16 = T == Float16
                __f = (args...) -> sum(first(instancenorm(
                    x, args..., training, act, epsilon)))
                skip_fd = act === relu
                @eval @test_gradients $__f $scale $bias soft_fail=$fp16 atol=1.0f-2 rtol=1.0f-2 gpu_testing=$on_gpu skip_finite_differences=$(skip_fd)
            end
        end
    end
end
