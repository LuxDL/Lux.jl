@testitem "Group Normalization" tags=[:normalization] setup=[SharedTestSetup] timeout=3600 begin
    rng = StableRNG(12345)

    function _setup_groupnorm(aType, T, sz, groups)
        x = __generate_fixed_array(T, sz) |> aType
        scale = __generate_fixed_array(T, sz[end - 1]) |> aType
        bias = __generate_fixed_array(T, sz[end - 1]) |> aType
        return x, scale, bias
    end

    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, ngroups $groups, $act" for T in (
                Float16, Float32, Float64),
            sz in ((4, 6, 2), (8, 8, 8, 6, 2), (3, 16, 16, 12, 2),
                (4, 4, 6, 2), (2, 2, 6, 2), (3, 3, 12, 4)),
            groups in (2, 3),
            act in (identity, relu, tanh_fast, sigmoid_fast, x -> x^3)

            _f = (args...) -> groupnorm(args..., groups, act, epsilon)

            epsilon = T(1e-5)
            x, scale, bias = _setup_groupnorm(aType, T, sz, groups)
            y = _f(x, scale, bias)

            @inferred groupnorm(x, scale, bias, groups, act, epsilon)

            # Stresses CI too much
            T !== Float16 && @jet groupnorm(x, scale, bias, groups, act, epsilon)

            @test y isa aType{T, length(sz)}
            @test size(y) == sz

            fp16 = T == Float16
            __f = (args...) -> sum(groupnorm(x, args..., groups, act, epsilon))
            skip_fd = act === relu
            @eval @test_gradients $__f $scale $bias gpu_testing=$on_gpu atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16 skip_finite_differences=$(skip_fd)
        end
    end
end
