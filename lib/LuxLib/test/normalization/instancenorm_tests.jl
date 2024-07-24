@testitem "Instance Normalization" tags=[:normalization] setup=[SharedTestSetup] begin
    using Statistics

    __istraining(::Val{training}) where {training} = training

    rng = StableRNG(12345)

    function _setup_instancenorm(aType, T, sz; affine::Bool=true)
        x = __generate_fixed_array(T, sz) |> aType
        scale = affine ? aType(__generate_fixed_array(T, sz[end - 1])) : nothing
        bias = affine ? aType(__generate_fixed_array(T, sz[end - 1])) : nothing
        return x, scale, bias
    end

    anonact = x -> x^3

    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $act" for T in (Float16, Float32, Float64),
            sz in ((4, 4, 6, 2), (3, 4, 2), (4, 4, 4, 3, 2)),
            training in (Val(true), Val(false)),
            affine in (true, false),
            act in (identity, relu, tanh_fast, sigmoid_fast, anonact)

            _f = (args...) -> instancenorm(args..., training, act, epsilon)

            epsilon = LuxLib.__default_epsilon(T)
            x, scale, bias = _setup_instancenorm(aType, T, sz; affine)

            y, nt = instancenorm(x, scale, bias, training, act, epsilon)

            @test @inferred(instancenorm(x, scale, bias, training, act, epsilon)) isa Any
            @jet instancenorm(x, scale, bias, training, act, epsilon)

            @test y isa aType{T, length(sz)}
            @test size(y) == sz

            fp16 = T == Float16
            atol = fp16 ? 1.0f-2 : 1.0f-3
            rtol = fp16 ? 1.0f-2 : 1.0f-3

            if __istraining(training) && affine
                __f = (args...) -> sum(first(instancenorm(
                    x, args..., training, act, epsilon)))
                skip_fd = act === relu
                allow_unstable() do
                    @eval @test_gradients $__f $scale $bias soft_fail=$fp16 atol=$atol rtol=$rtol gpu_testing=$on_gpu skip_finite_differences=$(skip_fd)
                end
            end

            if anonact !== act
                lfn = (x, sc, b, tr, act, ϵ) -> sum(first(instancenorm(
                    x, sc, b, tr, act, ϵ)))
                @test @inferred(Zygote.gradient(
                    lfn, x, scale, bias, training, act, epsilon)) isa Any
            end

            if !on_gpu && !fp16 && __istraining(training) && affine
                __f = (args...) -> sum(first(instancenorm(args..., training, act, epsilon)))
                ∂x, ∂scale, ∂bias = Zygote.gradient(__f, x, scale, bias)

                ∂x_enz = Enzyme.make_zero(x)
                ∂scale_enz = Enzyme.make_zero(scale)
                ∂bias_enz = Enzyme.make_zero(bias)
                Enzyme.autodiff(Reverse, __f, Active, Duplicated(x, ∂x_enz),
                    Duplicated(scale, ∂scale_enz), Duplicated(bias, ∂bias_enz))

                @test ∂x≈∂x_enz rtol=rtol atol=atol
                @test ∂scale≈∂scale_enz rtol=rtol atol=atol
                @test ∂bias≈∂bias_enz rtol=rtol atol=atol
            end
        end
    end
end
