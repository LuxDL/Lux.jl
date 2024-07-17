@testitem "Batch Normalization" tags=[:normalization] setup=[SharedTestSetup] begin
    rng = StableRNG(12345)

    function _setup_batchnorm(aType, T, sz; affine::Bool=true, track_stats::Bool)
        x = __generate_fixed_array(T, sz) |> aType
        scale = affine ? aType(__generate_fixed_array(T, sz[end - 1])) : nothing
        bias = affine ? aType(__generate_fixed_array(T, sz[end - 1])) : nothing

        if track_stats
            running_mean = __generate_fixed_array(T, sz[end - 1]) |> aType
            running_var = abs2.(__generate_fixed_array(T, sz[end - 1])) |> aType
            return x, scale, bias, running_mean, running_var
        else
            return x, scale, bias, nothing, nothing
        end
    end

    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $act" for T in (Float16, Float32, Float64),
            sz in ((4, 4, 6, 2), (8, 2), (4, 4, 4, 3, 2)),
            training in (Val(true), Val(false)),
            affine in (true, false),
            track_stats in (true, false),
            act in (identity, relu, tanh_fast, sigmoid_fast, x -> x^3)

            _f = (args...) -> batchnorm(args..., training, act, T(0.9), epsilon)

            epsilon = T(1e-5)
            x, scale, bias, rm, rv = _setup_batchnorm(aType, T, sz; track_stats, affine)

            y, nt = batchnorm(x, scale, bias, rm, rv, training, act, T(0.9), epsilon)

            @inferred batchnorm(x, scale, bias, rm, rv, training, act, T(0.9), epsilon)
            @jet batchnorm(x, scale, bias, rm, rv, training, act, T(0.9), epsilon)

            @test y isa aType{T, length(sz)}
            @test size(y) == sz
            if rm !== nothing
                @test size(nt.running_mean) == (size(x, length(sz) - 1),)
                @test size(nt.running_var) == (size(x, length(sz) - 1),)
            end

            if __istraining(training) && affine
                fp16 = T == Float16
                __f = (args...) -> sum(first(batchnorm(
                    x, args..., rm, rv, training, act, T(0.9), epsilon)))
                skip_fd = act === relu
                allow_unstable() do
                    @eval @test_gradients $__f $scale $bias gpu_testing=$on_gpu soft_fail=$fp16 atol=1.0f-2 rtol=1.0f-2 skip_finite_differences=$(skip_fd)
                end
            end
        end

        @testset "mixed precision" begin
            # Needed specifically for cudnn batchnorm
            x = rand(Float64, 4, 4, 6, 2) |> aType
            scale = rand(Float32, 6) |> aType
            bias = rand(Float32, 6) |> aType
            running_mean = rand(Float32, 6) |> aType
            running_var = rand(Float32, 6) |> aType

            y, nt = batchnorm(x, scale, bias, running_mean, running_var,
                Val(true), identity, 0.9f0, 1.0f-5)
            @test y isa aType{Float64, 4}
            @test nt.running_mean isa aType && length(nt.running_mean) == 6
            @test nt.running_var isa aType && length(nt.running_var) == 6

            __f = (args...) -> sum(first(batchnorm(
                x, args..., running_mean, running_var, Val(true), identity, 0.9f0, 1.0f-5)))
            allow_unstable() do
                @eval @test_gradients $__f $scale $bias gpu_testing=$on_gpu soft_fail=true atol=1.0f-2 rtol=1.0f-2
            end
        end
    end
end
