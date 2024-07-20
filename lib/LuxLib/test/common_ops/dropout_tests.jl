@testitem "Dropout" tags=[:common_ops] setup=[SharedTestSetup] begin
    using Statistics

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, on_gpu) in MODES
        for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            x = randn(rng, T, x_shape) |> aType

            @inferred dropout(rng, x, T(0.5), Val(true), T(2), Colon())

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(true), T(2), Colon())

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_

            __f = x -> sum(first(dropout(StableRNG(0), x, 0.5, Val(true), 2.0, Colon())))
            @test size(only(@inferred(Zygote.gradient(__f, x)))) == size(x)

            __f = let rng = rng, T = T
                x -> sum(first(dropout(rng, x, T(0.5), Val(true), T(2), Colon())))
            end

            allow_unstable() do
                @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 gpu_testing=$on_gpu skip_finite_differences=$(T ==
                                                                                                                    Float16)
            end

            if !on_gpu
                ∂x_zyg = only(Zygote.gradient(__f, x))
                ∂x_enz = zero.(x)
                Enzyme.autodiff(
                    Reverse, sum ∘ first ∘ dropout, Const(rng), Duplicated(x, ∂x_enz),
                    Const(T(0.5)), Const(Val(true)), Const(T(2)), Const(Colon()))
                @test ∂x_zyg≈∂x_enz atol=1.0f-3 rtol=1.0f-3
            end

            @jet sum(first(dropout(rng, x, T(0.5), Val(true), T(2), Colon())))
            @inferred dropout(rng, x, T(0.5), Val(true), T(2), Colon())

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(false), T(2), Colon())

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end
end

@testitem "Dropout with Preset Mask" tags=[:common_ops] setup=[SharedTestSetup] begin
    Enzyme.API.runtimeActivity!(true)  # TODO: remove in 1.0 after deprecation

    using Statistics

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, on_gpu) in MODES
        for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            x = randn(rng, T, x_shape) |> aType
            mask = rand(T, x_shape) |> aType

            # Update mask
            @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(true), T(2), Colon())

            y, mask_, rng_ = dropout(
                rng, x, mask, T(0.5), Val(true), Val(true), T(2), Colon())

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_
            @test mask != mask_

            __f = (x, mask) -> sum(first(dropout(
                StableRNG(0), x, mask, 0.5, Val(true), Val(true), 2.0, Colon())))
            @test size(first(@inferred(Zygote.gradient(__f, x, mask)))) == size(x)

            __f = let rng = rng, mask = mask
                x -> sum(first(dropout(
                    rng, x, mask, T(0.5), Val(true), Val(true), T(2), Colon())))
            end

            allow_unstable() do
                @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 gpu_testing=$on_gpu skip_finite_differences=$(T ==
                                                                                                                    Float16)
            end

            # Upstream bug: https://github.com/EnzymeAD/Enzyme.jl/issues/1651
            if !on_gpu && !(Sys.iswindows() && T == Float16)
                ∂x_zyg = only(Zygote.gradient(__f, x))
                ∂x_enz = zero.(x)
                Enzyme.autodiff(
                    Reverse, sum ∘ first ∘ dropout, Const(rng), Duplicated(x, ∂x_enz),
                    Const(mask), Const(T(0.5)), Const(Val(true)),
                    Const(Val(true)), Const(T(2)), Const(Colon()))
                @test ∂x_zyg≈∂x_enz atol=1.0f-3 rtol=1.0f-3
            end

            @jet sum(first(dropout(
                rng, x, mask, T(0.5), Val(true), Val(true), T(2), Colon())))

            # Try using mask if possible (possible!!)
            @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())

            y, mask_, rng_ = dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng == rng_
            @test mask == mask_

            __f = (x, mask) -> sum(first(dropout(
                StableRNG(0), x, mask, 0.5, Val(true), Val(false), 2.0, Colon())))
            # Branching based on runtime values
            @test_broken size(first(@inferred(Zygote.gradient(__f, x, mask)))) == size(x)

            __f = let rng = rng, mask = mask
                x -> sum(first(dropout(
                    rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())))
            end

            allow_unstable() do
                @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 gpu_testing=$on_gpu skip_finite_differences=$(T ==
                                                                                                                    Float16)
            end

            if !on_gpu
                ∂x_zyg = only(Zygote.gradient(__f, x))
                ∂x_enz = Enzyme.gradient(Reverse, __f, x)
                @test ∂x_zyg≈∂x_enz atol=1.0f-3 rtol=1.0f-3
            end

            @jet sum(first(dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())))
            mask = rand(T, (x_shape[1:(end - 1)]..., 13)) |> aType

            # Try using mask if possible (not possible!!)
            @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())

            y, mask_, rng_ = dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_
            @test mask != mask_

            __f = (x, mask) -> sum(first(dropout(
                StableRNG(0), x, mask, 0.5, Val(true), Val(false), 2.0, Colon())))
            # Branching based on runtime activity
            @test_broken size(first(@inferred(Zygote.gradient(__f, x, mask)))) == size(x)

            __f = let rng = rng, mask = mask
                x -> sum(first(dropout(
                    rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())))
            end

            allow_unstable() do
                @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 gpu_testing=$on_gpu skip_finite_differences=$(T ==
                                                                                                                    Float16)
            end

            if !on_gpu
                ∂x_zyg = only(Zygote.gradient(__f, x))
                ∂x_enz = zero.(x)
                Enzyme.autodiff(
                    Reverse, sum ∘ first ∘ dropout, Const(rng), Duplicated(x, ∂x_enz),
                    Const(mask), Const(T(0.5)), Const(Val(true)),
                    Const(Val(false)), Const(T(2)), Const(Colon()))
                @test ∂x_zyg≈∂x_enz atol=1.0f-3 rtol=1.0f-3
            end

            @jet sum(first(dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())))
            # Testing Mode
            @inferred dropout(rng, x, mask, T(0.5), Val(false), Val(false), T(2), Colon())

            y, mask_, rng_ = dropout(
                rng, x, mask, T(0.5), Val(false), Val(false), T(2), Colon())

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T, length(x_shape)}
            @test mask_ == mask
            @test rng == rng_
        end
    end
end

@testitem "Alpha Dropout" tags=[:common_ops] setup=[SharedTestSetup] begin
    using Statistics

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, on_gpu) in MODES
        for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            x = randn(rng, T, x_shape) |> aType

            @inferred alpha_dropout(rng, x, T(0.5), Val(true))

            y, rng_ = alpha_dropout(rng, x, T(0.5), Val(true))

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng != rng_

            @test_broken isapprox(std(y), std(x); atol=1.0f-2, rtol=1.0f-2)

            __f = x -> sum(first(alpha_dropout(StableRNG(0), x, 0.5, Val(true))))
            @test size(only(@inferred(Zygote.gradient(__f, x)))) == size(x)

            __f = let rng = rng
                x -> sum(first(alpha_dropout(rng, x, T(0.5), Val(true))))
            end

            allow_unstable() do
                @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 gpu_testing=$on_gpu skip_finite_differences=$(T ==
                                                                                                                    Float16)
            end

            if !on_gpu
                ∂x_zyg = only(Zygote.gradient(__f, x))
                ∂x_enz = zero.(x)
                Enzyme.autodiff(Reverse, sum ∘ first ∘ alpha_dropout, Const(rng),
                    Duplicated(x, ∂x_enz), Const(T(0.5)), Const(Val(true)))
                @test ∂x_zyg≈∂x_enz atol=1.0f-3 rtol=1.0f-3
            end

            @jet sum(first(alpha_dropout(rng, x, T(0.5), Val(true))))

            @inferred alpha_dropout(rng, x, T(0.5), Val(false))

            y, rng_ = alpha_dropout(rng, x, T(0.5), Val(false))

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end
end
