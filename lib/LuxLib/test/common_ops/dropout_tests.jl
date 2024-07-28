@testitem "Dropout" tags=[:other_ops] setup=[SharedTestSetup] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$T: $x_shape" for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            x = randn(rng, T, x_shape) |> aType

            @test @inferred(dropout(rng, x, T(0.5), Val(true), T(2), Colon())) isa Any

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(true), T(2), Colon())

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_

            @jet sum(first(dropout(rng, x, T(0.5), Val(true), T(2), Colon())))
            @test @inferred(dropout(rng, x, T(0.5), Val(true), T(2), Colon())) isa Any

            __f = x -> sum(first(dropout(StableRNG(0), x, 0.5, Val(true), 2.0, Colon())))
            @test @inferred(Zygote.gradient(__f, x)) isa Any

            __f = let rng = rng, T = T
                x -> sum(first(dropout(rng, x, T(0.5), Val(true), T(2), Colon())))
            end
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=(T == Float16 ? [AutoFiniteDiff()] : []))

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(false), T(2), Colon())

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end
end

@testitem "Dropout with Preset Mask" tags=[:other_ops] setup=[SharedTestSetup] begin
    Enzyme.API.runtimeActivity!(true)  # TODO: remove in 1.0 after deprecation

    using Statistics

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$T: $x_shape" for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            x = randn(rng, T, x_shape) |> aType
            mask = rand(T, x_shape) |> aType

            # Update mask
            @test @inferred(dropout(
                rng, x, mask, T(0.5), Val(true), Val(true), T(2), Colon())) isa Any

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
            @test @inferred(Zygote.gradient(__f, x, mask)) isa Any

            __f = let rng = rng, mask = mask
                x -> sum(first(dropout(
                    rng, x, mask, T(0.5), Val(true), Val(true), T(2), Colon())))
            end
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=(T == Float16 ? [AutoFiniteDiff()] : []))

            @jet sum(first(dropout(
                rng, x, mask, T(0.5), Val(true), Val(true), T(2), Colon())))

            # Try using mask if possible (possible!!)
            @test @inferred(dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())) isa Any

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
            @test @inferred(Zygote.gradient(__f, x, mask)) isa Any broken=true

            __f = let rng = rng, mask = mask
                x -> sum(first(dropout(
                    rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())))
            end
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=(T == Float16 ? [AutoFiniteDiff()] : []))

            @jet sum(first(dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())))
            mask = rand(T, (x_shape[1:(end - 1)]..., 13)) |> aType

            # Try using mask if possible (not possible!!)
            @test @inferred(dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())) isa Any

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
            @test @inferred(Zygote.gradient(__f, x, mask)) isa Any broken=true

            __f = let rng = rng, mask = mask
                x -> sum(first(dropout(
                    rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())))
            end
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=(T == Float16 ? [AutoFiniteDiff()] : []))

            @jet sum(first(dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), Colon())))
            # Testing Mode
            @test @inferred(dropout(
                rng, x, mask, T(0.5), Val(false), Val(false), T(2), Colon())) isa Any

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

@testitem "Alpha Dropout" tags=[:other_ops] setup=[SharedTestSetup] begin
    using Statistics

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$T: $x_shape" for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            x = randn(rng, T, x_shape) |> aType

            @test @inferred(alpha_dropout(rng, x, T(0.5), Val(true))) isa Any

            y, rng_ = alpha_dropout(rng, x, T(0.5), Val(true))

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng != rng_

            @test_broken std(y)≈std(x) atol=1.0f-2 rtol=1.0f-2

            __f = x -> sum(first(alpha_dropout(StableRNG(0), x, 0.5, Val(true))))
            @test @inferred(Zygote.gradient(__f, x)) isa Any

            __f = let rng = rng
                x -> sum(first(alpha_dropout(rng, x, T(0.5), Val(true))))
            end
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=(T == Float16 ? [AutoFiniteDiff()] : []))

            @jet sum(first(alpha_dropout(rng, x, T(0.5), Val(true))))
            @test @inferred(alpha_dropout(rng, x, T(0.5), Val(false))) isa Any

            y, rng_ = alpha_dropout(rng, x, T(0.5), Val(false))

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end
end
