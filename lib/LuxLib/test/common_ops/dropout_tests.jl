@testitem "Dropout" tags=[:other_ops] setup=[SharedTestSetup] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$T, $x_shape, $dims" for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1)),
            dims in (:, 1, (1, 2))

            x = randn(rng, T, x_shape) |> aType

            @test @inferred(dropout(rng, x, T(0.5), Val(true), T(2), dims)) isa Any

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(true), T(2), dims)

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T, length(x_shape)}
            dims isa Colon && @test size(mask_) == x_shape
            @test rng != rng_

            @jet sum(first(dropout(rng, x, T(0.5), Val(true), T(2), dims)))
            @test @inferred(dropout(rng, x, T(0.5), Val(true), T(2), dims)) isa Any

            __f = x -> sum(first(dropout(StableRNG(0), x, 0.5, Val(true), 2.0, dims)))
            @test @inferred(Zygote.gradient(__f, x)) isa Any

            __f = let rng = rng, T = T
                x -> sum(first(dropout(rng, x, T(0.5), Val(true), T(2), dims)))
            end
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3,
                soft_fail=(T == Float16 ? [AutoFiniteDiff()] : []),
                broken_backends=(T == Float16 && Sys.iswindows() ? [AutoEnzyme()] : []))

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(false), T(2), dims)

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end
end

@testitem "Dropout with Preset Mask" tags=[:other_ops] setup=[SharedTestSetup] begin
    using Statistics

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$T: $x_shape" for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            x = randn(rng, T, x_shape) |> aType
            mask = rand(T, x_shape) |> aType

            # Update mask
            @test @inferred(dropout(
                rng, x, mask, T(0.5), Val(true), Val(true), T(2), :)) isa Any

            y, mask_, rng_ = dropout(
                rng, x, mask, T(0.5), Val(true), Val(true), T(2), :)

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_
            @test mask != mask_

            __f = (x, mask) -> sum(first(dropout(
                StableRNG(0), x, mask, 0.5, Val(true), Val(true), 2.0, :)))
            @test @inferred(Zygote.gradient(__f, x, mask)) isa Any

            __f = let rng = rng, mask = mask, p = T(0.5), invp = T(2)
                x -> sum(first(dropout(rng, x, mask, p, Val(true), Val(true), invp, :)))
            end
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3,
                soft_fail=(T == Float16 ? [AutoFiniteDiff()] : []),
                broken_backends=(T == Float16 && Sys.iswindows() ? [AutoEnzyme()] : []))

            @jet sum(first(dropout(
                rng, x, mask, T(0.5), Val(true), Val(true), T(2), :)))

            # Try using mask if possible (possible!!)
            @test @inferred(dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), :)) isa Any

            y, mask_, rng_ = dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), :)

            @test y isa aType{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng == rng_
            @test mask == mask_

            __f = (x, mask) -> sum(first(dropout(
                StableRNG(0), x, mask, 0.5, Val(true), Val(false), 2.0, :)))
            @test @inferred(Zygote.gradient(__f, x, mask)) isa Any

            __f = let rng = rng, mask = mask, p = T(0.5), invp = T(2)
                x -> sum(first(dropout(rng, x, mask, p, Val(true), Val(false), invp, :)))
            end

            soft_fail = T == Float16 ? Any[AutoFiniteDiff()] : []
            skip_backends = length(x_shape) == 5 ? [AutoEnzyme()] : []

            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3, soft_fail, skip_backends,
                broken_backends=(T == Float16 && Sys.iswindows() ? [AutoEnzyme()] : []))

            @jet sum(first(dropout(
                rng, x, mask, T(0.5), Val(true), Val(false), T(2), :)))
            mask = rand(T, (x_shape[1:(end - 1)]..., 13)) |> aType

            # Testing Mode
            @test @inferred(dropout(
                rng, x, mask, T(0.5), Val(false), Val(false), T(2), :)) isa Any

            y, mask_, rng_ = dropout(
                rng, x, mask, T(0.5), Val(false), Val(false), T(2), :)

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
                soft_fail=(T == Float16 ? [AutoFiniteDiff()] : []),
                broken_backends=(T == Float16 && Sys.iswindows() ? [AutoEnzyme()] : []))

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
