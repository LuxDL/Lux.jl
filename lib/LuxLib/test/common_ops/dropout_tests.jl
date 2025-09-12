@testitem "Dropout" tags = [:misc] setup = [SharedTestSetup] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "$T, $x_shape, $dims" for T in (Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1)),
            dims in (:, 1, (1, 2))

            !fp64 && T == Float64 && continue

            x = aType(randn(rng, T, x_shape))

            @test @inferred(dropout(rng, x, T(0.5), Val(true), T(2), dims)) isa Any

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(true), T(2), dims)

            @test y isa aType{T,length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T,length(x_shape)}
            dims isa Colon && @test size(mask_) == x_shape
            @test rng != rng_

            @jet sum(first(dropout(rng, x, T(0.5), Val(true), T(2), dims)))
            @test @inferred(dropout(rng, x, T(0.5), Val(true), T(2), dims)) isa Any

            @test_gradients(
                sumabs2first,
                dropout,
                rng,
                x,
                T(0.5),
                Val(true),
                T(2),
                dims;
                atol=1.0f-3,
                rtol=1.0f-3
            )

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(false), T(2), dims)

            @test y isa aType{T,length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end
end

@testitem "Dropout with Preset Mask" tags = [:misc] setup = [SharedTestSetup] begin
    using Statistics

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "$T: $x_shape" for T in (Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            !fp64 && T == Float64 && continue

            x = aType(randn(rng, T, x_shape))
            mask = aType(rand(T, x_shape))

            # Update mask
            @test @inferred(
                dropout(rng, x, mask, T(0.5), Val(true), Val(true), T(2), :)
            ) isa Any

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(true), T(2), :)

            @test y isa aType{T,length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T,length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_
            @test mask != mask_

            @test_gradients(
                sumabs2first,
                dropout,
                rng,
                x,
                LuxTestUtils.Constant(mask),
                T(0.5),
                Val(true),
                Val(true),
                T(2),
                :;
                atol=1.0f-3,
                rtol=1.0f-3
            )

            @jet sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(true), T(2), :)))

            # Try using mask if possible (possible!!)
            @test @inferred(
                dropout(rng, x, mask, T(0.5), Val(true), Val(false), T(2), :)
            ) isa Any

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(false), T(2), :)

            @test y isa aType{T,length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T,length(x_shape)}
            @test size(mask_) == x_shape
            @test rng == rng_
            @test mask == mask_

            @test_gradients(
                sumabs2first,
                dropout,
                rng,
                x,
                LuxTestUtils.Constant(mask),
                T(0.5),
                Val(true),
                Val(false),
                T(2),
                :;
                atol=1.0f-3,
                rtol=1.0f-3
            )

            @jet sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(false), T(2), :)))
            mask = aType(rand(T, (x_shape[1:(end - 1)]..., 13)))

            # Testing Mode
            @test @inferred(
                dropout(rng, x, mask, T(0.5), Val(false), Val(false), T(2), :)
            ) isa Any

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(false), Val(false), T(2), :)

            @test y isa aType{T,length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa aType{T,length(x_shape)}
            @test mask_ == mask
            @test rng == rng_
        end
    end
end

@testitem "Alpha Dropout" tags = [:misc] setup = [SharedTestSetup] begin
    using Statistics

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "$T: $x_shape" for T in (Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            !fp64 && T == Float64 && continue

            x = aType(randn(rng, T, x_shape))

            @test @inferred(alpha_dropout(rng, x, T(0.5), Val(true))) isa Any

            y, rng_ = alpha_dropout(rng, x, T(0.5), Val(true))

            @test y isa aType{T,length(x_shape)}
            @test size(y) == x_shape
            @test rng != rng_

            @test_broken std(y) ≈ std(x) atol = 1.0f-2 rtol = 1.0f-2

            @test_gradients(
                sumabs2first,
                alpha_dropout,
                rng,
                x,
                T(0.5),
                Val(true);
                atol=1.0f-3,
                rtol=1.0f-3
            )

            @jet sum(first(alpha_dropout(rng, x, T(0.5), Val(true))))
            @test @inferred(alpha_dropout(rng, x, T(0.5), Val(false))) isa Any

            y, rng_ = alpha_dropout(rng, x, T(0.5), Val(false))

            @test y isa aType{T,length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end
end
