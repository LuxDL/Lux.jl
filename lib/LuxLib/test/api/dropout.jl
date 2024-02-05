using Statistics, Test, LuxLib

include("../test_utils.jl")

rng = get_stable_rng(12345)

@testset "$mode: Dropout" for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

        T === Float16 && mode == "AMDGPU" && continue

        x = randn(rng, T, x_shape) |> aType

        @inferred dropout(rng, x, T(0.5), Val(true); dims=Colon())

        y, mask_, rng_ = dropout(rng, x, T(0.5), Val(true); dims=Colon())

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test mask_ isa aType{T, length(x_shape)}
        @test size(mask_) == x_shape
        @test rng != rng_

        __f = x -> sum(first(dropout(rng, x, T(0.5), Val(true); dims=Colon())))

        fp16 = T == Float16
        @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16 gpu_testing=$on_gpu
        @jet sum(first(dropout(rng, x, T(0.5), Val(true); dims=Colon())))

        @inferred dropout(rng, x, T(0.5), Val(true); dims=Colon())

        y, mask_, rng_ = dropout(rng, x, T(0.5), Val(false); dims=Colon())

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test rng == rng_
        @test y == x
    end
end

@testset "$mode: Dropout with Preset Mask" for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

        T === Float16 && mode == "AMDGPU" && continue

        x = randn(rng, T, x_shape) |> aType
        mask = rand(T, x_shape) |> aType

        # Update mask
        @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(true); dims=Colon())

        y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(true); dims=Colon())

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test mask_ isa aType{T, length(x_shape)}
        @test size(mask_) == x_shape
        @test rng != rng_
        @test mask != mask_

        __f = x -> sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(true);
            dims=Colon())))

        fp16 = T == Float16
        @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16 gpu_testing=$on_gpu
        @jet sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(true); dims=Colon())))

        # Try using mask if possible (possible!!)
        @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())

        y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test mask_ isa aType{T, length(x_shape)}
        @test size(mask_) == x_shape
        @test rng == rng_
        @test mask == mask_

        __f = x -> sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(false);
            dims=Colon())))

        fp16 = T == Float16
        @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16 gpu_testing=$on_gpu
        @jet sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())))

        mask = rand(T, (x_shape[1:(end - 1)]..., 13)) |> aType

        # Try using mask if possible (not possible!!)
        @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())

        y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test mask_ isa aType{T, length(x_shape)}
        @test size(mask_) == x_shape
        @test rng != rng_
        @test mask != mask_

        __f = x -> sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(false);
            dims=Colon())))

        fp16 = T == Float16
        @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16 gpu_testing=$on_gpu
        @jet sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())))

        # Testing Mode
        @inferred dropout(rng, x, mask, T(0.5), Val(false), Val(false); dims=Colon())

        y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(false), Val(false); dims=Colon())

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test mask_ isa aType{T, length(x_shape)}
        @test mask_ == mask
        @test rng == rng_
    end
end

@testset "$mode: Alpha Dropout" for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

        T === Float16 && mode == "AMDGPU" && continue

        x = randn(rng, T, x_shape) |> aType

        @inferred alpha_dropout(rng, x, T(0.5), Val(true))

        y, rng_ = alpha_dropout(rng, x, T(0.5), Val(true))

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test rng != rng_

        @test_broken isapprox(std(y), std(x); atol=1.0f-2, rtol=1.0f-2)

        __f = x -> sum(first(alpha_dropout(rng, x, T(0.5), Val(true))))

        fp16 = T == Float16
        @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16 gpu_testing=$on_gpu
        @jet sum(first(alpha_dropout(rng, x, T(0.5), Val(true))))

        @inferred alpha_dropout(rng, x, T(0.5), Val(false))

        y, rng_ = alpha_dropout(rng, x, T(0.5), Val(false))

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test rng == rng_
        @test y == x
    end
end
