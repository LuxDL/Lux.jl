using LuxCUDA, Random, Statistics, Test
using LuxLib

include("../test_utils.jl")

rng = MersenneTwister(0)

@testset "Dropout" begin for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

        x = randn(rng, T, x_shape) |> aType

        @inferred dropout(rng, x, T(0.5), Val(true); dims=Colon())

        y, mask_, rng_ = dropout(rng, x, T(0.5), Val(true); dims=Colon())

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test mask_ isa aType{T, length(x_shape)}
        @test size(mask_) == x_shape
        @test rng != rng_

        __f = x -> sum(first(dropout(rng, x, T(0.5), Val(true); dims=Colon())))
        test_gradient_correctness(__f, x; gpu_testing=on_gpu, atol=1.0f-2, rtol=1.0f-2,
                                  soft_fail=T == Float16)
        run_JET_tests(__f, x)

        @inferred dropout(rng, x, T(0.5), Val(true); dims=Colon())

        y, mask_, rng_ = dropout(rng, x, T(0.5), Val(false); dims=Colon())

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test rng == rng_
        @test y == x
    end
end end

@testset "Dropout with Preset Mask" begin for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

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
        test_gradient_correctness(__f, x; gpu_testing=on_gpu, atol=1.0f-2, rtol=1.0f-2,
                                  soft_fail=T == Float16)
        run_JET_tests(__f, x)

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
        test_gradient_correctness(__f, x; gpu_testing=on_gpu, atol=1.0f-2, rtol=1.0f-2,
                                  soft_fail=T == Float16)
        run_JET_tests(__f, x)

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
        test_gradient_correctness(__f, x; gpu_testing=on_gpu, atol=1.0f-2, rtol=1.0f-2,
                                  soft_fail=T == Float16)
        run_JET_tests(__f, x)

        # Testing Mode
        @inferred dropout(rng, x, mask, T(0.5), Val(false), Val(false); dims=Colon())

        y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(false), Val(false); dims=Colon())

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test mask_ isa aType{T, length(x_shape)}
        @test mask_ == mask
        @test rng == rng_
    end
end end

@testset "Alpha Dropout" begin for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

        x = randn(rng, T, x_shape) |> aType

        @inferred alpha_dropout(rng, x, T(0.5), Val(true))

        y, rng_ = alpha_dropout(rng, x, T(0.5), Val(true))

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test rng != rng_
        @test_broken isapprox(std(y), std(x); atol=1.0f-2, rtol=1.0f-2)

        __f = x -> sum(first(alpha_dropout(rng, x, T(0.5), Val(true))))
        test_gradient_correctness(__f, x; gpu_testing=on_gpu, atol=1.0f-2, rtol=1.0f-2,
                                  soft_fail=T == Float16)
        run_JET_tests(__f, x)

        @inferred alpha_dropout(rng, x, T(0.5), Val(false))

        y, rng_ = alpha_dropout(rng, x, T(0.5), Val(false))

        @test y isa aType{T, length(x_shape)}
        @test size(y) == x_shape
        @test rng == rng_
        @test y == x
    end
end end
