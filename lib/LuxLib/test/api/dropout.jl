using CUDA, Random, Statistics, Test
using LuxLib

include("../test_utils.jl")

rng = MersenneTwister(0)

@testset "Dropout" begin
    if cpu_testing()
        for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            println("DRP_CPU: $T $(x_shape)")

            x = randn(rng, T, x_shape)

            @inferred dropout(rng, x, T(0.5), Val(true); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(true); dims=Colon())

            @test y isa Array{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa Array{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_

            __f = x -> sum(first(dropout(rng, x, T(0.5), Val(true); dims=Colon())))
            test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)

            @inferred dropout(rng, x, T(0.5), Val(false); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(false); dims=Colon())

            @test y isa Array{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end

    if gpu_testing()
        for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            println("DRP_GPU: $T $(x_shape)")

            x = T.(cu(randn(rng, T, x_shape)))

            @inferred dropout(rng, x, T(0.5), Val(true); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(true); dims=Colon())

            @test y isa CuArray{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa CuArray{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_

            # __f = x -> sum(first(dropout(rng, x, T(0.5), Val(true); dims=Colon())))
            # test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)

            @inferred dropout(rng, x, T(0.5), Val(false); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, T(0.5), Val(false); dims=Colon())

            @test y isa CuArray{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end
end

@testset "Alpha Dropout" begin
    if cpu_testing()
        for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            println("ADRP_CPU: $T $(x_shape)")

            x = randn(rng, T, x_shape)

            @inferred alpha_dropout(rng, x, T(0.5), Val(true))

            y, rng_ = alpha_dropout(rng, x, T(0.5), Val(true))

            @test y isa Array{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng != rng_
            # @test isapprox(std(y), std(x); atol=0.4, rtol=0.4)

            __f = x -> sum(first(alpha_dropout(rng, x, T(0.5), Val(true))))
            test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)

            @inferred alpha_dropout(rng, x, T(0.5), Val(false))

            y, rng_ = alpha_dropout(rng, x, T(0.5), Val(false))

            @test y isa Array{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end

    if gpu_testing()
        for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            println("ADRP_GPU: $T $(x_shape)")

            x = T.(cu(randn(rng, T, x_shape)))

            @inferred alpha_dropout(rng, x, T(0.5), Val(true))

            y, rng_ = alpha_dropout(rng, x, T(0.5), Val(true))

            @test y isa CuArray{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng != rng_
            # @test isapprox(std(y), std(x); atol=0.4, rtol=0.4)

            # __f = x -> sum(first(alpha_dropout(rng, x, T(0.5), Val(true))))
            # test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)

            @inferred alpha_dropout(rng, x, T(0.5), Val(false))

            y, rng_ = alpha_dropout(rng, x, T(0.5), Val(false))

            @test y isa CuArray{T, length(x_shape)}
            @test size(y) == x_shape
            @test rng == rng_
            @test y == x
        end
    end
end

@testset "Dropout with Preset Mask" begin
    if cpu_testing()
        for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            println("DRP_CPU: $T $(x_shape)")

            x = randn(rng, T, x_shape)
            mask = rand(T, x_shape)

            # Update mask
            @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(true); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(true);
                                     dims=Colon())

            @test y isa Array{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa Array{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_
            @test mask != mask_

            __f = x -> sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(true);
                                         dims=Colon())))
            test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)

            # Try using mask if possible (possible!!)
            @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(false);
                                     dims=Colon())

            @test y isa Array{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa Array{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng == rng_
            @test mask == mask_

            __f = x -> sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(false);
                                         dims=Colon())))
            test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)

            mask = rand(T, (x_shape[1:(end - 1)]..., 13))

            # Try using mask if possible (not possible!!)
            @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(false);
                                     dims=Colon())

            @test y isa Array{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa Array{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_
            @test mask != mask_

            __f = x -> sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(false);
                                         dims=Colon())))
            test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)

            # Testing Mode
            @inferred dropout(rng, x, mask, T(0.5), Val(false), Val(false); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(false), Val(false);
                                     dims=Colon())

            @test y isa Array{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa Array{T, length(x_shape)}
            @test mask_ == mask
            @test rng == rng_
        end
    end

    if gpu_testing()
        for T in (Float16, Float32, Float64),
            x_shape in ((2, 3), (2, 2, 3), (2, 2, 3, 1), (2, 2, 1, 3, 1))

            println("DRP_GPU: $T $(x_shape)")

            x = T.(cu(randn(rng, T, x_shape)))
            mask = T.(cu(rand(rng, T, x_shape)))

            # Update mask
            @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(true); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(true);
                                     dims=Colon())

            @test y isa CuArray{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa CuArray{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_
            @test mask != mask_

            # __f = x -> sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(true); dims=Colon())))
            # test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)

            # Try using mask if possible (possible!!)
            @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(false);
                                     dims=Colon())

            @test y isa CuArray{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa CuArray{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng == rng_
            @test mask == mask_

            # __f = x -> sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())))
            # test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)

            mask = CUDA.rand(T, (x_shape[1:(end - 1)]..., 13))

            # Try using mask if possible (not possible!!)
            @inferred dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(true), Val(false);
                                     dims=Colon())

            @test y isa CuArray{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa CuArray{T, length(x_shape)}
            @test size(mask_) == x_shape
            @test rng != rng_
            @test mask != mask_

            # __f = x -> sum(first(dropout(rng, x, mask, T(0.5), Val(true), Val(false); dims=Colon())))
            # test_gradient_correctness_fdm(__f, x; atol=1.0f-2, rtol=1.0f-2)

            # Testing Mode
            @inferred dropout(rng, x, mask, T(0.5), Val(false), Val(false); dims=Colon())

            y, mask_, rng_ = dropout(rng, x, mask, T(0.5), Val(false), Val(false);
                                     dims=Colon())

            @test y isa CuArray{T, length(x_shape)}
            @test size(y) == x_shape
            @test mask_ isa CuArray{T, length(x_shape)}
            @test mask_ == mask
            @test rng == rng_
        end
    end
end
