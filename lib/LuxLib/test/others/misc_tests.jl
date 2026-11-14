using Test, LuxLib
using LuxLib.Impl: matmuladd
using StaticArrays, JLArrays

include("../shared_testsetup.jl")

@testset "internal_operation_mode" begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        x = aType(rand(Float32, 4, 3))
        retval = ongpu ? LuxLib.GPUBroadcastOp : LuxLib.LoopedArrayOp
        @test LuxLib.internal_operation_mode(x) isa retval
    end
end

@testset "StaticArrays" begin
    x = JLArray(rand(Float32, 4, 3))
    @test LuxLib.internal_operation_mode(x) isa LuxLib.GenericBroadcastOp

    x = @SArray rand(Float32, 4, 3)
    @test LuxLib.internal_operation_mode(x) isa LuxLib.GenericBroadcastOp

    x = reshape(@SArray(rand(Float32, 4)), :, 1)
    @test LuxLib.internal_operation_mode(x) isa LuxLib.GenericBroadcastOp
end

@testset "Matmul: StaticArrays" begin
    A = rand(2, 2)
    bias = rand(2)

    # This works with LoopVectorization
    B = ones(SMatrix{2,1,Float64})
    @test matmuladd(A, B, bias) ≈ A * B .+ bias

    b = ones(SVector{2,Float64})
    @test matmuladd(A, b, bias) ≈ A * b .+ bias
end

@testset "StaticExt backwards compatibility" begin
    using Random, Static, Zygote

    @test LuxLib.Utils.static(Static.True()) === Val(true)
    @test LuxLib.Utils.static(Static.False()) === Val(false)
    @test LuxLib.Utils.static(Static.static(3)) === Val(3)
    @test LuxLib.Utils.static(Static.static(:foo)) === Val(:foo)
    @test LuxLib.Utils.known(Static.True()) === true
    @test LuxLib.Utils.unsafe_known(Static.False()) === false

    rng = Random.Xoshiro(0)
    x = randn(Float32, 4, 3, 2)
    γ = ones(Float32, 3)
    β = zeros(Float32, 3)
    rμ = zeros(Float32, 3)
    rσ² = ones(Float32, 3)

    y_bn, st_bn = batchnorm(x, γ, β, rμ, rσ², Static.True())
    y_bn_ref, _ = batchnorm(x, γ, β, rμ, rσ², Val(true))
    @test y_bn ≈ y_bn_ref

    g_bn = Zygote.gradient(z -> sum(first(batchnorm(z, γ, β, rμ, rσ², Static.True()))), x)[1]
    @test size(g_bn) == size(x)

    y_in, _ = instancenorm(x, γ, β, Static.False())
    y_in_ref, _ = instancenorm(x, γ, β, Val(false))
    @test y_in ≈ y_in_ref

    y_norm, _, _ = LuxLib.Impl.normalization(
        x, rμ, rσ², γ, β, (1, 3), Static.True(), 0.1f0, 1.0f-5
    )
    @test size(y_norm) == size(x)

    y_do, mask_do, _ = dropout(Random.Xoshiro(1), x, 0.5f0, Static.False(), 2.0f0, :)
    @test y_do == x
    y_do2, _, _ = dropout(
        Random.Xoshiro(1), x, mask_do, 0.5f0, Static.True(), Static.False(), 2.0f0, :
    )
    @test size(y_do2) == size(x)
    y_ado, _ = alpha_dropout(Random.Xoshiro(1), x, 0.5f0, Static.False())
    @test y_ado == x
end
