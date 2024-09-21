@testitem "internal_operation_mode: Wrapped Arrays" tags=[:others] setup=[SharedTestSetup] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        x = rand(Float32, 4, 3) |> aType
        retval = ongpu ? LuxLib.GPUBroadcastOp : LuxLib.LoopedArrayOp
        @test LuxLib.internal_operation_mode(x) isa retval
    end

    using StaticArrays, JLArrays

    x = rand(Float32, 4, 3) |> JLArray
    @test LuxLib.internal_operation_mode(x) isa LuxLib.GenericBroadcastOp

    x = @SArray rand(Float32, 4, 3)
    @test LuxLib.internal_operation_mode(x) isa LuxLib.GenericBroadcastOp

    x = reshape(@SArray(rand(Float32, 4)), :, 1)
    @test LuxLib.internal_operation_mode(x) isa LuxLib.GenericBroadcastOp
end

@testitem "Matmul: StaticArrays" tags=[:others] setup=[SharedTestSetup] begin
    using LuxLib.Impl: matmuladd
    using StaticArrays

    A = rand(2, 2)
    bias = rand(2)

    # This works with LoopVectorization
    B = ones(SMatrix{2, 1, Float64})
    @test matmuladd(A, B, bias) ≈ A * B .+ bias

    b = ones(SVector{2, Float64})
    @test matmuladd(A, b, bias) ≈ A * b .+ bias
end
