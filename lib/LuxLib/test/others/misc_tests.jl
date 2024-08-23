@testitem "internal_operation_mode: Wrapped Arrays" tags=[:others] setup=[SharedTestSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
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
