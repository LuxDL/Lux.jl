using OneHotArrays, LuxLib, Test

@testset "Specialized OneHotArrays Dispatch" begin
    x = onehotbatch("aabc", "abcdefghijklmnopqrstuv")
    weight = reshape(collect(Float32, 1:(1024 * 22)), 1024, 22)

    @test fused_dense_bias_activation(identity, weight, x, nothing) ≈
        fused_dense_bias_activation(identity, weight, Array(x), nothing)

    @test LuxLib.Utils.force_3arg_mul!_dispatch(weight, weight, x)
end
