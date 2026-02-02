using OneHotArrays, LuxLib, Test, BenchmarkTools

@testset "Specialized OneHotArrays Dispatch" begin
    x = onehotbatch("aabc", "abcdefghijklmnopqrstuv")
    weight = reshape(collect(Float32, 1:(1024 * 22)), 1024, 22)

    dense_res_time = @belapsed fused_dense_bias_activation(
        identity, $(weight), $(Array(x)), nothing
    )
    onehot_res_time = @belapsed fused_dense_bias_activation(
        identity, $(weight), $(x), nothing
    )

    @test onehot_res_time < dense_res_time / 5
    @test fused_dense_bias_activation(identity, weight, x, nothing) ≈
        fused_dense_bias_activation(identity, weight, Array(x), nothing)
end
