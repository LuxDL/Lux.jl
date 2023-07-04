using LuxCUDA, Test

@testset "LuxCUDA" begin
    @test LuxCUDA.USE_CUDA_GPU[] === nothing

    @test LuxCUDA.functional() isa Bool

    if VERSION ≥ v"1.9"
        @test !@isdefined(NNlibCUDA)
    end
end
