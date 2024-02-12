using Aqua, LuxCUDA, Test

@testset "LuxCUDA" begin
    @test LuxCUDA.USE_CUDA_GPU[] === nothing

    @test LuxCUDA.functional() isa Bool

    Aqua.test_all(LuxCUDA; ambiguities=false)
    Aqua.test_ambiguities(LuxCUDA)
end
