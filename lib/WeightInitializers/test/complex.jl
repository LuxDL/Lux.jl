using WeightInitializers, Test

@testset "Kaiming Uniform: Complex" begin
    x = kaiming_uniform(ComplexF32, 1024, 1024)
    @test eltype(x) == ComplexF32
    @test size(x) == (1024, 1024)
    @test minimum(imag.(x)) < 0.0
end
