using Test
using LuxCore, LuxDeviceUtils
using LuxAMDGPU, LuxCUDA  # Accelerators
using FillArrays, Zygote  # Extensions

@testset "LuxDeviceUtils Tests" begin
    @test 1 + 1 == 2
end
