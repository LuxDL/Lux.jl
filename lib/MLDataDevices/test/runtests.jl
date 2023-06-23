using SafeTestsets, Test
using LuxCore, LuxDeviceUtils

@testset "LuxDeviceUtils Tests" begin
    @safetestset "LuxCUDA" begin
        include("luxcuda.jl")
    end

    @safetestset "LuxAMDGPU" begin
        include("luxamdgpu.jl")
    end
end
