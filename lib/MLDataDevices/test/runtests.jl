using SafeTestsets, Test
using LuxCore, LuxDeviceUtils

@static if VERSION ≥ v"1.9"
    using Pkg
    Pkg.add("LuxAMDGPU")
end

@testset "LuxDeviceUtils Tests" begin
    @safetestset "LuxCUDA" begin
        include("luxcuda.jl")
    end

    @static if VERSION ≥ v"1.9"
        @safetestset "LuxAMDGPU" begin
            include("luxamdgpu.jl")
        end
    end
end
