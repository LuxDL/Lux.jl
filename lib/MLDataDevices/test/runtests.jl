using Aqua, SafeTestsets, Test, Pkg
using LuxCore, LuxDeviceUtils

const GROUP = get(ENV, "GROUP", "CUDA")

@testset "LuxDeviceUtils Tests" begin
    if GROUP == "CUDA"
        @safetestset "CUDA" begin
            include("cuda.jl")
        end
    end

    if GROUP == "AMDGPU"
        @safetestset "CUDA" begin
            include("amdgpu.jl")
        end
    end

    if GROUP == "Metal"
        @safetestset "Metal" begin
            include("metal.jl")
        end
    end

    @testset "Others" begin
        @testset "Aqua Tests" begin
            Aqua.test_all(LuxDeviceUtils)
        end
        @safetestset "Component Arrays" begin
            include("component_arrays.jl")
        end
    end
end
