using Aqua, SafeTestsets, Test, LuxDeviceUtils, TestSetExtensions

const GROUP = get(ENV, "GROUP", "NONE")

@testset ExtendedTestSet "LuxDeviceUtils Tests" begin
    if GROUP == "CUDA" || GROUP == "ALL"
        @safetestset "CUDA" include("cuda.jl")
    end

    if GROUP == "AMDGPU" || GROUP == "ALL"
        @safetestset "AMDGPU" include("amdgpu.jl")
    end

    if GROUP == "Metal" || GROUP == "ALL"
        @safetestset "Metal" include("metal.jl")
    end

    if GROUP == "oneAPI" || GROUP == "ALL"
        @safetestset "oneAPI" include("oneapi.jl")
    end

    @testset "Others" begin
        @testset "Aqua Tests" Aqua.test_all(LuxDeviceUtils)

        @safetestset "Component Arrays" include("component_arrays.jl")

        @safetestset "Explicit Imports" include("explicit_imports.jl")
    end
end
