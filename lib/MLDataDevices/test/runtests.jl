import Pkg
using Aqua, SafeTestsets, Test, LuxDeviceUtils, TestSetExtensions

const BACKEND_GROUP = get(ENV, "BACKEND_GROUP", "NONE")

@testset ExtendedTestSet "LuxDeviceUtils Tests" begin
    if BACKEND_GROUP == "CUDA" || BACKEND_GROUP == "ALL"
        Pkg.add("LuxCUDA")
        @safetestset "CUDA" include("cuda.jl")
    end

    if BACKEND_GROUP == "AMDGPU" || BACKEND_GROUP == "ALL"
        Pkg.add("AMDGPU")
        @safetestset "AMDGPU" include("amdgpu.jl")
    end

    if BACKEND_GROUP == "Metal" || BACKEND_GROUP == "ALL"
        Pkg.add("Metal")
        @safetestset "Metal" include("metal.jl")
    end

    if BACKEND_GROUP == "oneAPI" || BACKEND_GROUP == "ALL"
        Pkg.add("oneAPI")
        @safetestset "oneAPI" include("oneapi.jl")
    end

    @testset "Others" begin
        @testset "Aqua Tests" Aqua.test_all(LuxDeviceUtils)

        @safetestset "Misc Tests" include("misc.jl")

        @safetestset "Explicit Imports" include("explicit_imports.jl")
    end
end
