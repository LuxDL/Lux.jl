using SafeTestsets, Test, Pkg
using LuxCore, LuxDeviceUtils

const GROUP = get(ENV, "GROUP", "CUDA")

@info "Installing Accelerator Packages..."

GROUP == "CUDA" && Pkg.add("LuxCUDA")

@static if VERSION ≥ v"1.9"
    GROUP == "AMDGPU" && Pkg.add("LuxAMDGPU")

    GROUP == "Metal" && Pkg.add("Metal")
else
    if GROUP != "CUDA"
        @warn "AMDGPU and Metal are only available on Julia 1.9+"
    end
end

@info "Installed Accelerator Packages!"

@info "Starting Tests..."

@testset "LuxDeviceUtils Tests" begin
    if GROUP == "CUDA"
        @safetestset "CUDA" begin
            include("cuda.jl")
        end
    end

    @static if VERSION ≥ v"1.9"
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
    end
end
