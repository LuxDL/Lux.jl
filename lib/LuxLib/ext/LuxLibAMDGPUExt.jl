module LuxLibAMDGPUExt

isdefined(Base, :get_extension) ? (using LuxAMDGPU) : (using ..LuxAMDGPU)
using LuxLib

# utils.jl
@static if VERSION < v"1.7"
    # KA.get_device is not present in <= v0.7 but that is what works on julia 1.6
    LuxLib.get_device(x::ROCArray) = ROCDevice()
end

LuxLib._replicate(rng::AMDGPU.rocRAND.RNG) = deepcopy(rng)

end