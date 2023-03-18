module LuxAMDGPU

using Reexport

@reexport using AMDGPU, ROCKernels

const USE_AMD_GPU = Ref{Union{Nothing, Bool}}(nothing)

function _check_use_amdgpu!()
    USE_AMD_GPU[] === nothing || return

    USE_AMD_GPU[] = AMDGPU.functional()
    if USE_AMD_GPU[]
        if !AMDGPU.functional(:MIOpen)
            @warn """
            MIOpen is not functional in AMDGPU.jl, some functionality will not be available.
            """ maxlog=1
        end
    else
        @info """
        The AMDGPU function is being called but the AMDGPU is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return
end

"""
    functional()

Check if LuxAMDGPU is functional.
"""
function functional()::Bool
    _check_use_amdgpu!()
    return USE_AMD_GPU[]
end

end
