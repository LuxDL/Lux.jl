module LuxAMDGPU

using Reexport

@reexport using AMDGPU, NNlib, ROCKernels

const USE_AMDGPU = Ref{Union{Nothing, Bool}}(nothing)

function check_use_amdgpu!()
    USE_AMDGPU[] === nothing || return

    USE_AMDGPU[] = AMDGPU.functional()
    if USE_AMDGPU[]
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
    check_use_amdgpu!()
    return USE_AMDGPU[]
end

end
