module LuxLibAMDGPUExt

using LuxLib: LuxLib
using NNlib: NNlib
using AMDGPU: AMDGPU, ROCArray

const MIOPENFloat = Union{Float16, Float32}

# NNlib incorrectly defines some of the broadcasting rules. Probably this should be
# upstreamed to NNlib
@static if AMDGPU.functional(:MIOpen)
    # Just define for dims = 6 , 7, 8 and hope no one uses it beyond that
    for f in [NNlib.relu, NNlib.relu6, NNlib.softplus, NNlib.σ, Base.tanh], N in (6, 7, 8)
        @eval function Base.materialize(bc::Broadcast.Broadcasted{
                <:Any, <:Any, typeof($f), <:Tuple{ROCArray{<:MIOPENFloat, $N}}})
            return copy(bc)
        end
    end
end

end
