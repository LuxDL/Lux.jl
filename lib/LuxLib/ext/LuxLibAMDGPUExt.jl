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

for (wT, xT) in [(Float64, Float64), (Float64, Float32), (Float32, Float64)],
    fname in (:fused_conv_bias_activation, :__generic_conv_bias_activation)

    for bT in (Float32, Float64)
        @eval begin
            function LuxLib.$fname(σ::F, weight::ROCArray{$(wT), N}, x::ROCArray{$(xT), N},
                    b::ROCArray{$(bT), N}, cdims::NNlib.ConvDims) where {F, N}
                @warn "MIOpen doesn't support Float64 convolutions, type-casting \
                       everything to Float32 to avoid runtime errors" maxlog=1
                return LuxLib._ofeltype_array(Float64,
                    LuxLib.$fname(σ, LuxLib._ofeltype_array(Float32, weight),
                        LuxLib._ofeltype_array(Float32, x),
                        LuxLib._ofeltype_array(Float32, b), cdims))
            end
        end
    end

    @eval begin
        function LuxLib.$fname(σ::F, weight::ROCArray{$(wT), N}, x::ROCArray{$(xT), N},
                b::Nothing, cdims::NNlib.ConvDims) where {F, N}
            @warn "MIOpen doesn't support Float64 convolutions, type-casting everything \
                   to Float32 to avoid runtime errors" maxlog=1
            return LuxLib._ofeltype_array(Float64,
                LuxLib.$fname(σ, LuxLib._ofeltype_array(Float32, weight),
                    LuxLib._ofeltype_array(Float32, x), b, cdims))
        end
    end
end

end
