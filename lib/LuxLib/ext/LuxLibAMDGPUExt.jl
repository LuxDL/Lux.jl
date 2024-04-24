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

@inline function LuxLib.fused_conv_bias_activation(
        σ::F, weight::ROCArray{Float64, N}, x::ROCArray{Float64, N},
        b::ROCArray{Float64, N}, cdims::NNlib.ConvDims) where {F, N}
    @warn "MIOpen doesn't support Float64 convolutions, type-casting everything to Float32 \
           to avoid runtime errors" maxlog=1
    return LuxLib._oftype_array(Float64,
        LuxLib.fused_conv_bias_activation(
            σ, LuxLib._oftype_array(Float32, weight), LuxLib._oftype_array(Float32, x),
            LuxLib._oftype_array(Float32, b), cdims))
end

@inline function LuxLib.fused_conv_bias_activation(
        σ::F, weight::ROCArray{Float64, N}, x::ROCArray{Float64, N},
        b::Nothing, cdims::NNlib.ConvDims) where {F, N}
    @warn "MIOpen doesn't support Float64 convolutions, type-casting everything to Float32 \
           to avoid runtime errors" maxlog=1
    return LuxLib._oftype_array(Float64,
        LuxLib.fused_conv_bias_activation(σ, LuxLib._oftype_array(Float32, weight),
            LuxLib._oftype_array(Float32, x), b, cdims))
end

@inline function LuxLib.__generic_conv_bias_activation(
        act::F, weight::ROCArray{Float64, N}, x::ROCArray{Float64, N},
        bias::ROCArray{Float64, N}, cdims::NNlib.ConvDims) where {N, F}
    return LuxLib._oftype_array(Float64,
        LuxLib.__generic_conv_bias_activation(
            act, LuxLib._oftype_array(Float32, weight), LuxLib._oftype_array(Float32, x),
            LuxLib._oftype_array(Float32, bias), cdims))
end

@inline function LuxLib.__generic_conv_bias_activation(
        act::F, weight::ROCArray{Float64, N}, x::ROCArray{Float64, N},
        bias::Nothing, cdims::NNlib.ConvDims) where {N, F}
    return LuxLib._oftype_array(Float64,
        LuxLib.__generic_conv_bias_activation(act, LuxLib._oftype_array(Float32, weight),
            LuxLib._oftype_array(Float32, x), bias, cdims))
end

end
