module LuxLibTrackerAMDGPUExt

using AMDGPU: AMDGPU
using LuxLib: LuxLib, Optional
using NNlib: NNlib, ConvDims, PoolDims
using Tracker: Tracker, TrackedArray

const ROCTrackedArray{T, N} = TrackedArray{T, N, <:AMDGPU.ROCArray{T, N}}

# Taken from https://github.com/FluxML/NNlib.jl/blob/07833637dec96d12d0614308d3145b432fdb320a/ext/NNlibAMDGPUExt/NNlibAMDGPUExt.jl#L38
function nnlib_padding(dims)
    pd = NNlib.padding(dims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn """
        MIOpen does not support asymmetric padding, defaulting to symmetric choice:
        $pd -> $(pd[1:2:end]).
        """ maxlog=1
    end
    return pd[1:2:end]
end

# For meanpool and maxpool NNlib directly defines the rrules so we need to define special
# rules for Tracker
for poolname in (:maxpool, :meanpool)
    @eval begin
        Tracker.@grad function NNlib.$(poolname)(
                x_tracked::ROCTrackedArray{<:AMDGPU.MIOpen.MIOPENFloat, N},
                pdims::PoolDims) where {N}
            x = Tracker.data(x_tracked)
            y = similar(
                x, NNlib.output_size(pdims)..., NNlib.channels_out(pdims), size(x, N))
            nd = max(0, 4 - N)
            npdims = NNlib.insert_singleton_spatial_dimension(pdims, nd)

            # `workspace` is used in the pullback.
            _, workspace = AMDGPU.MIOpen.$(Symbol("$(poolname)!"))(
                NNlib.insert_singleton_spatial_dimension(y, nd),
                NNlib.insert_singleton_spatial_dimension(x, nd);
                dims=NNlib.kernel_size(npdims),
                padding=nnlib_padding(npdims), stride=NNlib.stride(npdims))

            function ∇pooling(Δ)
                dx = similar(x)
                AMDGPU.MIOpen.$(Symbol("∇$(poolname)!"))(
                    NNlib.insert_singleton_spatial_dimension(dx, nd),
                    NNlib.insert_singleton_spatial_dimension(Δ, nd),
                    NNlib.insert_singleton_spatial_dimension(y, nd),
                    NNlib.insert_singleton_spatial_dimension(x, nd);
                    dims=NNlib.kernel_size(npdims), padding=nnlib_padding(npdims),
                    stride=NNlib.stride(npdims), workspace)
                return Tracker.nobacksies($(Expr(:quote, poolname)), (dx, nothing))
            end

            return y, ∇pooling
        end
    end
end

function LuxLib.__generic_conv_bias_activation(
        act::F, weight::ROCTrackedArray{Float64, N}, x::ROCTrackedArray{Float64, N},
        bias::Optional{<:ROCTrackedArray{Float64, 1}}, cdims::ConvDims) where {N, F}
    return LuxLib._ofeltype_array(Float64,
        LuxLib.__generic_conv_bias_activation(act, LuxLib._ofeltype_array(Float32, weight),
            LuxLib._ofeltype_array(Float32, x),
            LuxLib._ofeltype_array(Float32, bias), cdims))
end

end
