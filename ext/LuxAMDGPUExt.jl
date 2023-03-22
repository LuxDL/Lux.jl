module LuxAMDGPUExt

isdefined(Base, :get_extension) ? (using LuxAMDGPU) : (using ..LuxAMDGPU)
using Adapt, ChainRulesCore, Functors, Lux, NNlib, Random
using Lux: GPU_BACKEND, LuxAMDGPUAdaptor, LuxCPUAdaptor
import Adapt: adapt, adapt_storage
import ChainRulesCore as CRC

__init__() = LuxAMDGPU.functional() && Lux.set_backend_if_higher_priority!(LuxAMDGPUAdaptor())

# device.jl
function _check_amd_gpu()
    if !LuxAMDGPU.functional()
        @warn """
        `gpu` function is called for AMDGPU but LuxAMDGPU.jl is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return nothing
end

CRC.@non_differentiable _check_amd_gpu()
CRC.@non_differentiable LuxAMDGPU.functional()

function Lux.gpu(::LuxAMDGPUAdaptor, x)
    _check_amd_gpu()
    return LuxAMDGPU.functional() ?
           fmap(x -> adapt(LuxAMDGPUAdaptor(), x), x; exclude=Lux._isleaf) : x
end

adapt_storage(::LuxAMDGPUAdaptor, x) = roc(x)
adapt_storage(::LuxAMDGPUAdaptor, rng::AbstractRNG) = rng

adapt_storage(::LuxCPUAdaptor, rng::AMDGPU.rocRAND.RNG) = Random.default_rng()

# utils.jl
Lux.replicate(rng::AMDGPU.rocRAND.RNG) = deepcopy(rng)

@inline function Lux._init_hidden_state(rng::AbstractRNG, rnn, x::ROCArray)
    return adapt(LuxAMDGPUAdaptor(), rnn.init_state(rng, rnn.out_dims, size(x, 2)))
end

@inline function Lux._conv(x::SubArray{T, N, <:ROCArray}, weight, cdims) where {T, N}
    return conv(copy(x), weight, cdims)
end

@inline function Lux._conv_transpose(x::SubArray{T, N, <:ROCArray}, weight,
                                     cdims) where {T, N}
    return ∇conv_data(copy(x), weight, cdims)
end

function Lux._conv_dims(x::ROCArray, weight::ROCArray; padding, stride, dilation, groups)
    return DenseConvDims(DenseConvDims(x, weight; stride, padding, dilation, groups);
                         F=true)
end

function Lux._conv_transpose_dims(x::ROCArray, weight::ROCArray; padding, stride, dilation,
                                  groups)
    # Calculate size of "input", from ∇conv_data()'s perspective...
    combined_pad = (padding[1:2:end] .+ padding[2:2:end])
    I = (size(x)[1:(end - 2)] .- 1) .* stride .+ 1 .+
        (size(weight)[1:(end - 2)] .- 1) .* dilation .- combined_pad
    C_in = size(weight)[end - 1] * groups
    batch_size = size(x)[end]
    # Create DenseConvDims() that looks like the corresponding conv()
    w_size = size(weight)
    return DenseConvDims((I..., C_in, batch_size), w_size; stride, padding, dilation,
                         groups, flipkernel=true)
end

# chainrules.jl
CRC.rrule(::Type{Array}, x::ROCArray) = Array(x), d -> (NoTangent(), roc(d))

function CRC.rrule(::typeof(adapt_storage), to::LuxCPUAdaptor, x::ROCArray)
    return adapt_storage(to, x),
           d -> (NoTangent(), NoTangent(), adapt_storage(LuxAMDGPUAdaptor(), d))
end

function CRC.rrule(::typeof(adapt_storage), to::LuxAMDGPUAdaptor, x::Array)
    return adapt_storage(to, x),
           d -> (NoTangent(), NoTangent(), adapt_storage(LuxCPUAdaptor(), d))
end

end