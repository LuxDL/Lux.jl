module LuxLib

using Compat: @compat
using Reexport: @reexport
using Static: Static, StaticBool, True, False, static, known
using UnrolledUtilities: unrolled_filter, unrolled_mapreduce

using ChainRulesCore: ChainRulesCore, NoTangent

using LuxCore: LuxCore
using MLDataDevices: get_device_type, AMDGPUDevice, CUDADevice, CPUDevice,
                     AbstractGPUDevice, AbstractDevice

@reexport using NNlib

const Optional{T} = Union{Nothing, T}
const Numeric = Union{AbstractArray{<:T}, T} where {T <: Number}
const ∂∅ = NoTangent()
const CRC = ChainRulesCore

include("utils.jl")
include("traits.jl")

include("impl/Impl.jl")

include("api/API.jl")

export fast_activation, fast_activation!!

@compat(public,
    (internal_operation_mode, GenericBroadcastOp, GPUBroadcastOp, LoopedArrayOp))

end
