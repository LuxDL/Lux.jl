module LuxLib

using Compat: @compat
using Reexport: @reexport
using Static: Static, known
using UnrolledUtilities: unrolled_filter

using ChainRulesCore: ChainRulesCore, NoTangent

using LuxCore: LuxCore
using MLDataDevices: get_device_type, AbstractGPUDevice
using NNlib: NNlib

const Optional{T} = Union{Nothing, T}
const Numeric = Union{AbstractArray{<:T}, T} where {T <: Number}
const ∂∅ = NoTangent()
const CRC = ChainRulesCore

include("utils.jl")
include("traits.jl")
include("impl/Impl.jl")
include("api/API.jl")

@compat(public,
    (internal_operation_mode, GenericBroadcastOp, GPUBroadcastOp, LoopedArrayOp))

end
