module LuxLib

using Compat: @compat
using Preferences: @load_preference
using Reexport: @reexport
using Static: Static, known

using ChainRulesCore: ChainRulesCore, NoTangent

using LuxCore: LuxCore
using MLDataDevices: get_device_type, AbstractGPUDevice
using NNlib: NNlib

const Optional{T} = Union{Nothing, T}
const Numeric = Union{AbstractArray{<:T}, T} where {T <: Number}
const ∂∅ = NoTangent()
const CRC = ChainRulesCore

const DISABLE_LOOP_VECTORIZATION = @load_preference("disable_loop_vectorization", false)

include("utils.jl")
include("traits.jl")
include("impl/Impl.jl")
include("api/API.jl")

@compat(public,
    (internal_operation_mode, GenericBroadcastOp, GPUBroadcastOp, LoopedArrayOp))

end
