module LuxLib

using SciMLPublic: @public
using Preferences: load_preference
using Reexport: @reexport
using Static: Static, known
using UUIDs: UUID

using ChainRulesCore: ChainRulesCore, NoTangent

using LuxCore: LuxCore
using MLDataDevices: get_device_type, AbstractGPUDevice, ReactantDevice
using NNlib: NNlib

const Optional{T} = Union{Nothing,T}
const Numeric = Union{AbstractArray{<:T},T} where {T<:Number}
const ∂∅ = NoTangent()
const CRC = ChainRulesCore

const LuxLibUUID = UUID("82251201-b29d-42c6-8e01-566dec8acb11")

const DISABLE_LOOP_VECTORIZATION = load_preference(
    LuxLibUUID, "disable_loop_vectorization", false
)

include("utils.jl")
include("traits.jl")
include("impl/Impl.jl")
include("api/API.jl")

@public (internal_operation_mode, GenericBroadcastOp, GPUBroadcastOp, LoopedArrayOp)

end
