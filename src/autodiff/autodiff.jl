module AutoDiffInternalImpl

using ArgCheck: @argcheck
using ADTypes: AbstractADType, AutoForwardDiff
using ChainRulesCore: ChainRulesCore, HasReverseMode, NoTangent, RuleConfig
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using Functors: fmap
using LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer
using MLDataDevices: get_device, get_device_type, CPUDevice

using ..Lux: Lux, StatefulLuxLayer

const CRC = ChainRulesCore

# Forward declare the functions we want to extend
function vector_jacobian_product end
function vector_jacobian_product_impl end

function jacobian_vector_product end
function jacobian_vector_product_impl end

function batched_jacobian end
function batched_jacobian_impl end

include("utils.jl")

include("jacvec_product.jl")
include("batched_autodiff.jl")
include("nested_autodiff.jl")

include("forwarddiff.jl")

end
