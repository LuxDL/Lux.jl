module AutoDiffInternalImpl

using ArrayInterface: ArrayInterface
using ADTypes: AbstractADType, AutoForwardDiff
using ChainRulesCore: ChainRulesCore, HasReverseMode, NoTangent, RuleConfig, ZeroTangent
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using Functors: fmap
using MLDataDevices: get_device, get_device_type, CPUDevice

using ..Lux: Lux, StatefulLuxLayer, Utils
using ..LuxPreferences: AUTOMATIC_NESTED_AD_SWITCHING

const CRC = ChainRulesCore

# Forward declare the functions we want to extend
function vector_jacobian_product end
function vector_jacobian_product_impl end

function jacobian_vector_product end
function jacobian_vector_product_impl end

## Call Structure for `batched_jacobian`
## Users call `batched_jacobian(f, ad, x)`
## This calls either `batched_jacobian_internal(f, ad, x)` or
## `batched_jacobian_internal(f, ad, x, y)` (if `f` is a `ComposedFunction` of correct form)
## We define rrule on `batched_jacobian_internal`.
function batched_jacobian end
function batched_jacobian_internal end
function batched_jacobian_impl end

const AD_CONVERTIBLE_FUNCTIONS = [
    # Input Gradient/Jacobian
    ComposedFunction{<:Any,<:StatefulLuxLayer},
    ComposedFunction{<:StatefulLuxLayer,<:Any},
    StatefulLuxLayer,
    # Parameter Gradient/Jacobian
    ComposedFunction{<:Any,<:Base.Fix1{<:StatefulLuxLayer}},
    ComposedFunction{<:Base.Fix1{<:StatefulLuxLayer},<:Any},
    Base.Fix1{<:StatefulLuxLayer},
]

include("utils.jl")

include("jac_products.jl")
include("batched_autodiff.jl")
include("nested_autodiff.jl")

include("forwarddiff.jl")

end
