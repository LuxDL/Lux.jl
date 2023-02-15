# Deprecations for v0.5

## Device transfer of AbstractExplicitLayers
function cpu(l::AbstractExplicitLayer)
    Base.depwarn("`cpu` on a layer has been deprecated and will be removed in v0.5. Apply" *
                 " `cpu` on the layer's parameters and states instead.", :cpu)
    return l
end

function gpu(l::AbstractExplicitLayer)
    Base.depwarn("`gpu` on a layer has been deprecated and will be removed in v0.5. Apply" *
                 " `gpu` on the layer's parameters and states instead.", :gpu)
    return l
end

## Trainmode/Testmode with argument
function trainmode(st::NamedTuple, mode::Bool)
    Base.depwarn("Setting `mode` for `trainmode` is deprecated and will be removed in v0.5.",
                 :trainmode)
    return mode ? trainmode(st) : testmode(st)
end

function testmode(st::NamedTuple, mode::Bool)
    Base.depwarn("Setting `mode` for testmode is deprecated and will be removed in v0.5",
                 :testmode)
    return mode ? testmode(st) : trainmode(st)
end

## Fallback `initialparameters` / `initialstates`
function initialparameters(::AbstractRNG, l::Any)
    Base.depwarn("Default fallback for non `AbstractExplicitLayer` types are deprecated" *
                 " and will be removed in v0.5. Define" *
                 " `Lux.initialparameters(::AbstractRNG, ::$(typeof(l)))`",
                 :initialparameters)
    return NamedTuple()
end

function initialstates(::AbstractRNG, l::Any)
    Base.depwarn("Default fallback for non `AbstractExplicitLayer` types are deprecated" *
                 "and will be removed in v0.5. Define" *
                 " `Lux.initialstates(::AbstractRNG, ::$(typeof(l)))`", :initialstates)
    return NamedTuple()
end

## Fallback `parameterlength` / `statelength`
function parameterlength(x::Any)
    Base.depwarn("Fallback for `parameterlength` of type $(typeof(x)) is deprecated." *
                 " This will generate an error from v0.5.", :parameterlength)
    return 0
end

function statelength(x::Any)
    Base.depwarn("Fallback for `statelength` of type $(typeof(x)) is deprecated." *
                 " This will generate an error from v0.5.", :statelength)
    return 0
end

## Layers
"""
    ActivationFunction(f)

Broadcast `f` on the input.

## Arguments

  - `f`: Activation function

## Inputs

  - `x`: Any array type s.t. `f` can be broadcasted over it

## Returns

  - Broadcasted Activation `f.(x)`
  - Empty `NamedTuple()`

!!! warning
    
    This layer is deprecated and will be removed in v0.5. Use [`WrappedFunction`](@ref) with
    manual broadcasting
"""
function ActivationFunction(f)
    Base.depwarn("`Lux.ActivationFunction(f)` has been deprecated and will be removed in" *
                 " v0.5. Use `Lux.WrappedFunction(x -> f.(x))` instead.",
                 :ActivationFunction)
    return WrappedFunction(Base.Fix1(broadcast, f))
end

"""
    applyactivation(f::Function, x::AbstractArray)

Apply the function `f` on `x` elementwise, i.e. `f.(x)`. Dispatches to CUDNN if possible.

!!! warning
    
    This function has been deprecated. Use `f.(x)` instead.
"""
@inline function applyactivation(f::Function, x::AbstractArray)
    Base.depwarn("`Lux.applyactivation` has been deprecated and will be removed in" *
                 " v0.5. Directly apply broadcasting instead.", :applyactivation)
    return f.(x)
end

"""
    elementwise_add(x, y)

Computes `x .+ y`. Dispatches to CUDNN if possible.

!!! warning
    
    This function has been deprecated. Use `x .+ y` instead.
"""
@inline function elementwise_add(x, y)
    Base.depwarn("`Lux.elementwise_add` has been deprecated and will be removed in" *
                 " v0.5. Use `x .+ y` instead.", :elementwise_add)
    return x .+ y
end

"""
    elementwise_mul(x, y)

Computes `x .* y`. Dispatches to CUDNN if possible.

!!! warning
    
    This function has been deprecated. Use `x .* y` instead.
"""
@inline function elementwise_mul(x, y)
    Base.depwarn("`Lux.elementwise_mul` has been deprecated and will be removed in" *
                 " v0.5. Use `x .* y` instead.", :elementwise_mul)
    return x .* y
end

# Dropout
"""
    dropout(rng::AbstractRNG, x, p, q, dims, ::Val{training})
    dropout(rng::AbstractRNG, x, mask, p, q, dims, t::Val{training}, ::Val{update_mask})

If `training` then dropout is applied on `x` with probability `p` along `dims`. If `mask` is
passed it is used if `update_mask` is false. If `update_mask` is true then the mask is
generated and used.

!!! warning
    
    This function has been deprecated and will be removed in v0.5. Use `LuxLib.dropout`
    instead.
"""
@inline function dropout(rng::AbstractRNG, x, p, q, dims, t::Val)
    # Deprecated Functionality (Remove in v0.5)
    Base.depwarn("`Lux.dropout` has been deprecated and will be removed in v0.5. Use " *
                 "`LuxLib.dropout` instead.", :dropout)

    return LuxLib.dropout(rng, x, p, t; invp=q, dims)
end

@inline function dropout(rng::AbstractRNG, x, mask, p, q, dims, t::Val, um::Val)
    # Deprecated Functionality (Remove in v0.5)
    Base.depwarn("`Lux.dropout` has been deprecated and will be removed in v0.5. Use " *
                 "`LuxLib.dropout` instead.", :dropout)

    return (LuxLib.dropout(rng, x, mask, p, t, um; invp=q, dims)..., Val(false))
end

# Normalization Implementation
"""
    normalization(x, running_mean, running_var, scale, bias, activation, reduce_dims,
                  ::Val{training}, momentum, epsilon)

Performs BatchNorm/GroupNorm based on input configuration

!!! warning
    
    This function has been deprecated and will be removed in v0.5. Use
    `LuxLib.(batch/group)norm` instead.
"""
@inline function normalization(x::AbstractArray{T, N},
                               running_mean::Union{Nothing, AbstractVector{T}},
                               running_var::Union{Nothing, AbstractVector{T}},
                               scale::Union{Nothing, AbstractVector{T}},
                               bias::Union{Nothing, AbstractVector{T}}, activation,
                               reduce_dims, t::Val, momentum::T=T(0.1),
                               epsilon::T=T(1e-5)) where {T, N}
    # Deprecated Functionality (Remove in v0.5)
    Base.depwarn("`Lux.normalization` has been deprecated and will be removed in v0.5. " *
                 "Use `LuxLib.(batch/group)norm` instead.", :normalization)

    return activation.(LuxLib._normalization(x, running_mean, running_var, scale, bias,
                                             reduce_dims, t, momentum, epsilon))
end
