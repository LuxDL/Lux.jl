module LuxCore

using SciMLPublic: @public
using DispatchDoctor: @stable
using Random: Random, AbstractRNG

# PRNG Handling
"""
    replicate(rng::AbstractRNG)

Creates a copy of the `rng` state depending on its type.
"""
@generated function replicate(rng::T) where {T<:AbstractRNG}
    hasmethod(copy, (T,)) && return :(copy(rng))
    return :(deepcopy(rng))
end
function replicate(rng::Random.TaskLocalRNG)
    @warn "`replicate` doesn't work for `TaskLocalRNG`. Returning the same \
           `TaskLocalRNG`." maxlog = 1
    return rng
end

"""
    abstract type AbstractLuxLayer

Abstract Type for all Lux Layers

Users implementing their custom layer, **must** implement

  - `initialparameters(rng::AbstractRNG, layer::CustomAbstractLuxLayer)` -- This
    returns a `NamedTuple` containing the trainable parameters for the layer.
  - `initialstates(rng::AbstractRNG, layer::CustomAbstractLuxLayer)` -- This returns a
    NamedTuple containing the current state for the layer. For most layers this is typically
    empty. Layers that would potentially contain this include `BatchNorm`, `LSTM`, `GRU`,
    etc.

Optionally:

  - `parameterlength(layer::CustomAbstractLuxLayer)` -- These can be automatically
    calculated, but it is recommended that the user defines these.
  - `statelength(layer::CustomAbstractLuxLayer)` -- These can be automatically
    calculated, but it is recommended that the user defines these.

See also [`AbstractLuxContainerLayer`](@ref)
"""
abstract type AbstractLuxLayer end

"""
    initialparameters(rng::AbstractRNG, layer)

Generate the initial parameters of the layer `l`.
"""
function initialparameters end

"""
    initialstates(rng::AbstractRNG, layer)

Generate the initial states of the layer `l`.
"""
function initialstates end

for op in (:initialparameters, :initialstates)
    @eval begin
        $(op)(::AbstractRNG, ::Union{AbstractLuxLayer,Nothing}) = NamedTuple()
        $(op)(rng::AbstractRNG, l::NamedTuple) = map(Base.Fix1($op, rng), l)
        function $(op)(rng::AbstractRNG, l)
            contains_lux_layer(l) || throw(MethodError($op, (rng, l)))
            return Internal.fmap(Base.Fix1($op, rng), l; exclude=Internal.isleaf)
        end
    end
end

"""
    parameterlength(layer)

Return the total number of parameters of the layer `l`.
"""
function parameterlength(l::AbstractLuxLayer)
    return parameterlength(initialparameters(Internal.default_rng(), l))
end
function parameterlength(nt::Union{NamedTuple,Tuple})
    return length(nt) == 0 ? 0 : sum(parameterlength, nt)
end
parameterlength(a::AbstractArray) = length(a)

"""
    statelength(layer)

Return the total number of states of the layer `l`.
"""
statelength(l::AbstractLuxLayer) = statelength(initialstates(Internal.default_rng(), l))
statelength(nt::Union{NamedTuple,Tuple}) = length(nt) == 0 ? 0 : sum(statelength, nt)
statelength(a::AbstractArray) = length(a)
statelength(::Any) = 1

"""
    outputsize(layer, x, rng)

Return the output size of the layer.

The fallback implementation of this function assumes the inputs were batched, i.e.,
if any of the outputs are Arrays, with `ndims(A) > 1`, it will return
`size(A)[1:(end - 1)]`. If this behavior is undesirable, provide a custom
`outputsize(layer, x, rng)` implementation).

!!! warning "Fallback Implementation"

    The fallback implementation of this function is defined once `Lux.jl` is loaded.

!!! warning "Changes from Pre-1.0 Behavior"

    Previously it was possible to override this function by defining `outputsize(layer)`.
    However, this can potentially introduce a bug that is hard to bypass. See
    [this PR](https://github.com/LuxDL/LuxCore.jl/pull/43) for more information.
"""
function outputsize end

"""
    setup(rng::AbstractRNG, layer)

Shorthand for getting the parameters and states of the layer `l`. Is equivalent to
`(initialparameters(rng, l), initialstates(rng, l))`.

!!! warning

    This function is not pure, it mutates `rng`.
"""
setup(rng::AbstractRNG, l) = (initialparameters(rng, l), initialstates(rng, l))

"""
    apply(model, x, ps, st)

In most cases this function simply calls `model(x, ps, st)`. However, it is still
recommended to call `apply` instead of `model(x, ps, st)` directly. Some of the reasons for
this include:

 1. For certain types of inputs `x`, we might want to perform preprocessing before calling
    `model`. For eg, if `x` is an Array of `ReverseDiff.TrackedReal`s this can cause
    significant regressions in `model(x, ps, st)` (since it won't hit any of the BLAS
    dispatches). In those cases, we would automatically convert `x` to a
    `ReverseDiff.TrackedArray`.
 2. Certain user defined inputs need to be applied to specific layers but we want the
    datatype of propagate through all the layers (even unsupported ones). In these cases,
    we can unpack the input in `apply` and pass it to the appropriate layer and then
    repack it before returning. See the Lux manual on Custom Input Types for a motivating
    example.

!!! tip

    `apply` is integrated with `DispatchDoctor.jl` that allows automatic verification of
    type stability. By default this is "disable"d. For more information, see the
    [documentation](https://github.com/MilesCranmer/DispatchDoctor.jl).
"""
@stable default_mode = "disable" function apply(model::AbstractLuxLayer, x, ps, st)
    return model(x, ps, st)
end

"""
    stateless_apply(model, x, ps)

Calls `apply` and only returns the first argument. This function requires that `model` has
an empty state of `NamedTuple()`. Behavior of other kinds of models are undefined and it is
the responsibility of the user to ensure that the model has an empty state.
"""
function stateless_apply(model::AbstractLuxLayer, x, ps)
    return first(apply(model, x, ps, Internal.get_empty_state(model)))
end

"""
    display_name(layer::AbstractLuxLayer)

Printed Name of the `layer`. If the `layer` has a field `name` that is used, else the type
name is used.
"""
@generated function display_name(l::L) where {L<:AbstractLuxLayer}
    hasfield(L, :name) &&
        return :(ifelse(l.name === nothing, $(string(nameof(L))), string(l.name)))
    return :($(string(nameof(L))))
end
display_name(::T) where {T} = string(nameof(T))

# Abstract Container Layers
"""
    abstract type AbstractLuxContainerLayer{layers} <: AbstractLuxLayer

Abstract Container Type for certain Lux Layers. `layers` is a tuple containing fieldnames
for the layer, and constructs the parameters and states using those.

Users implementing their custom layer can extend the same functions as in
[`AbstractLuxLayer`](@ref).

!!! tip "Advanced Structure Manipulation"

    Advanced structure manipulation of these layers post construction is possible via
    `Functors.fmap`. For a more flexible interface, we recommend using
    `Lux.Experimental.@layer_map`.

!!! note "`fmap` Support"

    `fmap` support needs to be explicitly enabled by loading `Functors.jl` and
    `Setfield.jl`.

!!! warning "Changes from Pre-1.0 Behavior"

    Previously if `layers` was a singleton tuple, [`initialparameters`](@ref) and
    [`initialstates`](@ref) would return the parameters and states for the single field
    `layers`. From `v1.0.0` onwards, even for singleton tuples, the parameters/states
    are wrapped in a `NamedTuple` with the same name as the field. See
    [`AbstractLuxWrapperLayer`](@ref) to replicate the previous behavior of singleton
    tuples.
"""
abstract type AbstractLuxContainerLayer{layers} <: AbstractLuxLayer end

function initialparameters(
    rng::AbstractRNG, l::AbstractLuxContainerLayer{layers}
) where {layers}
    return NamedTuple{layers}(initialparameters.(rng, getfield.((l,), layers)))
end

function initialstates(
    rng::AbstractRNG, l::AbstractLuxContainerLayer{layers}
) where {layers}
    return NamedTuple{layers}(initialstates.(rng, getfield.((l,), layers)))
end

function parameterlength(l::AbstractLuxContainerLayer{layers}) where {layers}
    return sum(parameterlength, getfield.((l,), layers))
end

function statelength(l::AbstractLuxContainerLayer{layers}) where {layers}
    return sum(statelength, getfield.((l,), layers))
end

"""
    abstract type AbstractLuxWrapperLayer{layer} <: AbstractLuxLayer

See [`AbstractLuxContainerLayer`](@ref) for detailed documentation. This abstract type is
very similar to [`AbstractLuxContainerLayer`](@ref) except that it allows for a single
layer to be wrapped in a container.

Additionally, on calling [`initialparameters`](@ref) and [`initialstates`](@ref), the
parameters and states are **not** wrapped in a `NamedTuple` with the same name as the
field.

As a convenience, we define the fallback call `(::AbstractLuxWrapperLayer)(x, ps, st)`,
which calls `getfield(x, layer)(x, ps, st)`.
"""
abstract type AbstractLuxWrapperLayer{layer} <: AbstractLuxLayer end

function initialparameters(
    rng::AbstractRNG, l::AbstractLuxWrapperLayer{layer}
) where {layer}
    return initialparameters(rng, getfield(l, layer))
end

function initialstates(rng::AbstractRNG, l::AbstractLuxWrapperLayer{layer}) where {layer}
    return initialstates(rng, getfield(l, layer))
end

function parameterlength(l::AbstractLuxWrapperLayer{layer}) where {layer}
    return parameterlength(getfield(l, layer))
end

function statelength(l::AbstractLuxWrapperLayer{layer}) where {layer}
    return statelength(getfield(l, layer))
end

function (l::AbstractLuxWrapperLayer{layer})(x, ps, st) where {layer}
    return apply(getfield(l, layer), x, ps, st)
end

# Test Mode
"""
    testmode(st::NamedTuple)

Make all occurrences of `training` in state `st` -- `Val(false)`.
"""
testmode(st::NamedTuple) = update_state(st, :training, Val(false))

"""
    trainmode(st::NamedTuple)

Make all occurrences of `training` in state `st` -- `Val(true)`.
"""
trainmode(st::NamedTuple) = update_state(st, :training, Val(true))

"""
    update_state(st::NamedTuple, key::Symbol, value; exclude=Internal.isleaf)

Recursively update all occurrences of the `key` in the state `st` with the `value`.
`exclude` is a function that is passed to `Functors.fmap_with_path`'s `exclude` keyword.

!!! warning "Needs Functors.jl"

    This function requires `Functors.jl` to be loaded.
"""
function update_state(st::NamedTuple, key::Symbol, value; exclude=Internal.isleaf)
    fmap_fn = let key = key, value = value
        (kp, val) -> begin
            last(kp) == key && return value
            return val
        end
    end
    return Internal.fmap_with_path(fmap_fn, st; exclude)
end

"""
    contains_lux_layer(l) -> Bool

Check if the structure `l` is a Lux AbstractLuxLayer or a container of such a layer.
"""
function contains_lux_layer(l)
    return check_fmap_condition(Base.Fix2(isa, AbstractLuxLayer), AbstractLuxLayer, l)
end

"""
    check_fmap_condition(cond, tmatch::Union{Type, Nothing}, x) -> Bool

`fmap`s into the structure `x` and see if `cond` is satisfied for any of the leaf elements.

## Arguments

  - `cond` - A function that takes a single argument and returns a `Bool`.
  - `tmatch` - A shortcut to check if `x` is of type `tmatch`. Can be disabled by passing
    `nothing`.
  - `x` - The structure to check.

## Returns

A Boolean Value
"""
check_fmap_condition(cond::C, ::Nothing, x) where {C} = any(cond, Internal.fleaves(x))
check_fmap_condition(cond::C, ::Nothing, ::NamedTuple{()}) where {C} = any(cond, ())
function check_fmap_condition(cond::C, ::Type{T}, x) where {C,T}
    x isa T && return true
    return check_fmap_condition(cond, nothing, x)
end

"""
    preserves_state_type(l::AbstractLuxLayer)

Return `true` if the layer `l` preserves the state type of the layer. For example, if the
input state type for your layer is `T1` and the output state type is `T2`, then this
function should return `T1 == T2`.

By default this function returns `true` for `AbstractLuxLayer`. For container layers, this
function returns `true` if all the layers in the container preserve the state type.
"""
preserves_state_type(::AbstractLuxLayer) = true
function preserves_state_type(l::AbstractLuxWrapperLayer{layer}) where {layer}
    return preserves_state_type(getfield(l, layer))
end
function preserves_state_type(l::AbstractLuxContainerLayer{layers}) where {layers}
    return all(preserves_state_type, getfield.((l,), layers))
end
preserves_state_type(l::Tuple) = all(preserves_state_type, l)
preserves_state_type(l::NamedTuple) = all(preserves_state_type, values(l))

include("stateful.jl")

module Internal

    using Random: Xoshiro
    using ..LuxCore: AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer

    is_extension_loaded(::Val) = false

    function fmap_impl end            # Defined in FunctorsExt
    function fmap_with_path_impl end  # Defined in FunctorsExt
    function fleaves_impl end         # Defined in FunctorsExt
    function isleaf_impl end          # Defined in FunctorsExt

    for op in (:fmap, :fleaves, :isleaf, :fmap_with_path)
        main_op = Symbol(op, :_impl)
        err_msg = "`$op` requires `Functors.jl` to be loaded."
        @eval begin
            function $(op)(args...; kwargs...)
                is_extension_loaded(Val(:Functors)) || throw(ArgumentError($err_msg))
                return $main_op(args...; kwargs...)
            end
        end
    end

    isleaf(::AbstractLuxLayer) = true

    function setfield_impl end  # Defined in SetfieldExt

    function setfield(args...; kwargs...)
        is_extension_loaded(Val(:Setfield)) && return setfield_impl(args...; kwargs...)
        throw(ArgumentError("`setfield` requires `Setfield.jl` to be loaded."))
    end

    default_rng() = Xoshiro(1234)

    get_empty_state(::AbstractLuxLayer) = NamedTuple()
    get_empty_state(l::NamedTuple) = map(get_empty_state, l)
    function get_empty_state(l::AbstractLuxContainerLayer{layers}) where {layers}
        return NamedTuple{layers}(get_empty_state.(getfield.((l,), layers)))
    end
    function get_empty_state(l::AbstractLuxWrapperLayer{layer}) where {layer}
        return get_empty_state(getfield(l, layer))
    end

end

@public (
    replicate,
    trainmode,
    testmode,
    update_state,
    contains_lux_layer,
    check_fmap_condition,
    initialparameters,
    initialstates,
    parameterlength,
    statelength,
    outputsize,
    setup,
    apply,
    stateless_apply,
    display_name,
    preserves_state_type,
    StatefulLuxLayer,
)

export AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer

end
