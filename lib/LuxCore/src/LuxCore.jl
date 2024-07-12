module LuxCore

using Compat: @compat
using DispatchDoctor: @stable
using Functors: Functors, fmap, fleaves
using Random: Random, AbstractRNG, Xoshiro
using Setfield: Setfield

# PRNG Handling
"""
    replicate(rng::AbstractRNG)

Creates a copy of the `rng` state depending on its type.
"""
replicate(rng::AbstractRNG) = deepcopy(rng)
function replicate(rng::Random.TaskLocalRNG)
    @warn "`replicate` doesn't work for `TaskLocalRNG`. Returning the same `TaskLocalRNG`." maxlog=1
    return deepcopy(rng)
end

@inline _default_rng() = Xoshiro(1234)

"""
    abstract type AbstractExplicitLayer

Abstract Type for all Lux Layers

Users implementing their custom layer, **must** implement

  - `initialparameters(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This
    returns a `NamedTuple` containing the trainable parameters for the layer.
  - `initialstates(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a
    NamedTuple containing the current state for the layer. For most layers this is typically
    empty. Layers that would potentially contain this include `BatchNorm`, `LSTM`, `GRU`,
    etc.

Optionally:

  - `parameterlength(layer::CustomAbstractExplicitLayer)` -- These can be automatically
    calculated, but it is recommended that the user defines these.
  - `statelength(layer::CustomAbstractExplicitLayer)` -- These can be automatically
    calculated, but it is recommended that the user defines these.

See also [`AbstractExplicitContainerLayer`](@ref)
"""
abstract type AbstractExplicitLayer end

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
        $(op)(::AbstractRNG, ::Union{AbstractExplicitLayer, Nothing}) = NamedTuple()
        function $(op)(rng::AbstractRNG, l::Union{NamedTuple, Tuple, AbstractArray})
            return map(Base.Fix1($op, rng), l)
        end
        function $(op)(rng::AbstractRNG, l)
            contains_lux_layer(l) && return fmap(Base.Fix1($op, rng), l)
            throw(MethodError($op, (rng, l)))
        end
    end
end

@inline _getemptystate(::AbstractExplicitLayer) = NamedTuple()
@inline _getemptystate(l::NamedTuple) = map(_getemptystate, l)

"""
    parameterlength(layer)

Return the total number of parameters of the layer `l`.
"""
function parameterlength(l::AbstractExplicitLayer)
    return parameterlength(initialparameters(_default_rng(), l))
end
function parameterlength(nt::Union{NamedTuple, Tuple})
    return length(nt) == 0 ? 0 : sum(parameterlength, nt)
end
parameterlength(a::AbstractArray) = length(a)

"""
    statelength(layer)

Return the total number of states of the layer `l`.
"""
statelength(l::AbstractExplicitLayer) = statelength(initialstates(_default_rng(), l))
statelength(nt::Union{NamedTuple, Tuple}) = length(nt) == 0 ? 0 : sum(statelength, nt)
statelength(a::AbstractArray) = length(a)
statelength(::Any) = 1

"""
    inputsize(layer)

Return the input size of the layer.
"""
function inputsize end

@inline __size(x::AbstractVector{T}) where {T} = isbitstype(T) ? size(x) : __size.(x)
@inline function __size(x::AbstractArray{T, N}) where {T, N}
    return isbitstype(T) ? size(x)[1:(N - 1)] : __size.(x)
end
@inline __size(x::Tuple) = __size.(x)
@inline __size(x::NamedTuple{fields}) where {fields} = NamedTuple{fields}(__size.(values(x)))
@inline __size(x) = fmap(__size, x)

"""
    outputsize(layer, x, rng)

Return the output size of the layer. If `outputsize(layer)` is defined, that method
takes precedence, else we compute the layer output to determine the final size.

The fallback implementation of this function assumes the inputs were batched, i.e.,
if any of the outputs are Arrays, with `ndims(A) > 1`, it will return
`size(A)[1:(end - 1)]`. If this behavior is undesirable, provide a custom
`outputsize(layer, x, rng)` implementation).
"""
function outputsize(layer, x, rng)
    hasmethod(outputsize, Tuple{typeof(layer)}) && return outputsize(layer)
    ps, st = setup(rng, layer)
    y = first(apply(layer, x, ps, st))
    return __size(y)
end

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
@stable default_mode="disable" function apply(model::AbstractExplicitLayer, x, ps, st)
    return model(x, ps, st)
end

"""
    stateless_apply(model, x, ps)

Calls `apply` and only returns the first argument. This function requires that `model` has
an empty state of `NamedTuple()`. Behavior of other kinds of models are undefined and it is
the responsibility of the user to ensure that the model has an empty state.
"""
function stateless_apply(model::AbstractExplicitLayer, x, ps)
    return first(apply(model, x, ps, _getemptystate(model)))
end

"""
    display_name(layer::AbstractExplicitLayer)

Printed Name of the `layer`. If the `layer` has a field `name` that is used, else the type
name is used.
"""
@generated function display_name(l::L) where {L <: AbstractExplicitLayer}
    hasfield(L, :name) &&
        return :(ifelse(l.name === nothing, $(string(nameof(L))), string(l.name)))
    return :($(string(nameof(L))))
end
display_name(::T) where {T} = string(nameof(T))

# Abstract Container Layers
"""
    abstract type AbstractExplicitContainerLayer{layers} <: AbstractExplicitLayer

Abstract Container Type for certain Lux Layers. `layers` is a tuple containing fieldnames
for the layer, and constructs the parameters and states using those.

Users implementing their custom layer can extend the same functions as in
[`AbstractExplicitLayer`](@ref).

!!! tip

    Advanced structure manipulation of these layers post construction is possible via
    `Functors.fmap`. For a more flexible interface, we recommend using
    `Lux.Experimental.@layer_map`.
"""
abstract type AbstractExplicitContainerLayer{layers} <: AbstractExplicitLayer end

function initialparameters(rng::AbstractRNG,
        l::AbstractExplicitContainerLayer{layers}) where {layers}
    length(layers) == 1 && return initialparameters(rng, getfield(l, layers[1]))
    return NamedTuple{layers}(initialparameters.(rng, getfield.((l,), layers)))
end

function initialstates(rng::AbstractRNG,
        l::AbstractExplicitContainerLayer{layers}) where {layers}
    length(layers) == 1 && return initialstates(rng, getfield(l, layers[1]))
    return NamedTuple{layers}(initialstates.(rng, getfield.((l,), layers)))
end

function parameterlength(l::AbstractExplicitContainerLayer{layers}) where {layers}
    return sum(parameterlength, getfield.((l,), layers))
end

function statelength(l::AbstractExplicitContainerLayer{layers}) where {layers}
    return sum(statelength, getfield.((l,), layers))
end

function _getemptystate(l::AbstractExplicitContainerLayer{layers}) where {layers}
    length(layers) == 1 && return _getemptystate(getfield(l, first(layers)))
    return NamedTuple{layers}(_getemptystate.(getfield.((l,), layers)))
end

# Make AbstractExplicit Layers Functor Compatible
function Functors.functor(::Type{<:AbstractExplicitContainerLayer{layers}},
        x) where {layers}
    _children = NamedTuple{layers}(getproperty.((x,), layers))
    recon_fn = (l, (c, n)) -> Setfield.set(l, Setfield.PropertyLens{n}(), c)
    layer_reconstructor = let x = x, recon_fn = recon_fn, layers = layers
        z -> reduce(recon_fn, zip(z, layers); init=x)
    end
    return _children, layer_reconstructor
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
    update_state(st::NamedTuple, key::Symbol, value;
        layer_check=_default_layer_check(key))

Recursively update all occurrences of the `key` in the state `st` with the `value`.
"""
function update_state(st::NamedTuple, key::Symbol, value;
        layer_check::LC=_default_layer_check(key)) where {LC}
    fmap_fn = let key = key, value = value
        _st -> Setfield.set(_st, Setfield.PropertyLens{key}(), value)
    end
    return fmap(fmap_fn, st; exclude=layer_check)
end

function _default_layer_check(key)
    return let key = key
        x -> hasmethod(keys, (typeof(x),)) ? (key ∈ keys(x)) : false
    end
end

"""
    contains_lux_layer(l) -> Bool

Check if the structure `l` is a Lux AbstractExplicitLayer or a container of such a layer.
"""
function contains_lux_layer(l)
    return check_fmap_condition(Base.Fix2(isa, AbstractExplicitLayer),
        AbstractExplicitLayer, l)
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
check_fmap_condition(cond::C, ::Nothing, x) where {C} = any(cond, fleaves(x))
function check_fmap_condition(cond::C, ::Type{T}, x) where {C, T}
    x isa T && return true
    return check_fmap_condition(cond, nothing, x)
end

@compat(public,
    (replicate, trainmode, testmode, update_state, contains_lux_layer,
        check_fmap_condition, AbstractExplicitLayer, AbstractExplicitContainerLayer,
        initialparameters, initialstates, parameterlength, statelength,
        inputsize, outputsize, setup, apply, stateless_apply, display_name))

end
