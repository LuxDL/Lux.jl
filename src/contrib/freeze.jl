"""
    FrozenLayer(l::AbstractLuxLayer, which_params::Optional{Tuple})

Freeze the parameters with name `which_params` of the layer `l`.

!!! tip "Use `Lux.Experimental.freeze` instead"

    It is always recommended to use the [`Lux.Experimental.freeze`](@ref) function instead
    of directly using the `FrozenLayer` constructor.

!!! warning "No checks for `which_params`"

    There are no checks for `which_params`. For example, if the original layer has
    parameters named `(:weight, :bias)`, and `which_params` is set to `(:myweight,)`
    then none of the parameters are frozen and no error is thrown.

## Arguments

  - `l`: Lux AbstractLuxLayer.
  - `which_params`: Parameter Names to be Frozen. Can be set to `nothing`, in which case all
    parameters are frozen.

# Extended Help

## Parameters

  - Parameters of the layer `l` excluding `which_params`.

## States

  - `frozen_params`: Parameters that are frozen, i.e., `which_params`.
  - `states`: The state of the inner layer `l`.

## Note on Internal Layer Implementation

The inner layer should work with `NamedTuple` parameters. In order to support custom
parameter types, users need to implement `Lux.Utils.merge(::CustomParamType, ::NamedTuple)`
or extend `Lux.Utils.named_tuple(::CustomParamType)` to return a `NamedTuple`.

## Example

```jldoctest
julia> Lux.Experimental.FrozenLayer(Dense(2 => 2), (:weight,))
FrozenLayer(Dense(2 => 2), (:weight,))  # 2 parameters, plus 4 non-trainable
```

See also [`Lux.Experimental.freeze`](@ref), [`Lux.Experimental.unfreeze`](@ref).
"""
struct FrozenLayer{which_params,L<:AbstractLuxLayer} <: AbstractLuxLayer
    layer::L

    function FrozenLayer(l::AbstractLuxLayer, which_params::Optional{Tuple}=nothing)
        if which_params !== nothing && length(which_params) == 0
            @warn "Layer `FrozenLayer($l, (,))` is same as `l`, returning `l`."
            return l
        end
        return new{which_params,typeof(l)}(l)
    end
end

function LuxCore.initialparameters(
    rng::AbstractRNG, l::FrozenLayer{which_params}
) where {which_params}
    ps = LuxCore.initialparameters(rng, l.layer)
    ps_trainable = []
    for (k, v) in pairs(ps)
        (which_params === nothing || k in which_params) && continue
        push!(ps_trainable, k => v)
    end
    return (; ps_trainable...)
end

function LuxCore.initialstates(
    rng::AbstractRNG, l::FrozenLayer{which_params}
) where {which_params}
    ps = LuxCore.initialparameters(rng, l.layer)
    st = LuxCore.initialstates(rng, l.layer)
    ps_frozen = []
    for (k, v) in pairs(ps)
        !(which_params === nothing || k in which_params) && continue
        push!(ps_frozen, k => v)
    end
    return (frozen_params=(; ps_frozen...), states=st)
end

function (f::FrozenLayer)(x, ps, st::NamedTuple)
    y, stₙ = f.layer(x, Lux.Utils.merge(ps, st.frozen_params), st.states)
    return y, merge(st, (; states=stₙ))
end

function Base.show(io::IO, f::FrozenLayer{which_params}) where {which_params}
    which_params === nothing && return print(io, "FrozenLayer(", f.layer, ")")
    return print(io, "FrozenLayer(", f.layer, ", ", which_params, ")")
end

"""
    freeze(l::AbstractLuxLayer, which_params::Optional{Tuple} = nothing)

Constructs a version of `l` with `which_params` frozen. If `which_params` is nothing,
then all parameters are frozen.
"""
function freeze(l::AbstractLuxLayer, which_params::Optional{Tuple}=nothing)
    return FrozenLayer(l, which_params)
end

"""
    freeze(l::AbstractLuxLayer, ps, st::NamedTuple,
        which_params::Optional{Tuple} = nothing)

Construct a [`Lux.Experimental.FrozenLayer`](@ref) for `l` with the current parameters and
states. If `which_params` is nothing, then all parameters are frozen.
"""
function freeze(
    l::AbstractLuxLayer, ps, st::NamedTuple, which_params::Optional{Tuple}=nothing
)
    fl = freeze(l, which_params)
    ps_frozen = []
    ps_trainable = []
    for (k, v) in Utils.pairs(ps)
        if which_params === nothing || k in which_params
            push!(ps_frozen, k => v)
        else
            push!(ps_trainable, k => v)
        end
    end
    ps = (; ps_trainable...)
    st = (frozen_params=(; ps_frozen...), states=st)
    return fl, ps, st
end

"""
    unfreeze(l::FrozenLayer)

Unfreezes the layer `l`.
"""
unfreeze(l::FrozenLayer) = l.layer

"""
    unfreeze(l::FrozenLayer, ps, st::NamedTuple)

Unwraps a [`Lux.Experimental.FrozenLayer`](@ref) `l` with the current parameters and states.
"""
function unfreeze(fl::AbstractLuxLayer, ps, st::NamedTuple)
    return unfreeze(fl), merge(ps, st.frozen_params), st.states
end
