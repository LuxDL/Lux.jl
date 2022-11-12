"""
    FrozenLayer(l::AbstractExplicitLayer, which_params::Union{Tuple, Nothing})

Freeze the parameters with name `which_params` of the layer `l`.

!!! tip
    
    It is always recommended to use the [`Lux.freeze`](@ref) function instead of directly
    using the `FrozenLayer` constructor.

!!! warning
    
    There are no checks for `which_params`. For example, if the original layer has
    parameters named `(:weight, :bias)``, and `which_params`is set to`(:myweight,)`
    then none of the parameters are frozen and no error is thrown.

## Arguments

  - `l`: Lux AbstractExplicitLayer.
  - `which_params`: Parameter Names to be Frozen. Can be set to `nothing`, in which case all
    parameters are frozen.

## Input

  - `x`: Input to the layer `l`.

## Returns

  - Output of the inner layer `l`
  - Updated State

## Parameters

  - Parameters of the layer `l` excluding `which_params`.

## States

  - `frozen_params`: Parameters that are frozen, i.e., `which_params`.
  - `states`: The state of the inner layer `l`.

## Note on Internal Layer Implementation

The inner layer should work with `NamedTuple` parameters. In order to support custom
parameter types, users need to implement `Lux._merge(::CustomParamType, ::NamedTuple)`.

## Example

```julia
m = Lux.FrozenLayer(Dense(2 => 2), (:weight,))
```

See also [`Lux.freeze`](@ref), [`Lux.unfreeze`](@ref).
"""
struct FrozenLayer{which_params, L <: AbstractExplicitLayer} <: AbstractExplicitLayer
    layer::L

    function FrozenLayer(l::AbstractExplicitLayer,
                         which_params::Union{Tuple, Nothing}=nothing)
        if which_params !== nothing && length(which_params) == 0
            @warn "Layer `FrozenLayer($l, (,))` is same as `l`, returning `l`."
            return l
        end
        return new{which_params, typeof(l)}(l)
    end
end

function initialparameters(rng::AbstractRNG,
                           l::FrozenLayer{which_params}) where {which_params}
    ps = initialparameters(rng, l.layer)
    ps_trainable = []
    for (k, v) in pairs(ps)
        (which_params === nothing || k in which_params) && continue
        push!(ps_trainable, k => v)
    end
    return (; ps_trainable...)
end

function initialstates(rng::AbstractRNG, l::FrozenLayer{which_params}) where {which_params}
    ps = initialparameters(rng, l.layer)
    st = initialstates(rng, l.layer)
    ps_frozen = []
    for (k, v) in pairs(ps)
        !(which_params === nothing || k in which_params) && continue
        push!(ps_frozen, k => v)
    end
    return (frozen_params=(; ps_frozen...), states=st)
end

_merge(nt1::NamedTuple, nt2::NamedTuple) = merge(nt1, nt2)
_merge(nt1::ComponentArray, nt2::NamedTuple) = merge(NamedTuple(nt1), nt2)
_merge(nt1::NamedTuple, nt2::ComponentArray) = merge(nt1, NamedTuple(nt2))

function (f::FrozenLayer)(x, ps, st::NamedTuple)
    y, st_ = f.layer(x, _merge(ps, st.frozen_params), st.states)
    st = merge(st, (; states=st_))
    return y, st
end

function Base.show(io::IO, f::FrozenLayer{which_params}) where {which_params}
    if which_params === nothing
        return print(io, f.layer, " (with all parameters frozen)")
    end
    wp = join(map(x -> "`$(x)`", which_params), ", ", " & ")
    return print(io, f.layer, " (with ", wp, " frozen)")
end

"""
    freeze(l::AbstractExplicitLayer, which_params::Union{Tuple, Nothing} = nothing)

Constructs a version of `l` with `which_params` frozen. If `which_params` is nothing,
then all parameters are frozen.
"""
function freeze(l::AbstractExplicitLayer, which_params::Union{Tuple, Nothing}=nothing)
    return FrozenLayer(l, which_params)
end

"""
    freeze(l::AbstractExplicitLayer, ps, st::NamedTuple,
           which_params::Union{Tuple, Nothing} = nothing)

Construct a [`Lux.FrozenLayer`](@ref) for `l` with the current parameters and states. If
`which_params` is nothing, then all parameters are frozen.
"""
function freeze(l::AbstractExplicitLayer, ps, st::NamedTuple,
                which_params::Union{Tuple, Nothing}=nothing)
    fl = freeze(l, which_params)
    ps_frozen = []
    ps_trainable = []
    for (k, v) in pairs(ps)
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

Unwraps a [`Lux.FrozenLayer`](@ref) `l` with the current parameters and states.
"""
function unfreeze(fl::AbstractExplicitLayer, ps, st::NamedTuple)
    return unfreeze(fl), merge(ps, st.frozen_params), st.states
end
