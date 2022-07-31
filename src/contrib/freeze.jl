"""
    FrozenLayer(l::AbstractExplicitLayer, which_params::Tuple)

Freeze the parameters with name `which_params` of the layer `l`.

## Arguments

  - `l`: Lux AbstractExplicitLayer.
  - `which_params`: Parameter Names to be Frozen.

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

## Example

```julia
m = Lux.FrozenLayer(Lux.Dense(2 => 2), (:weight,))
```

See also [`Lux.freeze`](@ref).
"""
struct FrozenLayer{which_params, L <: AbstractExplicitLayer} <: AbstractExplicitLayer
    layer::L

    function FrozenLayer(l::AbstractExplicitLayer, which_params::Tuple)
        return new{which_params, typeof(l)}(l)
    end
end

function initialparameters(rng::AbstractRNG,
                           l::FrozenLayer{which_params}) where {which_params}
    ps = initialparameters(rng, l.layer)
    ps_trainable = []
    for (k, v) in pairs(ps)
        k in which_params && continue
        push!(ps_trainable, k => v)
    end
    return (; ps_trainable...)
end

function initialstates(rng::AbstractRNG, l::FrozenLayer{which_params}) where {which_params}
    ps = initialparameters(rng, l.layer)
    st = initialstates(rng, l.layer)
    ps_frozen = []
    for (k, v) in pairs(ps)
        !(k in which_params) && continue
        push!(ps_frozen, k => v)
    end
    return (frozen_params=(; ps_frozen...), states=st)
end

function (f::FrozenLayer)(x, ps, st::NamedTuple)
    y, st_ = f.layer(x, merge(ps, st.frozen_params), st.states)
    st = merge(st, (; states=st_))
    return y, st
end

function Base.show(io::IO, f::FrozenLayer{which_params}) where {which_params}
    wp = join(map(x -> "`$(x)`", which_params), ", ", " & ")
    return print(io, f.layer, " (with ", wp, " frozen)")
end

"""
    freeze(l::AbstractExplicitLayer, which_params::Tuple)

Constructs a version of `l` with `which_params` frozen.
"""
freeze(l::AbstractExplicitLayer, which_params::Tuple) = FrozenLayer(l, which_params)

"""
    freeze(l::AbstractExplicitLayer, ps, st::NamedTuple, which_params::Tuple)

Construct a [`Lux.FrozenLayer`](@ref) for `l` with the current parameters and states.
"""
function freeze(l::AbstractExplicitLayer, ps, st::NamedTuple, which_params::Tuple)
    fl = FrozenLayer(l, which_params)
    ps_frozen = []
    ps_trainable = []
    for (k, v) in pairs(ps)
        if k in which_params
            push!(ps_frozen, k => v)
        else
            push!(ps_trainable, k => v)
        end
    end
    ps = (; ps_trainable...)
    st = (frozen_params=(; ps_frozen...), states=st)
    return fl, ps, st
end
