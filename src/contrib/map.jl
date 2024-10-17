@doc doc"""
    layer_map(f, l::AbstractLuxLayer, ps, st::NamedTuple)

Map the function `f` over the model `l`, with the parameters `ps` and states `st`. This is
different from `Functors.fmap` since it zips the layers, parameters, and states and invokes
the function on all of them together.

!!! tip "KeyPath provided to the function"

    The `KeyPath` depths on the structure of the parameters and states. This is of
    consequence exclusively for [`AbstractLuxWrapperLayer`](@ref) where the structure of the
    layer doesn't match the structure of the parameters and states. In the example, provided
    below, the `KeyPath` is `(:chain, :dense_1)` for the first layer (following the
    structure in `ps`) while accessing the same layer in the chain is done with `( 
    :chain, :layers, :dense_1)`.

## Call Signature for `f`

  - Must take 4 inputs -- `AbstractLuxLayer`, Corresponding Parameters, Corresponding
    States, and the `Functors.KeyPath` to the layer.
  - Must return a tuple of 3 elements -- `AbstractLuxLayer`, new parameters and the new
    states.

# Extended Help

## Example

```jldoctest
julia> using Lux, Random

julia> c = Parallel(
           +; chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3), dense_2=Dense(3 => 5)),
           dense_3=Dense(5 => 1));

julia> rng = Random.default_rng();

julia> ps, st = Lux.setup(rng, c);

julia> # Makes parameters of Dense Layers inside Chain zero
       function zero_dense_params(l, ps, st, name)
           if l isa Dense
               println("zeroing params of $name")
               ps = merge(ps, (; weight=zero.(ps.weight), bias=zero.(ps.bias)))
           end
           return l, ps, st
       end;

julia> _, ps_new, _ = Lux.Experimental.layer_map(zero_dense_params, c, ps, st);
zeroing params of KeyPath(:chain, :dense_1)
zeroing params of KeyPath(:chain, :dense_2)
zeroing params of KeyPath(:dense_3,)

julia> all(iszero, (ps_new.chain.dense_1.weight, ps_new.chain.dense_1.bias,
                    ps_new.chain.dense_2.weight, ps_new.chain.dense_2.bias,
                    ps_new.dense_3.weight, ps_new.dense_3.bias))
true
```
"""
function layer_map(f, l, ps, st)
    return fmap_with_path(l, ps, st; walk=LayerWalkWithPath()) do kp, layer, ps_, st_
        return f(layer, ps_, st_, kp)
    end
end

struct LayerWalkWithPath <: Functors.AbstractWalk end

function (::LayerWalkWithPath)(
        recurse::R, kp::KeyPath, layer::AbstractLuxWrapperLayer{field},
        ps, st) where {R, field}
    layer_children, layer_re = functor(getfield(layer, field))
    ps_children, ps_re = functor(ps)
    st_children, st_re = functor(st)

    layer_children_new, ps_children_new,
    st_children_new = perform_layer_map(
        recurse, kp, ps_children, st_children, layer_children)

    inner_layer = layer_re(layer_children_new)
    return (Setfield.set(layer, Setfield.PropertyLens{field}(), inner_layer),
        ps_re(ps_children_new), st_re(st_children_new))
end

function (::LayerWalkWithPath)(
        recurse::R, kp::KeyPath, layer::AbstractLuxLayer, ps, st) where {R}
    layer_children, layer_re = functor(layer)
    ps_children, ps_re = functor(ps)
    st_children, st_re = functor(st)

    layer_children_new, ps_children_new,
    st_children_new = perform_layer_map(
        recurse, kp, ps_children, st_children, layer_children)

    return layer_re(layer_children_new), ps_re(ps_children_new), st_re(st_children_new)
end

function perform_layer_map(recurse, kp, ps_children, st_children, layer_children)
    @argcheck keys(layer_children) == keys(ps_children) == keys(st_children)

    kps = NamedTuple{keys(ps_children)}(map(Base.Fix1(KeyPath, kp), keys(ps_children)))

    ys = map(recurse, kps, layer_children, ps_children, st_children)
    layer_children_new = map(Base.Fix2(getindex, 1), ys)
    ps_children_new = map(Base.Fix2(getindex, 2), ys)
    st_children_new = map(Base.Fix2(getindex, 3), ys)

    return layer_children_new, ps_children_new, st_children_new
end
