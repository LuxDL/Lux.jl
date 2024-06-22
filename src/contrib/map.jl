@doc doc"""
    @layer_map func layer ps st

See the documentation of [`Lux.Experimental.layer_map`](@ref) for more details. This macro
eliminates the need to the set the layer name, and uses the variable name as the starting
point.

## Example

```jldoctest
julia> using Lux, Random

julia> c = Parallel(
           +; chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3), dense_2=Dense(3 => 5)),
           dense_3=Dense(5 => 1))
Parallel(
    +
    chain = Chain(
        dense_1 = Dense(2 => 3),        # 9 parameters
        bn = BatchNorm(3, affine=true, track_stats=true),  # 6 parameters, plus 7
        dense_2 = Dense(3 => 5),        # 20 parameters
    ),
    dense_3 = Dense(5 => 1),            # 6 parameters
)         # Total: 41 parameters,
          #        plus 7 states.

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

julia> _, ps_new, _ = Lux.Experimental.@layer_map zero_dense_params c ps st;
zeroing params of c.layers.chain.layers.dense_1
zeroing params of c.layers.chain.layers.dense_2
zeroing params of c.layers.dense_3

julia> all(iszero, (ps_new.chain.dense_1.weight, ps_new.chain.dense_1.bias,
                    ps_new.chain.dense_2.weight, ps_new.chain.dense_2.bias,
                    ps_new.dense_3.weight, ps_new.dense_3.bias))
true
```
"""
macro layer_map(f, l, ps, st)
    quote
        layer_map($(esc(f)), $(esc(l)), $(esc(ps)), $(esc(st)), $(string(l)))
    end
end

@doc doc"""
    layer_map(f::Function, l::AbstractExplicitLayer, ps, st::NamedTuple,
              name::String="model")

Map the function `f` over the model `l`, with the parameters `ps` and states `st`. This is
different from `Functors.fmap` since it zips the layers, parameters, and states and invokes
the function on all of them together.

## Call Signature for `f`

  - Must take 4 inputs -- `AbstractExplicitLayer`, Corresponding Parameters, Corresponding
    States, and the name of the layer.
  - Must return a tuple of 3 elements -- `AbstractExplicitLayer`, new parameters and the new
    states.

!!! note

    Starting `v0.6`, instead of the name of the layer, we will provide the [KeyPath to the
    layer](https://fluxml.ai/Functors.jl/stable/api/#KeyPath). The current version of
    providing a String has been deprecated.

!!! tip

    We recommend using the macro `Lux.@layer_map` instead of this function. It automatically
    sets the `name` of the layer to be the variable name.

## Example

```jldoctest
julia> using Lux, Random


julia> c = Parallel(
           +; chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3), dense_2=Dense(3 => 5)),
           dense_3=Dense(5 => 1))
Parallel(
    +
    chain = Chain(
        dense_1 = Dense(2 => 3),        # 9 parameters
        bn = BatchNorm(3, affine=true, track_stats=true),  # 6 parameters, plus 7
        dense_2 = Dense(3 => 5),        # 20 parameters
    ),
    dense_3 = Dense(5 => 1),            # 6 parameters
)         # Total: 41 parameters,
          #        plus 7 states.

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
zeroing params of model.layers.chain.layers.dense_1
zeroing params of model.layers.chain.layers.dense_2
zeroing params of model.layers.dense_3

julia> all(iszero, (ps_new.chain.dense_1.weight, ps_new.chain.dense_1.bias,
                    ps_new.chain.dense_2.weight, ps_new.chain.dense_2.bias,
                    ps_new.dense_3.weight, ps_new.dense_3.bias))
true
```
"""
function layer_map(f::F, l, ps, st, name::String="model") where {F <: Function}
    # TODO: In v0.6 deprecate passing the string
    f_wrapper = @closure (kp, layer, ps_, st_) -> f(
        layer, ps_, st_, __keypath_to_string(name, kp))
    return fmap_with_path(f_wrapper, l, ps, st; walk=LayerWalkWithPath())
end

@inline __keypath_to_string(kp::KeyPath) = join(kp.keys, ".")
@inline __keypath_to_string(str::String, kp::KeyPath) = "$(str).$(__keypath_to_string(kp))"

struct LayerWalkWithPath <: Functors.AbstractWalk end

function (::LayerWalkWithPath)(recurse, kp::KeyPath, layer, ps, st)
    _layer_children, layer_re = functor(layer)
    ps_children, ps_re = functor(ps)
    st_children, st_re = functor(st)

    _children = keys(ps_children)
    needs_correction = _children != keys(_layer_children)
    _key = needs_correction ? only(keys(_layer_children)) : nothing
    layer_children = needs_correction ? getfield(layer, _key) : _layer_children
    @assert keys(layer_children) == keys(ps_children) == keys(st_children)

    kps = NamedTuple{_children}(map(
        x -> needs_correction ? KeyPath(kp, _key, x) : KeyPath(kp, x), _children))

    ys = map(recurse, kps, layer_children, ps_children, st_children)
    layer_children_new = map(Base.Fix2(getindex, 1), ys)
    ps_children_new = map(Base.Fix2(getindex, 2), ys)
    st_children_new = map(Base.Fix2(getindex, 3), ys)

    layer_new = needs_correction ? layer_re(NamedTuple{(_key,)}((layer_children_new,))) :
                layer_re(layer_children_new)
    ps_new = ps_re(ps_children_new)
    st_new = st_re(st_children_new)

    return layer_new, ps_new, st_new
end
