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
    Chain(
        dense_1 = Dense(2 => 3),        # 9 parameters
        bn = BatchNorm(3, affine=true, track_stats=true),  # 6 parameters, plus 7
        dense_2 = Dense(3 => 5),        # 20 parameters
    ),
    Dense(5 => 1),                      # 6 parameters
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
    Chain(
        dense_1 = Dense(2 => 3),        # 9 parameters
        bn = BatchNorm(3, affine=true, track_stats=true),  # 6 parameters, plus 7
        dense_2 = Dense(3 => 5),        # 20 parameters
    ),
    Dense(5 => 1),                      # 6 parameters
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
function layer_map(f::Function, l, ps, st, name::String="model")
    l_c, l_re = __consistent_functor(l)
    ps_c, ps_re = __consistent_functor(ps)
    st_c, st_re = __consistent_functor(st)

    length(l_c) == 0 && return f(l, ps, st, name)

    @assert fieldnames(typeof(st_c)) == fieldnames(typeof(ps_c))

    if length(l_c) == 1 && fieldnames(typeof(l_c)) != fieldnames(typeof(ps_c))
        __correct_luxcore_inconsistency = true
        ps_c = NamedTuple{fieldnames(typeof(l_c))}((ps_c,))
        st_c = NamedTuple{fieldnames(typeof(l_c))}((st_c,))
    else
        __correct_luxcore_inconsistency = false
    end

    l_c_new, ps_c_new, st_c_new = [], [], []
    for k in keys(l_c)
        l_c_new_, ps_c_new_, st_c_new_ = layer_map(
            f, getproperty(l_c, k), getproperty(ps_c, k),
            getproperty(st_c, k), join((name, k), "."))
        push!(l_c_new, k => l_c_new_)
        push!(ps_c_new, k => ps_c_new_)
        push!(st_c_new, k => st_c_new_)
    end

    l_new = l_re((; l_c_new...))
    ps_new, st_new = __correct_luxcore_inconsistency ?
                     (
        ps_re((; last(first(ps_c_new))...)), st_re((; last(first(st_c_new))...))) :
                     (ps_re((; ps_c_new...)), st_re((; st_c_new...)))

    return l_new, ps_new, st_new
end

function __consistent_functor(x)
    c, re = functor(x)
    c isa NamedTuple && return (c, re)
    c_fixed = NamedTuple{Tuple(collect(Symbol.(1:(length(c)))))}(c)
    function re_updated(y::NamedTuple)
        c isa AbstractVector && return re([values(y)...])
        return re(values(y))
    end
    return (c_fixed, re_updated)
end

function __fix_tuple_functor(x::Tuple, ::NamedTuple{names}) where {names}
    length(x) == 1 && length(names) > 1 && first(x) isa NamedTuple && return first(x)
    @assert length(x)==length(names) lazy"length(x) ($(length(x))) != length(names) ($(length(names))). This should never happen, please open an issue."
    return NamedTuple{names}(x)
end
__fix_tuple_functor(x, _) = x
