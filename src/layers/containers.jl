
"""
    SkipConnection(layer, connection)

Create a skip connection which consists of a layer or [`Chain`](@ref) of consecutive layers
and a shortcut connection linking the block's input to the output through a user-supplied
2-argument callable. The first argument to the callable will be propagated through the given
`layer` while the second is the unchanged, "skipped" input.

The simplest "ResNet"-type connection is just `SkipConnection(layer, +)`.

## Arguments

  - `layer`: Layer or `Chain` of layers to be applied to the input

  - `connection`:

      + A 2-argument function that takes `layer(input)` and the input OR
      + An AbstractExplicitLayer that takes `(layer(input), input)` as input

## Inputs

  - `x`: Will be passed directly to `layer`

## Returns

  - Output of `connection(layer(input), input)`
  - Updated state of `layer`

## Parameters

  - Parameters of `layer` OR
  - If `connection` is an AbstractExplicitLayer, then NamedTuple with fields `:layers` and
    `:connection`

## States

  - States of `layer` OR
  - If `connection` is an AbstractExplicitLayer, then NamedTuple with fields `:layers` and
    `:connection`

See [`Parallel`](@ref) for a more general implementation.
"""
struct SkipConnection{T <: AbstractExplicitLayer, F} <:
       AbstractExplicitContainerLayer{(:layers,)}
    layers::T
    connection::F
end

function initialparameters(rng::AbstractRNG,
                           l::SkipConnection{T, <:AbstractExplicitLayer}) where {T}
    return (layers=initialparameters(rng, l.layers),
            connection=initialparameters(rng, l.connection))
end

function initialstates(rng::AbstractRNG,
                       l::SkipConnection{T, <:AbstractExplicitLayer}) where {T}
    return (layers=initialstates(rng, l.layers),
            connection=initialstates(rng, l.connection))
end

function (skip::SkipConnection)(x, ps, st::NamedTuple)
    mx, st = Lux.apply(skip.layers, x, ps, st)
    return skip.connection(mx, x), st
end

function (skip::SkipConnection{<:AbstractExplicitLayer, <:AbstractExplicitLayer})(x, ps,
                                                                                  st::NamedTuple)
    mx, st1 = Lux.apply(skip.layers, x, ps.layers, st.layers)
    y, st2 = Lux.apply(skip.connection, (mx, x), ps.connection, st.connection)
    return y, (layers=st1, connection=st2)
end

"""
    Parallel(connection, layers...)
    Parallel(connection; layers...)

Create a layer which passes an input to each path in `layers`, before reducing the output
with `connection`.

## Arguments

  - `connection`: An `N`-argument function that is called after passing the input through
    each layer. If `connection = nothing`, we return a tuple
    `Parallel(nothing, f, g)(x, y) = (f(x), g(y))`

  - Layers can be specified in two formats:

      + A list of `N` Lux layers
      + Specified as `N` keyword arguments.

## Inputs

  - `x`: If `x` is not a tuple, then return is computed as
    `connection([l(x) for l in layers]...)`. Else one is passed to each layer, thus
    `Parallel(+, f, g)(x, y) = f(x) + g(y)`.

## Returns

  - See the Inputs section for how the output is computed
  - Updated state of the `layers`

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

See also [`SkipConnection`](@ref) which is `Parallel` with one identity.
"""
struct Parallel{F, T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    connection::F
    layers::T
end

function Parallel(connection, layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return Parallel(connection, NamedTuple{names}(layers))
end

Parallel(connection; kwargs...) = Parallel(connection, (; kwargs...))

function (m::Parallel)(x, ps, st::NamedTuple)
    return applyparallel(m.layers, m.connection, x, ps, st)
end

@generated function applyparallel(layers::NamedTuple{names}, connection::C, x::T, ps,
                                  st::NamedTuple) where {names, C, T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    getinput(i) = T <: Tuple ? :(x[$i]) : :x
    calls = []
    append!(calls,
            [:(($(y_symbols[i]), $(st_symbols[i])) = Lux.apply(layers.$(names[i]),
                                                               $(getinput(i)),
                                                               ps.$(names[i]),
                                                               st.$(names[i])))
             for i in 1:N])
    push!(calls, :(st = NamedTuple{$names}((($(Tuple(st_symbols)...),)))))
    if C == Nothing
        push!(calls, :($(y_symbols[N + 1]) = tuple($(Tuple(y_symbols[1:N])...))))
    else
        push!(calls, :($(y_symbols[N + 1]) = connection($(Tuple(y_symbols[1:N])...))))
    end
    push!(calls, :(return $(y_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

Base.keys(m::Parallel) = Base.keys(getfield(m, :layers))

"""
    BranchLayer(layers...)
    BranchLayer(; layers...)

Takes an input `x` and passes it through all the `layers` and returns a tuple of the
outputs.

## Arguments

  - Layers can be specified in two formats:

      + A list of `N` Lux layers
      + Specified as `N` keyword arguments.

## Inputs

  - `x`: Will be directly passed to each of the `layers`

## Returns

  - Tuple: `(layer_1(x), layer_2(x), ..., layer_N(x))`  (naming changes if using the kwargs
    API)
  - Updated state of the `layers`

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

!!! note "Comparison with Parallel"

    This is slightly different from [`Parallel(nothing, layers...)`](@ref)

      - If the input is a tuple, `Parallel` will pass each element individually to each
        layer.

      - `BranchLayer` essentially assumes 1 input comes in and is branched out into `N`
        outputs.

## Example

An easy way to replicate an input to an NTuple is to do

```julia
l = BranchLayer(NoOpLayer(), NoOpLayer(), NoOpLayer())
```
"""
struct BranchLayer{T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
end

function BranchLayer(layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return BranchLayer(NamedTuple{names}(layers))
end

BranchLayer(; kwargs...) = BranchLayer((; kwargs...))

function (m::BranchLayer)(x, ps, st::NamedTuple)
    return applybranching(m.layers, x, ps, st)
end

@generated function applybranching(layers::NamedTuple{names}, x, ps,
                                   st::NamedTuple) where {names}
    N = length(names)
    y_symbols = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = []
    append!(calls,
            [:(($(y_symbols[i]), $(st_symbols[i])) = Lux.apply(layers.$(names[i]), x,
                                                               ps.$(names[i]),
                                                               st.$(names[i])))
             for i in 1:N])
    push!(calls, :(st = NamedTuple{$names}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return tuple($(Tuple(y_symbols)...)), st))
    return Expr(:block, calls...)
end

Base.keys(m::BranchLayer) = Base.keys(getfield(m, :layers))

"""
    PairwiseFusion(connection, layers...)
    PairwiseFusion(connection; layers...)

```
x1 → layer1 → y1 ↘
                  connection → layer2 → y2 ↘
              x2 ↗                          connection → y3
                                        x3 ↗
```

## Arguments

  - `connection`: Takes 2 inputs and combines them

  - `layers`: [`AbstractExplicitLayer`](@ref)s. Layers can be specified in two formats:

      + A list of `N` Lux layers
      + Specified as `N` keyword arguments.

## Inputs

Layer behaves differently based on input type:

 1. If the input `x` is a tuple of length `N + 1`, then the `layers` must be a tuple of
    length `N`. The computation is as follows

```julia
y = x[1]
for i in 1:N
    y = connection(x[i + 1], layers[i](y))
end
```

 2. Any other kind of input

```julia
y = x
for i in 1:N
    y = connection(x, layers[i](y))
end
```

## Returns

  - See Inputs section for how the return value is computed
  - Updated model state for all the contained layers

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)
"""
struct PairwiseFusion{F, T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    connection::F
    layers::T
end

function PairwiseFusion(connection, layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return PairwiseFusion(connection, NamedTuple{names}(layers))
end

PairwiseFusion(connection; kwargs...) = PairwiseFusion(connection, (; kwargs...))

function (m::PairwiseFusion)(x, ps, st::NamedTuple)
    return applypairwisefusion(m.layers, m.connection, x, ps, st)
end

@generated function applypairwisefusion(layers::NamedTuple{names}, connection::C, x::T, ps,
                                        st::NamedTuple) where {names, C, T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    getinput(i) = T <: Tuple ? :(x[$i]) : :x
    calls = [:($(y_symbols[N + 1]) = $(getinput(1)))]
    append!(calls,
            [:(($(y_symbols[i]), $(st_symbols[i])) = Lux.apply(layers.$(names[i]),
                                                               $(y_symbols[N + 1]),
                                                               ps.$(names[i]),
                                                               st.$(names[i]));
               $(y_symbols[N + 1]) = connection($(y_symbols[i]), $(getinput(i + 1))))
             for i in 1:N])
    push!(calls, :(st = NamedTuple{$names}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(y_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

Base.keys(m::PairwiseFusion) = Base.keys(getfield(m, :layers))

"""
    Chain(layers...; disable_optimizations::Bool = false)
    Chain(; layers..., disable_optimizations::Bool = false)

Collects multiple layers / functions to be called in sequence on a given input.

## Arguments

  - Layers can be specified in two formats:

      + A list of `N` Lux layers
      + Specified as `N` keyword arguments.

## Keyword Arguments

  - `disable_optimizations`: Prevents any structural optimization

## Inputs

Input `x` is passed sequentially to each layer, and must conform to the input requirements
of the internal layers.

## Returns

  - Output after sequentially applying all the layers to `x`
  - Updated model states

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

## Optimizations

Performs a few optimizations to generate reasonable architectures. Can be disabled using
keyword argument `disable_optimizations`.

  - All sublayers are recursively optimized.
  - If a function `f` is passed as a layer and it doesn't take 3 inputs, it is converted to
    a [`WrappedFunction`](@ref)(`f`) which takes only one input.
  - If the layer is a Chain, it is flattened.
  - [`NoOpLayer`](@ref)s are removed.
  - If there is only 1 layer (left after optimizations), then it is returned without the
    `Chain` wrapper.
  - If there are no layers (left after optimizations), a [`NoOpLayer`](@ref) is returned.

## Miscellaneous Properties

  - Allows indexing. We can access the `i`th layer using `m[i]`. We can also index using
    ranges or arrays.

## Example

```julia
c = Chain(Dense(2, 3, relu), BatchNorm(3), Dense(3, 2))
```
"""
struct Chain{T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T

    function Chain(xs...; disable_optimizations::Bool=false)
        xs = disable_optimizations ? xs : _flatten_model(xs)
        length(xs) == 0 && return NoOpLayer()
        length(xs) == 1 && return first(xs)
        names = ntuple(i -> Symbol("layer_$i"), length(xs))
        layers = NamedTuple{names}(xs)
        return new{typeof(layers)}(layers)
    end

    function Chain(xs::AbstractVector; disable_optimizations::Bool=false)
        return Chain(xs...; disable_optimizations)
    end

    function Chain(nt::NamedTuple; disable_optimizations::Bool=true)
        if !disable_optimizations
            throw(ArgumentError("Chain(::NamedTuple) is not compatible with" *
                                " disable_optimizations=true"))
        end
        return new{typeof(nt)}(nt)
    end

    function Chain(; disable_optimizations::Bool=true, kwargs...)
        return Chain((; kwargs...); disable_optimizations)
    end
end

function _flatten_model(layers::Union{AbstractVector, Tuple})
    new_layers = []
    for l in layers
        f = _flatten_model(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
        elseif f isa Function
            if !hasmethod(f, (Any, Any, NamedTuple))
                if f === identity
                    continue
                else
                    push!(new_layers, WrappedFunction(f))
                end
            else
                push!(new_layers, f)
            end
        elseif f isa Chain
            append!(new_layers, f.layers)
        elseif f isa NoOpLayer
            continue
        else
            push!(new_layers, f)
        end
    end
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

_flatten_model(x) = x

function (c::Chain)(x, ps, st::NamedTuple)
    return applychain(c.layers, x, ps, st)
end

@generated function applychain(layers::NamedTuple{fields}, x, ps,
                               st::NamedTuple{fields}) where {fields}
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    st_symbols = [gensym() for _ in 1:N]
    calls = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
                                                                   $(x_symbols[i]),
                                                                   ps.$(fields[i]),
                                                                   st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(x_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

Base.keys(m::Chain) = Base.keys(getfield(m, :layers))

Base.getindex(c::Chain, i::Int) = c.layers[i]
Base.getindex(c::Chain, i::AbstractArray) = Chain(_index_namedtuple(c.layers, i))

Base.length(c::Chain) = length(c.layers)
Base.lastindex(c::Chain) = lastindex(c.layers)
Base.firstindex(c::Chain) = firstindex(c.layers)

"""
    Maxout(layers...)
    Maxout(; layers...)
    Maxout(f::Function, n_alts::Int)

This contains a number of internal layers, each of which receives the same input. Its output
is the elementwise maximum of the the internal layers' outputs.

Maxout over linear dense layers satisfies the univeral approximation theorem. See [1].

See also [`Parallel`](@ref) to reduce with other operators.

## Arguments

  - Layers can be specified in three formats:

      + A list of `N` Lux layers
      + Specified as `N` keyword arguments.
      + A no argument function `f` and an integer `n_alts` which specifies the number of
        layers.

## Inputs

  - `x`: Input that is passed to each of the layers

## Returns

  - Output is computed by taking elementwise `max` of the outputs of the individual layers.
  - Updated state of the `layers`

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

## References

[1] Goodfellow, Warde-Farley, Mirza, Courville & Bengio "Maxout Networks"
[https://arxiv.org/abs/1302.4389](https://arxiv.org/abs/1302.4389)
"""
struct Maxout{T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
end

function Maxout(layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return Maxout(NamedTuple{names}(layers))
end

Maxout(; kwargs...) = Maxout((; kwargs...))

Maxout(f::Function, n_alts::Int) = Maxout(ntuple(_ -> f(), n_alts)...)

# NOTE(@avik-pal): Calling `applyparallel` with broadcasted `max` is slower than this
#                  implementation.
(m::Maxout)(x, ps, st::NamedTuple) = applymaxout(m.layers, x, ps, st)

@generated function applymaxout(layers::NamedTuple{fields}, x, ps,
                                st::NamedTuple{fields}) where {fields}
    N = length(fields)
    y_symbols = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = [:(($(y_symbols[i]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]), x,
                                                               ps.$(fields[i]),
                                                               st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(res = max.($(Tuple(y_symbols)...))))
    push!(calls, :(return res, st))
    return Expr(:block, calls...)
end

Base.keys(m::Maxout) = Base.keys(getfield(m, :layers))
