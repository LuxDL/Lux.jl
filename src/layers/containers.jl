"""
    SkipConnection(layers, connection; name=nothing)
    SkipConnection(; layers, connection, name=nothing)

Create a skip connection which consists of a layer or [`Chain`](@ref) of consecutive layers
and a shortcut connection linking the block's input to the output through a user-supplied
2-argument callable. The first argument to the callable will be propagated through the given
`layer` while the second is the unchanged, "skipped" input.

The simplest "ResNet"-type connection is just `SkipConnection(layer, +)`.

## Arguments

  - `layer`: Layer or `Chain` of layers to be applied to the input

  - `connection`:

      + A 2-argument function that takes `layer(input)` and the input OR
      + An AbstractLuxLayer that takes `(layer(input), input)` as input

# Extended Help

## Inputs

  - `x`: Will be passed directly to `layer`

## Returns

  - Output of `connection(layer(input), input)`
  - Updated state of `layer`

## Parameters

  - Parameters of `layer` OR
  - If `connection` is an AbstractLuxLayer, then NamedTuple with fields `:layers` and
    `:connection`

## States

  - States of `layer` OR
  - If `connection` is an AbstractLuxLayer, then NamedTuple with fields `:layers` and
    `:connection`

See [`Parallel`](@ref) for a more general implementation.
"""
@concrete struct SkipConnection <: AbstractLuxWrapperLayer{:layers}
    layers
    connection
    name
end

PrettyPrinting.printable_children(l::SkipConnection) = (; l.connection, l.layers)

function SkipConnection(layers, connection; name::NAME_TYPE=nothing)
    return SkipConnection(; layers, connection, name)
end

function SkipConnection(; layers, connection, name::NAME_TYPE=nothing)
    return SkipConnection(layers, connection, name)
end

function initialparameters(
    rng::AbstractRNG, l::SkipConnection{T,<:AbstractLuxLayer}
) where {T}
    return (
        layers=initialparameters(rng, l.layers),
        connection=initialparameters(rng, l.connection),
    )
end

function initialstates(rng::AbstractRNG, l::SkipConnection{T,<:AbstractLuxLayer}) where {T}
    return (
        layers=initialstates(rng, l.layers), connection=initialstates(rng, l.connection)
    )
end

function (skip::SkipConnection)(x, ps, st::NamedTuple)
    mx, st = @inline apply(skip.layers, x, ps, st)
    return skip.connection(mx, x), st
end

function (skip::SkipConnection{<:AbstractLuxLayer,<:AbstractLuxLayer})(
    x, ps, st::NamedTuple
)
    mx, st1 = @inline apply(skip.layers, x, ps.layers, st.layers)
    y, st2 = @inline apply(skip.connection, (mx, x), ps.connection, st.connection)
    return y, (layers=st1, connection=st2)
end

"""
    Parallel(connection, layers...; name=nothing)
    Parallel(connection; name=nothing, layers...)
    Parallel(; connection, layers..., name=nothing)

Create a layer which passes an input to each path in `layers`, before reducing the output
with `connection`.

## Arguments

  - `connection`: An `N`-argument function that is called after passing the input through
    each layer, OR an AbstractLuxLayer that takes a tuple of `N` inputs. 
    If `connection = nothing`, we return a tuple: `Parallel(nothing, f, g)(x, y) = (f(x), g(y))`

  - Layers can be specified in two formats:

      + A list of `N` Lux layers
      + Specified as `N` keyword arguments.

# Extended Help

## Inputs

  - `x`: If `x` is not a tuple, then return is computed as
    `connection([l(x) for l in layers]...)`. Else one is passed to each layer, thus
    `Parallel(+, f, g)(x, y) = f(x) + g(y)`.

## Returns

  - See the Inputs section for how the output is computed
  - Updated state of the `layers` (and `connection` if it's a layer)

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)
  - If `connection` is an AbstractLuxLayer, parameters include both `layers` and `connection`

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)
  - If `connection` is an AbstractLuxLayer, states include both `layers` and `connection`

See also [`SkipConnection`](@ref) which is `Parallel` with one identity.

## Example

```jldoctest
julia> model = Parallel(nothing, Dense(2, 1), Dense(2, 1))
Parallel(
    layer_(1-2) = Dense(2 => 1),                  # 6 (3 x 2) parameters
)         # Total: 6 parameters,
          #        plus 0 states.

julia> using Random;
       rng = Random.seed!(123);
       ps, st = Lux.setup(rng, model);
       x1 = randn(rng, Float32, 2);
       x2 = randn(rng, Float32, 2);

julia> size.(first(model((x1, x2), ps, st)))
((1,), (1,))
```
"""
@concrete struct Parallel <: AbstractLuxWrapperLayer{:layers}
    connection
    layers <: NamedTuple
    name
end

function PrettyPrinting.printable_children(l::Parallel)
    children = Functors.children(l)
    l.connection === nothing && return children.layers
    return merge((; l.connection), children.layers)
end

function Parallel(connection, layers...; name::NAME_TYPE=nothing)
    return Parallel(connection, Utils.named_tuple_layers(layers...), name)
end

function Parallel(connection; name::NAME_TYPE=nothing, kwargs...)
    return Parallel(; connection, name, kwargs...)
end

function Parallel(; name::NAME_TYPE=nothing, connection, kwargs...)
    return Parallel(connection, (; kwargs...), name)
end

function initialparameters(rng::AbstractRNG, l::Parallel{<:AbstractLuxLayer,<:NamedTuple})
    return (;
        layers=initialparameters(rng, l.layers),
        connection=initialparameters(rng, l.connection),
    )
end

function initialstates(rng::AbstractRNG, l::Parallel{<:AbstractLuxLayer,<:NamedTuple})
    return (;
        layers=initialstates(rng, l.layers), connection=initialstates(rng, l.connection)
    )
end

(m::Parallel)(x, ps, st::NamedTuple) = applyparallel(m.layers, m.connection, x, ps, st)

function (m::Parallel{<:AbstractLuxLayer,<:NamedTuple})(x, ps, st::NamedTuple)
    y_tuple, st_layers = applyparallel(m.layers, nothing, x, ps.layers, st.layers)
    y, st_connection = apply(m.connection, y_tuple, ps.connection, st.connection)
    return y, (layers=st_layers, connection=st_connection)
end

@generated function applyparallel(
    layers::NamedTuple{names}, connection::C, x::T, ps, st::NamedTuple
) where {names,C,T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    getinput(i) = T <: Tuple ? :(x[$i]) : :x
    calls = []
    append!(
        calls,
        [
            :(
                ($(y_symbols[i]), $(st_symbols[i])) = @inline apply(
                    layers.$(names[i]), $(getinput(i)), ps.$(names[i]), st.$(names[i])
                )
            ) for i in 1:N
        ],
    )
    push!(calls, :(st = NamedTuple{$names}((($(Tuple(st_symbols)...),)))))
    if C == Nothing
        push!(calls, :($(y_symbols[N + 1]) = tuple($(Tuple(y_symbols[1:N])...))))
    else
        push!(calls, :($(y_symbols[N + 1]) = connection($(Tuple(y_symbols[1:N])...))))
    end
    push!(calls, :(return $(y_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

"""
    BranchLayer(layers...; fusion=nothing)
    BranchLayer(; fusion=nothing, name=nothing, layers...)

Takes an input `x` and passes it through all the `layers` and returns a tuple of the
outputs. If `fusion` is provided, applies fusion to the tuple of outputs.

## Arguments

  - Layers can be specified in two formats:

      + A list of `N` Lux layers
      + Specified as `N` keyword arguments.

## Keyword Arguments

  - `fusion`: An optional layer or function to apply to the tuple of outputs.
    If `fusion = nothing`, returns the tuple as-is (default behavior).
    If `fusion` is provided, returns `fusion((layer_1(x), layer_2(x), ..., layer_N(x)))`.

# Extended Help

## Inputs

  - `x`: Will be directly passed to each of the `layers`

## Returns

  - If `fusion = nothing`: Tuple `(layer_1(x), layer_2(x), ..., layer_N(x))` (naming changes 
    if using the kwargs API)
  - If `fusion` is provided: `fusion((layer_1(x), layer_2(x), ..., layer_N(x)))`
  - Updated state of the `layers` (and `fusion` if it's a layer)

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)
  - If `fusion` is an AbstractLuxLayer, parameters include both `layers` and `fusion`

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)
  - If `fusion` is an AbstractLuxLayer, states include both `layers` and `fusion`

!!! tip "Comparison with Parallel"

    This is slightly different from [`Parallel(nothing, layers...)`](@ref)

      - If the input is a tuple, `Parallel` will pass each element individually to each
        layer.

      - `BranchLayer` essentially assumes 1 input comes in and is branched out into `N`
        outputs.

## Example

An easy way to replicate an input to an NTuple is to do

```jldoctest
julia> BranchLayer(NoOpLayer(), NoOpLayer(), NoOpLayer())
BranchLayer(
    layer_(1-3) = NoOpLayer(),
)         # Total: 0 parameters,
          #        plus 0 states.
```
"""
@concrete struct BranchLayer <: AbstractLuxWrapperLayer{:layers}
    layers <: NamedTuple
    fusion
    name
end

function BranchLayer(layers...; fusion=nothing, name::NAME_TYPE=nothing)
    return BranchLayer(Utils.named_tuple_layers(layers...), fusion, name)
end

function BranchLayer(; fusion=nothing, name::NAME_TYPE=nothing, kwargs...)
    return BranchLayer((; kwargs...), fusion, name)
end

function PrettyPrinting.printable_children(l::BranchLayer)
    children = Functors.children(l)
    l.fusion === nothing && return children.layers
    return merge(children.layers, (; l.fusion))
end

function initialparameters(
    rng::AbstractRNG, l::BranchLayer{<:NamedTuple,<:AbstractLuxLayer}
)
    return (;
        layers=initialparameters(rng, l.layers), fusion=initialparameters(rng, l.fusion)
    )
end

function initialstates(rng::AbstractRNG, l::BranchLayer{<:NamedTuple,<:AbstractLuxLayer})
    return (; layers=initialstates(rng, l.layers), fusion=initialstates(rng, l.fusion))
end

(m::BranchLayer)(x, ps, st::NamedTuple) = applybranching(m.layers, m.fusion, x, ps, st)

function (m::BranchLayer{<:NamedTuple,<:AbstractLuxLayer})(x, ps, st::NamedTuple)
    y_tuple, st_layers = applybranching(m.layers, nothing, x, ps.layers, st.layers)
    y, st_fusion = apply(m.fusion, y_tuple, ps.fusion, st.fusion)
    return y, (layers=st_layers, fusion=st_fusion)
end

@generated function applybranching(
    layers::NamedTuple{names}, fusion::F, x, ps, st::NamedTuple
) where {names,F}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    calls = []
    append!(
        calls,
        [
            :(
                ($(y_symbols[i]), $(st_symbols[i])) = @inline apply(
                    layers.$(names[i]), x, ps.$(names[i]), st.$(names[i])
                )
            ) for i in 1:N
        ],
    )
    push!(calls, :(st = NamedTuple{$names}((($(Tuple(st_symbols)...),)))))
    if F == Nothing
        push!(calls, :($(y_symbols[N + 1]) = tuple($(Tuple(y_symbols[1:N])...))))
    else
        push!(calls, :($(y_symbols[N + 1]) = fusion($(Tuple(y_symbols[1:N])...))))
    end
    push!(calls, :(return $(y_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

"""
    PairwiseFusion(connection, layers...; name=nothing)
    PairwiseFusion(connection; name=nothing, layers...)
    PairwiseFusion(; connection, layers..., name=nothing)

```
x1 → layer1 → y1 ↘
                  connection → layer2 → y2 ↘
              x2 ↗                          connection → y3
                                        x3 ↗
```

## Arguments

  - `connection`: Takes 2 inputs and combines them

  - `layers`: `AbstractLuxLayer`s. Layers can be specified in two formats:

      + A list of `N` Lux layers
      + Specified as `N` keyword arguments.

# Extended Help

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
@concrete struct PairwiseFusion <: AbstractLuxWrapperLayer{:layers}
    connection
    layers <: NamedTuple
    name
end

function PrettyPrinting.printable_children(l::PairwiseFusion)
    children = Functors.children(l)
    l.connection === nothing && return children.layers
    return merge((; l.connection), children.layers)
end

function PairwiseFusion(connection, layers...; name::NAME_TYPE=nothing)
    return PairwiseFusion(connection, Utils.named_tuple_layers(layers...), name)
end

function PairwiseFusion(connection; name::NAME_TYPE=nothing, kwargs...)
    return PairwiseFusion(; connection, name, kwargs...)
end

function PairwiseFusion(; name::NAME_TYPE=nothing, connection, kwargs...)
    return PairwiseFusion(connection, (; kwargs...), name)
end

function (m::PairwiseFusion)(x, ps, st::NamedTuple)
    return applypairwisefusion(m.layers, m.connection, x, ps, st)
end

@generated function applypairwisefusion(
    layers::NamedTuple{names}, connection::C, x::T, ps, st::NamedTuple
) where {names,C,T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    getinput(i) = T <: Tuple ? :(x[$i]) : :x
    calls = [:($(y_symbols[N + 1]) = $(getinput(1)))]
    append!(
        calls,
        [
            :(
                ($(y_symbols[i]), $(st_symbols[i])) = @inline apply(
                    layers.$(names[i]), $(y_symbols[N + 1]), ps.$(names[i]), st.$(names[i])
                );
                $(y_symbols[N + 1]) = connection($(y_symbols[i]), $(getinput(i + 1)))
            ) for i in 1:N
        ],
    )
    push!(
        calls,
        :(return $(y_symbols[N + 1]), NamedTuple{$names}((($(Tuple(st_symbols)...),)))),
    )
    return Expr(:block, calls...)
end

"""
    Chain(layers...; name=nothing)
    Chain(; layers..., name=nothing)

Collects multiple layers / functions to be called in sequence on a given input.

## Arguments

  - Layers can be specified in two formats:

      + A list of `N` Lux layers
      + Specified as `N` keyword arguments.

# Extended Help

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

## Miscellaneous Properties

  - Allows indexing and field access syntax. We can access the `i`th layer by `m[i]` or
    `m.layer_i`. We can also index using ranges or arrays.

## Example

```jldoctest
julia> Chain(Dense(2, 3, relu), BatchNorm(3), Dense(3, 2))
Chain(
    layer_1 = Dense(2 => 3, relu),                # 9 parameters
    layer_2 = BatchNorm(3, affine=true, track_stats=true),  # 6 parameters, plus 7 non-trainable
    layer_3 = Dense(3 => 2),                      # 8 parameters
)         # Total: 23 parameters,
          #        plus 7 states.

julia> Chain(Dense(2, 3, relu), BatchNorm(3), Dense(3, 2); name="MyFancyChain")
MyFancyChain(
    layer_1 = Dense(2 => 3, relu),                # 9 parameters
    layer_2 = BatchNorm(3, affine=true, track_stats=true),  # 6 parameters, plus 7 non-trainable
    layer_3 = Dense(3 => 2),                      # 8 parameters
)         # Total: 23 parameters,
          #        plus 7 states.
```
"""
@concrete struct Chain <: AbstractLuxWrapperLayer{:layers}
    layers <: NamedTuple
    name
end

function Chain(xs...; name::NAME_TYPE=nothing)
    return Chain(Utils.named_tuple_layers(wrap_functions_in_chain_call(xs)...), name)
end
Chain(xs::AbstractVector; kwargs...) = Chain(xs...; kwargs...)
Chain(nt::NamedTuple; name::NAME_TYPE=nothing) = Chain(nt, name)

function Chain(; name::NAME_TYPE=nothing, kwargs...)
    if name === nothing && isempty(kwargs)
        # a valid chain that does nothing
        return Chain(NoOpLayer())
    end
    return Chain((; kwargs...); name=name)
end

function wrap_functions_in_chain_call(layers::Union{AbstractVector,Tuple})
    new_layers = []
    for l in layers
        f = wrap_functions_in_chain_call(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
        elseif f isa Function
            push!(new_layers, WrappedFunction(f))
        elseif f isa AbstractLuxLayer
            push!(new_layers, f)
        else
            throw("Encountered a non-AbstractLuxLayer in Chain.")
        end
    end
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

wrap_functions_in_chain_call(x) = x

(c::Chain)(x, ps, st::NamedTuple) = applychain(c.layers, x, ps, st)

@generated function applychain(
    layers::NamedTuple{fields}, x, ps, st::NamedTuple{fields}
) where {fields}
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    st_symbols = [gensym() for _ in 1:N]
    calls = [
        :(
            ($(x_symbols[i + 1]), $(st_symbols[i])) = @inline apply(
                layers.$(fields[i]), $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i])
            )
        ) for i in 1:N
    ]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(x_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

Base.getindex(c::Chain, i::Int) = c.layers[i]
Base.getindex(c::Chain, i::AbstractArray) = Chain(Utils.index_namedtuple(c.layers, i))

function Base.getproperty(c::Chain, name::Symbol)
    hasfield(typeof(c), name) && return getfield(c, name)
    layers = getfield(c, :layers)
    hasfield(typeof(layers), name) && return getfield(layers, name)
    throw(ArgumentError("$(typeof(c)) has no field or layer $name"))
end

Base.length(c::Chain) = length(c.layers)
Base.lastindex(c::Chain) = lastindex(c.layers)
Base.firstindex(c::Chain) = firstindex(c.layers)

"""
    Maxout(layers...)
    Maxout(; layers...)
    Maxout(f::Function, n_alts::Int)

This contains a number of internal layers, each of which receives the same input. Its output
is the elementwise maximum of the the internal layers' outputs.

Maxout over linear dense layers satisfies the universal approximation theorem
[goodfellow2013maxout](@cite).

See also [`Parallel`](@ref) to reduce with other operators.

## Arguments

  - Layers can be specified in three formats:

      + A list of `N` Lux layers
      + Specified as `N` keyword arguments.
      + A no argument function `f` and an integer `n_alts` which specifies the number of
        layers.

# Extended Help

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
"""
@concrete struct Maxout <: AbstractLuxWrapperLayer{:layers}
    layers <: NamedTuple
end

Maxout(layers...) = Maxout(Utils.named_tuple_layers(layers...))
Maxout(; kwargs...) = Maxout((; kwargs...))
Maxout(f::Function, n_alts::Int) = Maxout(ntuple(Returns(f()), n_alts)...)

# NOTE(@avik-pal): Calling `applyparallel` with broadcasted `max` is slower than this
#                  implementation.
(m::Maxout)(x, ps, st::NamedTuple) = applymaxout(m.layers, x, ps, st)

@generated function applymaxout(
    layers::NamedTuple{fields}, x, ps, st::NamedTuple{fields}
) where {fields}
    N = length(fields)
    y_symbols = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = [
        :(
            ($(y_symbols[i]), $(st_symbols[i])) = @inline apply(
                layers.$(fields[i]), x, ps.$(fields[i]), st.$(fields[i])
            )
        ) for i in 1:N
    ]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(res = max.($(Tuple(y_symbols)...))))
    push!(calls, :(return res, st))
    return Expr(:block, calls...)
end

"""
    RepeatedLayer(model; repeats::Val = Val(10), input_injection::Val = Val(false))

Iteratively applies `model` for `repeats` number of times. The initial input is passed
into the model repeatedly if `input_injection = Val(true)`. This layer unrolls the
computation, however, semantically this is same as:

  - `input_injection = Val(false)`

    ```julia
    res = x
    for i in 1:repeats
        res, st = model(res, ps, st)
    end
    ```

  - `input_injection = Val(true)`

    ```julia
    res = x
    for i in 1:repeats
        res, st = model((res, x), ps, st)
    end
    ```

It is expected that `repeats` will be a reasonable number below `20`, beyond that compile
times for gradients might be unreasonably high.

## Arguments

  - `model` must be an `AbstractLuxLayer`

## Keyword Arguments

  - `repeats`: Number of times to apply the model
  - `input_injection`: If `true`, then the input is passed to the model along with the
    output

# Extended Help

## Inputs

  - `x`: Input as described above

## Returns

  - Output is computed by as described above
  - Updated state of the `model`

## Parameters

  - Parameters of `model`

## States

  - State of `model`
"""
@concrete struct RepeatedLayer <: AbstractLuxWrapperLayer{:model}
    nrepeats <: StaticInt
    input_injection <: StaticBool
    model <: AbstractLuxLayer
end

function LuxCore.display_name(r::RepeatedLayer)
    return "RepeatedLayer{nrepeats = $(known(r.nrepeats)), \
                          input_injection = $(known(r.input_injection))}"
end

function RepeatedLayer(
    model::AbstractLuxLayer;
    repeats::Union{StaticInt,Integer,Val}=Val(10),
    input_injection::Union{StaticBool,Bool,Val{true},Val{false}}=Val(false),
)
    return RepeatedLayer(static(repeats), static(input_injection), model)
end

(m::RepeatedLayer)(x, ps, st) = repeatedlayer(m, m.model, x, ps, st)

@generated function repeatedlayer(::RepeatedLayer{N,IJ}, model, x, ps, st) where {N,IJ}
    sts = ntuple(_ -> gensym("st"), known(N))
    xs = ntuple(_ -> gensym("x"), known(N) + known(IJ))
    calls = []
    known(IJ) && push!(calls, :($(xs[1]) = x))
    for i in 1:known(N)
        push!(
            calls,
            :(
                ($(xs[i + known(IJ)]), $(sts[i])) = @inline apply(
                    model,
                    $(known(IJ) ? :(($(xs[i]), x)) : :x),
                    ps,
                    $(i == 1 ? :st : sts[i - 1]),
                )
            ),
        )
    end
    return quote
        $(calls...)
        return $(last(xs)), $(last(sts))
    end
end
