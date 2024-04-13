"""
    ToSimpleChainsAdaptor(input_dims, convert_to_array::Bool=false)

Adaptor for converting a Lux Model to SimpleChains. The returned model is still a Lux model,
and satisfies the `AbstractExplicitLayer` interfacem but all internal calculations are
performed using SimpleChains.

!!! warning

    There is no way to preserve trained parameters and states when converting to
    `SimpleChains.jl`.

!!! warning

    Any kind of initialization function is not preserved when converting to
    `SimpleChains.jl`.

## Arguments

  - `input_dims`: Tuple of input dimensions excluding the batch dimension. These must be of
    `static` type as `SimpleChains` expects.
  - `convert_to_array`: SimpleChains.jl by default outputs `StrideArraysCore.StrideArray`,
    but this might not compose well with other packages. If `convert_to_array` is set to
    `true`, the output will be converted to a regular `Array`.

## Example

```jldoctest
julia> import SimpleChains: static

julia> using Adapt, Lux, Random

julia> lux_model = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
           Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
           Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)));

julia> adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))
ToSimpleChainsAdaptor{Tuple{Static.StaticInt{28}, Static.StaticInt{28}, Static.StaticInt{1}}}((static(28), static(28), static(1)), false)

julia> simple_chains_model = adapt(adaptor, lux_model) # or adaptor(lux_model)
SimpleChainsLayer()  # 47_154 parameters

julia> ps, st = Lux.setup(Random.default_rng(), simple_chains_model);

julia> x = randn(Float32, 28, 28, 1, 1);

julia> size(first(simple_chains_model(x, ps, st))) == (10, 1)
true
```
"""
struct ToSimpleChainsAdaptor{ID} <: AbstractFromLuxAdaptor
    input_dims::ID
    convert_to_array::Bool

    function ToSimpleChainsAdaptor(input_dims, convert_to_array::Bool=false)
        input_dims isa Number && (input_dims = (input_dims,))
        if input_dims isa Tuple{Vararg{Integer}}
            throw(ArgumentError("`input_dims` must be a Tuple of `static` integers."))
        end
        return new{typeof(input_dims)}(input_dims, convert_to_array)
    end
end

"""
    Adapt.adapt(from::ToSimpleChainsAdaptor, L::AbstractExplicitLayer)

Adapt a Flux model to Lux model. See [`ToSimpleChainsAdaptor`](@ref) for more details.
"""
function Adapt.adapt(to::ToSimpleChainsAdaptor, L::AbstractExplicitLayer)
    if Base.get_extension(@__MODULE__, :LuxSimpleChainsExt) === nothing
        error("`ToSimpleChainsAdaptor` requires `SimpleChains.jl` to be loaded.")
    end
    sc_layer = __fix_input_dims_simplechain(__to_simplechains_adaptor(L), to.input_dims)
    return SimpleChainsLayer{to.convert_to_array}(sc_layer)
end

function __to_simplechains_adaptor end
function __fix_input_dims_simplechain end

struct SimpleChainsModelConversionError <: Exception
    msg::String
end

function SimpleChainsModelConversionError(layer::AbstractExplicitLayer)
    return SimpleChainsModelConversionError(lazy"Conversion to SimpleChains not supported for $(typeof(layer))")
end

function Base.showerror(io::IO, e::SimpleChainsModelConversionError)
    print(io, "SimpleChainsModelConversionError(", e.msg, ")")
    return
end

"""
    SimpleChainsLayer{ToArray}(layer)
    SimpleChainsLayer(layer, ToArray::Bool=false)

Wraps a `SimpleChains` layer into a `Lux` layer. All operations are performed using
`SimpleChains` but the layer satisfies the `AbstractExplicitLayer` interface.

`ToArray` is a boolean flag that determines whether the output should be converted to a
regular `Array` or not. Default is `false`.

## Arguments

  - `layer`: SimpleChains layer

!!! note

    Using the 2nd constructor makes the generation of the model struct type unstable.

!!! note

    If using `Tracker.jl`, the output will always be a regular `Array`.
"""
struct SimpleChainsLayer{ToArray, L} <: AbstractExplicitLayer
    layer::L
end

@inline function SimpleChainsLayer{ToArray}(layer) where {ToArray}
    return SimpleChainsLayer{ToArray, typeof(layer)}(layer)
end
@inline SimpleChainsLayer(layer, ToArray::Bool=false) = SimpleChainsLayer{ToArray}(layer)

@inline initialstates(::AbstractRNG, ::SimpleChainsLayer) = (;)

@inline function (sc::SimpleChainsLayer{false})(x, ps, st)
    return __apply_simple_chain(sc.layer, x, ps.params, get_device(x)), st
end

@inline function (sc::SimpleChainsLayer{true})(x, ps, st)
    return convert(Array, __apply_simple_chain(sc.layer, x, ps.params, get_device(x))), st
end

@inline __apply_simple_chain(layer, x, ps, ::LuxCPUDevice) = layer(x, ps)

function __apply_simple_chain(layer, x, ps, dev)
    throw(ArgumentError(lazy"`SimpleChains.jl` only supports CPU operations. Current device detected as $(dev)."))
end

# Workaround for SimpleChains not being able to handle some input types
function CRC.rrule(::typeof(__apply_simple_chain), layer, x, ps, ::LuxCPUDevice)
    res, pb = CRC.rrule(layer, x, ps)
    __∇apply_simple_chain = @closure Δ -> begin
        # Safety measure to prevent errors from weird Array types that SimpleChains doesn't
        # support
        ∂layer, ∂x, ∂ps = pb(convert(Array, Δ))
        return CRC.NoTangent(), ∂layer, ∂x, ∂ps, CRC.NoTangent()
    end
    return res, __∇apply_simple_chain
end
