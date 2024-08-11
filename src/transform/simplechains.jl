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
julia> import SimpleChains

julia> using Adapt, Lux, Random

julia> lux_model = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
           Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
           Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)));

julia> adaptor = ToSimpleChainsAdaptor((28, 28, 1))
ToSimpleChainsAdaptor{Tuple{Static.StaticInt{28}, Static.StaticInt{28}, Static.StaticInt{1}}}((static(28), static(28), static(1)), false)

julia> simple_chains_model = adapt(adaptor, lux_model) # or adaptor(lux_model)
SimpleChainsLayer{false}(
    Chain(
        layer_1 = Conv((5, 5), 1 => 6, relu),  # 156 parameters
        layer_2 = MaxPool((2, 2)),
        layer_3 = Conv((5, 5), 6 => 16, relu),  # 2_416 parameters
        layer_4 = MaxPool((2, 2)),
        layer_5 = FlattenLayer{Int64}(3),
        layer_6 = Dense(256 => 128, relu),  # 32_896 parameters
        layer_7 = Dense(128 => 84, relu),  # 10_836 parameters
        layer_8 = Dense(84 => 10),      # 850 parameters
    ),
)         # Total: 47_154 parameters,
          #        plus 0 states.

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
        input_dims isa Tuple{Vararg{Integer}} && (input_dims = static(input_dims))
        return new{typeof(input_dims)}(input_dims, convert_to_array)
    end
end

"""
    Adapt.adapt(from::ToSimpleChainsAdaptor, L::AbstractExplicitLayer)

Adapt a Simple Chains model to Lux model. See [`ToSimpleChainsAdaptor`](@ref) for more
details.
"""
function Adapt.adapt(to::ToSimpleChainsAdaptor, L::AbstractExplicitLayer)
    if Base.get_extension(@__MODULE__, :LuxSimpleChainsExt) === nothing
        error("`ToSimpleChainsAdaptor` requires `SimpleChains.jl` to be loaded.")
    end
    sc_layer = fix_simplechain_input_dims(make_simplechain_network(L), to.input_dims)
    return SimpleChainsLayer{to.convert_to_array}(sc_layer, L)
end

function make_simplechain_network end
function fix_simplechain_input_dims end
