module SimpleChainsExt

using SimpleChains: SimpleChains
using Random: AbstractRNG

using LuxCore: LuxCore
using Lux:
    Lux,
    SimpleChainsModelConversionException,
    SimpleChainsLayer,
    make_simplechain_network,
    fix_simplechain_input_dims,
    Chain,
    Conv,
    Dense,
    Dropout,
    FlattenLayer,
    MaxPool,
    SamePad
using NNlib: NNlib

Lux.is_extension_loaded(::Val{:SimpleChains}) = true

function Lux.fix_simplechain_input_dims(layers::Vector, input_dims)
    L = Tuple(layers)
    return SimpleChains.SimpleChain{typeof(input_dims),typeof(L)}(input_dims, L)
end

function Lux.fix_simplechain_input_dims(layers, input_dims)
    @warn "The model provided is not a `Chain`. Trying to wrap it into a `Chain` but this \
           might fail. Please consider using `Chain` directly."
    return fix_simplechain_input_dims([layers], input_dims)
end

equivalent_simplechains_fn(::typeof(NNlib.relu)) = SimpleChains.relu
equivalent_simplechains_fn(f::F) where {F} = f

function Lux.make_simplechain_network(layer::Dense)
    return SimpleChains.TurboDense{Lux.has_bias(layer)}(
        equivalent_simplechains_fn(layer.activation), layer.out_dims
    )
end

function Lux.make_simplechain_network(layer::Chain)
    return reduce(vcat, map(make_simplechain_network, layer.layers))
end

function Lux.make_simplechain_network(layer::Conv)
    if all(==(1), layer.stride) &&
        layer.groups == 1 &&
        all(==(1), layer.dilation) &&
        (!(layer.pad isa SamePad) && all(==(0), layer.pad))
        return SimpleChains.Conv(
            equivalent_simplechains_fn(layer.activation), layer.kernel_size, layer.out_chs
        )
    end
    throw(SimpleChainsModelConversionException("Conv with non-standard parameters not \
                                                supported."))
end

function Lux.make_simplechain_network(layer::Dropout)
    layer.dims isa Colon && return SimpleChains.Dropout(layer.p)
    throw(SimpleChainsModelConversionException("Dropout with non-standard parameters not \
                                                supported."))
end

function Lux.make_simplechain_network(layer::FlattenLayer)
    if layer.N === nothing
        throw(
            SimpleChainsModelConversionException("`FlattenLayer(nothing)` not supported. \
                                                  For `SimpleChains.Flatten` you must \
                                                  use `FlattenLayer(N::Int)`")
        )
    end
    return SimpleChains.Flatten(layer.N)
end

function Lux.make_simplechain_network(layer::MaxPool)
    if layer.layer.mode.stride == layer.layer.mode.kernel_size &&
        all(==(0), layer.layer.mode.pad)
        return SimpleChains.MaxPool(layer.layer.mode.kernel_size)
    end
    throw(SimpleChainsModelConversionException("MaxPool with non-standard parameters not \
                                                supported."))
end

Lux.make_simplechain_network(layer) = throw(SimpleChainsModelConversionException(layer))

function LuxCore.initialparameters(rng::AbstractRNG, layer::SimpleChainsLayer)
    return (; params=Array(SimpleChains.init_params(layer.layer; rng)))
end

# Some type-piracy for nicer interaction with NNlib
NNlib.logsoftmax(x::SimpleChains.StrideArray{T,2}) where {T} = SimpleChains.logsoftmax(x)

function NNlib.logsoftmax!(
    y::SimpleChains.StrideArray{T1,2},
    x::Union{SimpleChains.StrideArray{T2,2},SimpleChains.PtrArray{T2,2}};
    dims=1,
) where {T1,T2}
    @assert dims == 1
    m = similar(y, SimpleChains.static_size(y, 2))
    SimpleChains.logsoftmax!(y, m, x)
    return y
end

end
