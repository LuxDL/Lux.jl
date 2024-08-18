module LuxCoreFunctorsExt

using LuxCore: LuxCore
using Functors: Functors

LuxCore.Internal.is_extension_loaded(::Val{:Functors}) = true

LuxCore.Internal.isleaf(x) = Functors.isleaf(x)
LuxCore.Internal.fmap(args...; kwargs...) = Functors.fmap(args...; kwargs...)
LuxCore.Internal.fleaves(args...; kwargs...) = Functors.fleaves(args...; kwargs...)

function Functors.functor(::Type{<:LuxCore.AbstractLuxContainerLayer{layers}},
        x) where {layers}
    _children = NamedTuple{layers}(getproperty.((x,), layers))
    layer_reconstructor = let x = x, layers = layers
        z -> reduce(LuxCore._setfield, zip(layers, z); init=x)
    end
    return _children, layer_reconstructor
end

function Functors.functor(::Type{<:LuxCore.AbstractLuxWrapperLayer{layer}},
        x) where {layer}
    _children = NamedTuple{(layer,)}((getproperty(x, layer),))
    layer_reconstructor = let x = x, layer = layer
        z -> LuxCore._setfield(x, layer, getproperty(z, layer))
    end
    return _children, layer_reconstructor
end

end
