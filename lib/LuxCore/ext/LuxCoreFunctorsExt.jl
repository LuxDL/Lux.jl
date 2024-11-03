module LuxCoreFunctorsExt

using LuxCore: LuxCore
using Functors: Functors

LuxCore.Internal.is_extension_loaded(::Val{:Functors}) = true

LuxCore.Internal.isleaf_impl(args...; kwargs...) = Functors.isleaf(args...; kwargs...)
LuxCore.Internal.fmap_impl(args...; kwargs...) = Functors.fmap(args...; kwargs...)
function LuxCore.Internal.fmap_with_path_impl(args...; kwargs...)
    return Functors.fmap_with_path(args...; kwargs...)
end
LuxCore.Internal.fleaves_impl(args...; kwargs...) = Functors.fleaves(args...; kwargs...)

function Functors.functor(::Type{<:LuxCore.AbstractLuxContainerLayer{layers}},
        x) where {layers}
    children = NamedTuple{layers}(getproperty.((x,), layers))
    layer_reconstructor = let x = x, layers = layers
        z -> reduce(LuxCore.Internal.setfield, zip(layers, z); init=x)
    end
    return children, layer_reconstructor
end

function Functors.functor(::Type{<:LuxCore.AbstractLuxWrapperLayer{layer}},
        x) where {layer}
    children = NamedTuple{(layer,)}((getproperty(x, layer),))
    layer_reconstructor = let x = x, layer = layer
        z -> LuxCore.Internal.setfield(x, layer, getproperty(z, layer))
    end
    return children, layer_reconstructor
end

end
