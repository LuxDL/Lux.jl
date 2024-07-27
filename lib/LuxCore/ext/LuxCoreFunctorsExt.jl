module LuxCoreFunctorsExt

using LuxCore: LuxCore
using Functors: Functors

LuxCore._is_extension_loaded(::Val{:Functors}) = true

LuxCore._isleaf(x) = Functors.isleaf(x)
LuxCore._fmap(args...; kwargs...) = Functors.fmap(args...; kwargs...)
LuxCore._fleaves(args...; kwargs...) = Functors.fleaves(args...; kwargs...)

function Functors.functor(::Type{<:LuxCore.AbstractLuxContainerLayer{layers}},
        x) where {layers}
    if !LuxCore._is_extension_loaded(Val(:Setfield))
        throw(ArgumentError("`Functors.functor` for `AbstractLuxContainerLayer` requires \
                             `Setfield.jl` to be loaded."))
    end
    _children = NamedTuple{layers}(getproperty.((x,), layers))
    layer_reconstructor = let x = x, layers = layers
        z -> reduce(LuxCore._setfield, zip(layers, z); init=x)
    end
    return _children, layer_reconstructor
end

end
