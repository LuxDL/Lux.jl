module LuxForwardDiffExt

using ChainRulesCore: ChainRulesCore
using Lux: Lux
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using Functors: Functors

const CRC = ChainRulesCore

@inline Lux._is_extension_loaded(::Val{:ForwardDiff}) = true

@inline __partials(::Type{Tag}, x::AbstractArray, i) where {Tag} = ForwardDiff.partials.(
    Tag, x, i)
@inline __partials(::Type{Tag}, x::Tuple, i) where {Tag} = map(
    @closure(xᵢ->__partials(Tag, xᵢ, i)), x)
@inline __partials(::Type{Tag}, x::NamedTuple{F}, i) where {Tag, F} = NamedTuple{F}(map(
    @closure(xᵢ->__partials(Tag, xᵢ, i)), values(x)))
@inline __partials(::Type{Tag}, x, i) where {Tag} = fmap(
    @closure(xᵢ->__partials(Tag, xᵢ, i)), x)

# This is not a general jvp code, but rather meant to be efficient for nested AD calls
function Lux.__forwarddiff_jvp(
        f::F, x::AbstractArray{xT}, Δx::AbstractArray{ΔxT}, ps) where {F, xT, ΔxT}
    T = promote_type(xT, ΔxT)
    Tag = typeof(ForwardDiff.Tag(f, T))
    partials = ForwardDiff.Partials{1, T}.(tuple.(Δx))
    x_dual = ForwardDiff.Dual{Tag, T, 1}.(x, reshape(partials, size(x)))
    y_dual, ps_dual = f(x_dual, ps)
    return __partials(Tag, y_dual, 1), __partials(Tag, ps_dual, 1)
end

end
