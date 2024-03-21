# This file temporarily exists here. It will be moved to LuxLib.jl in the future.
using FastBroadcast: @..

# Adapted from NNlib.jl
# This just saves typing `only.(only.(` many times:
@inline function __only_derivative(y, f::F, x) where {F}
    return only(only(CRC.derivatives_given_output(y, f, x)))
end

# This has no methods, used for testing whether `derivatives_given_output(Ω, f, x)`
# is independent of `x`, as `_return_type` says `Union{}` when calling is an error.
struct NotaNumber <: Real end

@inline fast_apply_activation!!(::typeof(identity), x::AbstractArray) = x
@inline function fast_apply_activation!!(f::F, x::AbstractArray) where {F}
    @.. x = f(x)
    return x
end

function CRC.rrule(cfg::RuleConfig{>:CRC.HasReverseMode}, ::typeof(fast_apply_activation!!),
        f::F, x::AbstractArray{T}) where {F, T}
    if f === identity
        ∇identity_shortcut(Δ) = NoTangent(), NoTangent(), Δ
        return x, ∇identity_shortcut
    end

    # Fast path: it is now safe to overwrite x, since this is not needed for gradient of σ
    if isconcretetype(Core.Compiler._return_type(
        __only_derivative, Tuple{T, F, NotaNumber}))
        Ω = fast_apply_activation!!(f, x)
        @inline function ∇fast_apply_activation!!_fast(Δ)
            ∂x = __only_derivative.(Ω, f, NotaNumber()) .* CRC.unthunk(Δ)
            return NoTangent(), NoTangent(), ∂x
        end
        return Ω, ∇fast_apply_activation!!_fast
    end

    return CRC.rrule_via_ad(cfg, broadcast, f, x)
end
