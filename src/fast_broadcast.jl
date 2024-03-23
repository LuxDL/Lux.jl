# This file temporarily exists here. It will be moved to LuxLib.jl in the future.
using FastBroadcast: @..

# Adapted from NNlib.jl
# This just saves typing `only.(only.(` many times:
# `sigmoid_fast` fails if we use the fast path, don't know why we just avoid the fast
# gradient path for it
@inline function __only_derivative(y, f::F, x) where {F}
    return only(only(CRC.derivatives_given_output(y, f, x)))
end

# This has no methods, used for testing whether `derivatives_given_output(Ω, f, x)`
# is independent of `x`, as `_return_type` says `Union{}` when calling is an error.
struct NotaNumber <: Real end

@inline fast_apply_activation!!(::typeof(identity), x::AbstractArray) = x
@inline function fast_apply_activation!!(f::F, x::AbstractArray) where {F}
    return fast_fast_broadcast!!(f, x)
end

function CRC.rrule(cfg::RuleConfig{>:CRC.HasReverseMode}, ::typeof(fast_apply_activation!!),
        f::F, x::AbstractArray{T}) where {F, T}
    if f === identity
        ∇identity_shortcut(Δ) = NoTangent(), NoTangent(), Δ
        return x, ∇identity_shortcut
    end

    # Fast path: it is now safe to overwrite x, since this is not needed for gradient of σ
    if f !== sigmoid_fast && isconcretetype(Core.Compiler._return_type(
        __only_derivative, Tuple{T, F, NotaNumber}))
        Ω = fast_apply_activation!!(f, x)
        ∇fast_apply_activation!!_fast = @closure Δ -> begin
            ∂x = __only_derivative.(Ω, f, NotaNumber()) .* CRC.unthunk(Δ)
            return NoTangent(), NoTangent(), ∂x
        end
        return Ω, ∇fast_apply_activation!!_fast
    end

    return CRC.rrule_via_ad(cfg, broadcast, f, x)
end

# Bias Activation Fused
function fast_bias_activation!!(f::F, x::AbstractArray, b::AbstractArray) where {F}
    f === identity && return fast_broadcast!!(+, x, b)
    return fast_broadcast!!(f ∘ +, x, b)
end

function CRC.rrule(cfg::RuleConfig{>:CRC.HasReverseMode}, ::typeof(fast_bias_activation!!),
        f::F, x::AbstractArray{T, N}, b::AbstractArray) where {F, T, N}
    # Summing over ndims(x)+1 is a trick to make b_dims type-stable
    dims = ntuple(d -> ifelse(size(b, d) == 1, d, N + 1), N)
    ∇bias(dx) = reshape(sum(dx; dims), size(b))

    if f === identity
        Ω = fast_bias_activation!!(f, x, b)
        ∇identity_shortcut(Δ) = NoTangent(), NoTangent(), Δ, ∇bias(Δ)
        return Ω, ∇identity_shortcut
    end

    if f !== sigmoid_fast && isconcretetype(Core.Compiler._return_type(
        __only_derivative, Tuple{T, F, NotaNumber}))
        Ω = fast_bias_activation!!(f, x, b)
        ∇fast_bias_activation!!_fast = @closure Δ -> begin
            ∂x = __only_derivative.(Ω, f, NotaNumber()) .* CRC.unthunk(Δ)
            return NoTangent(), NoTangent(), ∂x, ∇bias(∂x)
        end
        return Ω, ∇fast_bias_activation!!_fast
    end

    return CRC.rrule_via_ad(cfg, fast_broadcast!!, f ∘ +, x, b)
end

# FastBroadcast.jl is efficient only for same axes arrays
@inline fast_broadcast!!(f::F, x) where {F} = fast_fast_broadcast!!(f, x)
@inline function fast_broadcast!!(f::F, x, ys...) where {F}
    ax = axes(x)
    all(x -> axes(x) == ax, ys) && return fast_fast_broadcast!!(f, x, ys...)
    return fast_generic_broadcast!!(f, x, ys...)
end

## Just use non-mutating version for the broadcast
function CRC.rrule(cfg::RuleConfig{>:CRC.HasReverseMode},
        ::typeof(fast_broadcast!!), f::F, x, ys...) where {F}
    return CRC.rrule_via_ad(cfg, broadcast, f, x, ys...)
end

@inline function fast_fast_broadcast!!(f::F, x, ys...) where {F}
    ArrayInterface.can_setindex(x) && return @..(x=f(x, ys...))
    return @..(f(x, ys...))
end

@inline function fast_generic_broadcast!!(f::F, x, ys...) where {F}
    if all(ArrayInterface.fast_scalar_indexing, (x, ys...))
        bc = Broadcast.instantiate(Broadcast.broadcasted(f, x, ys...))
        ArrayInterface.can_setindex(x) || return copy(bc)
        @simd ivdep for idx in eachindex(bc)
            @inbounds x[idx] = bc[idx]
        end
        return x
    end
    ArrayInterface.can_setindex(x) && return @.(x=f(x, ys...))
    return @.(f(x, ys...))
end
