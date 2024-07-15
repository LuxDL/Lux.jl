function __activation_gradient(Δ, out, act::F, x) where {F}
    only_deriv = @closure (Δᵢ, oᵢ, xᵢ) -> Δᵢ * only_derivative(oᵢ, act, xᵢ)
    return _fast_broadcast(only_deriv, Δ, out, x)
end

# Entry Points to the implementation
function _fast_broadcast(f::F, x::AbstractArray, args...) where {F}
    unrolled_any(__has_tracked_value, (x, args...)) && return broadcast(f, x, args...)
    return __fast_broadcast_impl(get_device_type((x, args...)), f, x, args...)
end

_fast_broadcast(::typeof(identity), x::AbstractArray) = x

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(_fast_broadcast),
        f::F, x::AbstractArray, args::AbstractArray...) where {F}
    return CRC.rrule_via_ad(cfg, broadcast, f, x, args...)
end

function _fast_broadcast!(f::F, x::AbstractArray, args...) where {F}
    unrolled_any(__has_tracked_value, (x, args...)) && return broadcast!(f, x, x, args...)
    return __fast_broadcast_impl!(get_device_type((x, args...)), f, x, args...)
end

_fast_broadcast!(::typeof(identity), x::AbstractArray) = x

# Main Implementations: Generic Version
## OOP Version
function __fast_broadcast_impl(::Type{T}, f::F, x::AbstractArray, args...) where {F, T}
    if unrolled_all(fast_scalar_indexing, (x, args...))
        bc = Broadcast.instantiate(Broadcast.broadcasted(f, x, args...))
        RT = Core.Compiler._return_type(f, Tuple{T})
        y = similar(x, ifelse(isconcretetype(RT), RT, eltype(x)))
        @simd ivdep for I in eachindex(bc)
            @inbounds y[I] = bc[I]
        end
        return y
    end
    return __fast_broadcast_impl(Nothing, f, x, args...)
end

for f in (sigmoid_fast, swish)
    comp_type = typeof(f ∘ +)
    @eval function __fast_broadcast_impl(::Type{<:AbstractLuxGPUDevice}, f::$(comp_type),
            x::AbstractArray, y::AbstractArray)
        return @. $(f)(x + y)
    end
end

function __fast_broadcast_impl(
        ::Type{<:AbstractLuxGPUDevice}, f::F, x::AbstractArray, args...) where {F}
    return @. f(x, args...)
end

## IIP Version
function __fast_broadcast_impl!(
        ::Type{LuxCPUDevice}, f::F, x::AbstractArray, args...) where {F}
    if unrolled_all(fast_scalar_indexing, (x, args...))
        bc = Broadcast.instantiate(Broadcast.broadcasted(f, x, args...))
        @simd ivdep for I in eachindex(bc)
            @inbounds x[I] = bc[I]
        end
        return x
    end
    return __fast_broadcast_impl!(Nothing, f, x, args...)
end

for f in (sigmoid_fast, swish)
    comp_type = typeof(f ∘ +)
    @eval function __fast_broadcast_impl!(::Type{<:AbstractLuxGPUDevice}, f::$(comp_type),
            x::AbstractArray, y::AbstractArray)
        @. x = $(f)(x + y)
        return x
    end
end

function __fast_broadcast_impl!(::Type{T}, f::F, x::AbstractArray, args...) where {F, T}
    return broadcast!(f, x, x, args...)
end

# Special Cases where we don't need to go down the generic path
## rrule for activation functions -- we need to define this on `fast_broadcast!!`
function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(fast_broadcast!!),
        f::F, x::AbstractArray{T}) where {F, T}
    σ === identity && return x, @closure(Δ->(NoTangent(), NoTangent(), Δ))

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        x = fast_broadcast!!(f, x) # Safe to overwrite x
        ∇__fast_broadcast_impl_no_cached = @closure Δ -> begin
            ∂x = __activation_gradient(Δ, x, σ, NotaNumber())
            return NoTangent(), NoTangent(), ∂x
        end
        return x, ∇__fast_broadcast_impl_no_cached
    end

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        y = _fast_broadcast(f, x)
        ∇__fast_broadcast_impl_cached_crc = @closure Δ -> begin
            ∂y = __activation_gradient(CRC.unthunk(Δ), y, f, x)
            return NoTangent(), NoTangent(), ∂y
        end
        return y, ∇__fast_broadcast_impl_cached_crc
    end

    return CRC.rrule_via_ad(cfg, broadcast, f, x)
end
