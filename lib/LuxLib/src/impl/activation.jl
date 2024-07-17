# Used inside rrules
__activation_gradient(Δ, out, ::typeof(identity), x) = Δ
function __activation_gradient(Δ, out, act::F, x) where {F}
    if unrolled_all(fast_scalar_indexing, (Δ, out, x))  # All sizes are same
        y = similar(out)
        if x isa NotaNumber
            @simd ivdep for i in eachindex(Δ, out)
                @inbounds y[i] = only_derivative(out[i], act, x) * Δ[i]
            end
        else
            @simd ivdep for i in eachindex(Δ, out, x)
                @inbounds y[i] = only_derivative(out[i], act, x[i]) * Δ[i]
            end
        end
        return y
    end
    only_deriv = @closure (Δᵢ, oᵢ, xᵢ) -> Δᵢ * only_derivative(oᵢ, act, xᵢ)
    return broadcast(only_deriv, Δ, out, x)
end

# Entry Points to the implementation
_fast_activation(::typeof(identity), x::AbstractArray) = x

@stable default_mode="warn" function _fast_activation(σ::F, x::AbstractArray) where {F}
    if fast_scalar_indexing(x)
        RT = Core.Compiler._return_type(f, Tuple{eltype(x)})
        y = similar(x, RT)
        @simd ivdep for I in eachindex(y, x)
            @inbounds y[I] = σ(x[I])
        end
        return y
    end
    return broadcast(σ, x)
end

@stable default_mode="warn" function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(_fast_activation),
        σ::F, x::AbstractArray{T}) where {F, T}
    return CRC.rrule_via_ad(cfg, broadcast, σ, x)
end

_fast_activation!(::typeof(identity), x::AbstractArray) = x

@stable default_mode="warn" function _fast_activation!(σ::F, x::AbstractArray) where {F}
    if fast_scalar_indexing(x)
        @simd ivdep for I in eachindex(x)
            @inbounds x[I] = σ(x[I])
        end
        return x
    end
    broadcast!(σ, x, x)
    return x
end

# Define rrule for `fast_activation!!`
@stable default_mode="warn" function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(fast_activation!!),
        σ::F, x::AbstractArray{T}) where {F, T}
    can_setindex(typeof(x)) || return CRC.rrule_via_ad(cfg, _fast_activation, σ, x)

    σ === identity && return x, @closure(Δ->(NoTangent(), NoTangent(), Δ))

    if __no_intermediate_needed(σ, T)
        _fast_activation!(σ, x) # Safe to overwrite x
        proj_x_no_cached = CRC.ProjectTo(x)
        ∇__fast_activation_impl_no_cached = @closure Δ -> begin
            ∂x = __activation_gradient(Δ, x, σ, NotaNumber())
            return NoTangent(), NoTangent(), proj_x_no_cached(∂x)
        end
        return x, ∇__fast_activation_impl_no_cached
    end

    if __needs_intermediate_but_has_rrule(σ, T)
        y = _fast_activation(σ, x)
        proj_x_cached = CRC.ProjectTo(x)
        ∇__fast_activation_impl_cached_crc = @closure Δ -> begin
            ∂x = __activation_gradient(CRC.unthunk(Δ), y, σ, x)
            return NoTangent(), NoTangent(), proj_x_cached(∂x)
        end
        return y, ∇__fast_activation_impl_cached_crc
    end

    return CRC.rrule_via_ad(cfg, _fast_activation, σ, x)
end
