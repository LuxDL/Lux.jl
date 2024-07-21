# Used inside rrules
__activation_gradient(Δ, out, ::typeof(identity), x) = Δ
function __activation_gradient(Δ, out, act::F, x) where {F}
    opmode = internal_operation_mode((Δ, out, x))
    if opmode isa LoopedArrayOp  # All sizes are same
        y = similar(out)
        if x isa NotaNumber
            @inbounds @simd ivdep for i in eachindex(Δ, out)
                y[i] = only_derivative(out[i], act, x) * Δ[i]
            end
        else
            @inbounds @simd ivdep for i in eachindex(Δ, out, x)
                y[i] = only_derivative(out[i], act, x[i]) * Δ[i]
            end
        end
        return y
    end
    only_deriv = @closure (Δᵢ, oᵢ, xᵢ) -> Δᵢ * only_derivative(oᵢ, act, xᵢ)
    return broadcast(only_deriv, Δ, out, x)
end

# Entry Points to the implementation
_fast_activation(::typeof(identity), x::AbstractArray) = x

@stable default_mode="disable" function _fast_activation(σ::F, x::AbstractArray) where {F}
    if internal_operation_mode(x) isa LoopedArrayOp
        RT = Core.Compiler._return_type(σ, Tuple{eltype(x)})
        y = similar(x, RT)
        @inbounds @simd ivdep for I in eachindex(y, x)
            y[I] = σ(x[I])
        end
        return y
    end
    return broadcast(σ, x)
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(_fast_activation),
        σ::F, x::AbstractArray{T}) where {F, T}
    return CRC.rrule_via_ad(cfg, broadcast, σ, x)
end

_fast_activation!(::typeof(identity), x::AbstractArray) = x

@stable default_mode="disable" function _fast_activation!(σ::F, x::AbstractArray) where {F}
    if internal_operation_mode(x) isa LoopedArrayOp
        @inbounds @simd ivdep for I in eachindex(x)
            x[I] = σ(x[I])
        end
        return x
    end
    broadcast!(σ, x, x)
    return x
end

# Define rrule for `fast_activation!!`
function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(fast_activation!!),
        σ::F, x::AbstractArray{T}) where {F, T}
    can_setindex(typeof(x)) || return CRC.rrule_via_ad(cfg, _fast_activation, σ, x)

    σ === identity && return x, @closure(Δ->(∂∅, ∂∅, Δ))

    if __no_intermediate_needed(σ, T)
        _fast_activation!(σ, x) # Safe to overwrite x
        proj_x_no_cached = CRC.ProjectTo(x)
        ∇__fast_activation_impl_no_cached = @closure Δ -> begin
            ∂x = __activation_gradient(Δ, x, σ, NotaNumber())
            return ∂∅, ∂∅, proj_x_no_cached(∂x)
        end
        return x, ∇__fast_activation_impl_no_cached
    end

    if __needs_intermediate_but_has_rrule(σ, T)
        y = _fast_activation(σ, x)
        proj_x_cached = CRC.ProjectTo(x)
        ∇__fast_activation_impl_cached_crc = @closure Δ -> begin
            ∂x = __activation_gradient(CRC.unthunk(Δ), y, σ, x)
            return ∂∅, ∂∅, proj_x_cached(∂x)
        end
        return y, ∇__fast_activation_impl_cached_crc
    end

    return CRC.rrule_via_ad(cfg, _fast_activation, σ, x)
end
