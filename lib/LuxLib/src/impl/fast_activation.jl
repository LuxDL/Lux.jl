# Specialized Implementation based off NNlib._fast_broadcast with added logic from
# ArrayInterface
# If we enter here, we already know that we can setindex into the array
@stable default_mode="warn" function __fast_activation_impl!!(
        σ::F, x::AbstractArray) where {F}
    return __fast_broadcast!(σ, x)
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode},
        ::typeof(__fast_activation_impl!!), σ::F, x::AbstractArray{T}) where {F, T}
    σ === identity && return x, @closure(Δ->(NoTangent(), NoTangent(), Δ))

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        x = __fast_activation_impl!!(σ, x)
        ∇__fast_activation_impl_no_cached = @closure Δ -> begin
            ∂x = __activation_gradient(Δ, x, σ, NotaNumber())
            return NoTangent(), NoTangent(), ∂x
        end
        return x, ∇__fast_activation_impl_no_cached
    end

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        y = __fast_broadcast(σ, x)
        ∇__fast_activation_impl_cached_crc = @closure Δ -> begin
            ∂y = __activation_gradient(CRC.unthunk(Δ), y, σ, x)
            return NoTangent(), NoTangent(), ∂y
        end
        return y, ∇__fast_activation_impl_cached_crc
    end

    return CRC.rrule_via_ad(cfg, broadcast, σ, x)
end
