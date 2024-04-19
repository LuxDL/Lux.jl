# Specialized Implementation based off NNlib._fast_broadcast with added logic from
# ArrayInterface
# If we enter here, we already know that we can setindex into the array
@inline function __fast_activation_impl!(σ::F, x::AbstractArray) where {F}
    if ArrayInterface.fast_scalar_indexing(x)
        bc = Broadcast.instantiate(Broadcast.broadcasted(σ, x))
        @simd ivdep for I in eachindex(bc)
            @inbounds x[I] = bc[I]
        end
    else
        @. x = σ(x)
    end
    return x
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__fast_activation_impl!), σ::F, x::AbstractArray{T}) where {F, T}
    σ === identity && return x, @closure(Δ->(CRC.NoTangent(), CRC.NoTangent(), Δ))

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        __fast_activation_impl!(σ, x)
        ∇__fast_activation_impl_no_cached = @closure Δ -> begin
            ∂x = only_derivative.(x, σ, NotaNumber()) .* CRC.unthunk(Δ)
            return CRC.NoTangent(), CRC.NoTangent(), ∂x
        end
        return x, ∇__fast_activation_impl_no_cached
    end

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        y = @. σ(x)
        ∇__fast_activation_impl_cached_crc = @closure Δ -> begin
            ∂z = only_derivative.(y, σ, x) .* CRC.unthunk(Δ)
            return CRC.NoTangent(), CRC.NoTangent(), ∂z
        end
        return z, ∇__fast_activation_impl_cached_crc
    end

    y, pb_f = CRC.rrule_via_ad(cfg, broadcast, σ, x)
    ∇__fast_activation_impl_cached = @closure Δ -> begin
        _, _, ∂x = pb_f(Δ)
        return CRC.NoTangent(), CRC.NoTangent(), ∂x
    end
    return y, ∇__fast_activation_impl_cached
end
