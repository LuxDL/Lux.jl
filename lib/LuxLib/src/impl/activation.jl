# Used inside rrules
__activation_gradient(Δ, out, ::typeof(identity), x) = Δ
function __activation_gradient(Δ, out, act::F, x) where {F}
    opmode = internal_operation_mode((Δ, out, x))
    if opmode isa LoopedArrayOp  # All sizes are same
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

@stable default_mode="disable" function _fast_activation(σ::F, x::AbstractArray) where {F}
    if internal_operation_mode(x) isa LoopedArrayOp
        σ_sleef = __sleefpirates_activation(σ)
        RT = Core.Compiler._return_type(σ_sleef, Tuple{eltype(x)})
        y = similar(x, RT)
        @simd ivdep for I in eachindex(y, x)
            @inbounds y[I] = σ_sleef(x[I])
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
        σ_sleef = __sleefpirates_activation(σ)
        @simd ivdep for I in eachindex(x)
            @inbounds x[I] = σ_sleef(x[I])
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

# Specialized functions that use SLEEFPirates.jl to speed up the activation functions
sigmoid_fast_sleefpirates(x) = SLEEFPirates.sigmoid_fast(x)
softplus_sleefpirates(x) = SLEEFPirates.softplus(x)
logsigmoid_sleefpirates(x) = -softplus_sleefpirates(-x)
elu_sleefpirates(x, α=1) = SLEEFPirates.Elu(α)(x)
gelu_sleefpirates(x) = SLEEFPirates.gelu(x)
swish_sleefpirates(x) = Base.FastMath.mul_fast(x, sigmoid_fast_sleefpirates(x))
lisht_sleefpirates(x) = Base.FastMath.mul_fast(x, tanh_fast_sleefpirates(x))
tanh_sleefpirates(x) = SLEEFPirates.tanh(x)
tanh_fast_sleefpirates(x) = SLEEFPirates.tanh_fast(x)

# Convert to SLEEFPirates.jl
__sleefpirates_activation(f::F, ::Type{T}) where {F, T} = f
__sleefpirates_activation(f::F, ::Type{Float32}) where {F} = __sleefpirates_activation(f)
__sleefpirates_activation(f::F, ::Type{Float64}) where {F} = __sleefpirates_activation(f)

for (fbase, ffast) in ((NNlib.sigmoid_fast, sigmoid_fast_sleefpirates),
    (NNlib.softplus, softplus_sleefpirates), (NNlib.logsigmoid, logsigmoid_sleefpirates),
    (NNlib.elu, elu_sleefpirates), (NNlib.gelu, gelu_sleefpirates),
    (NNlib.swish, swish_sleefpirates), (NNlib.lisht, lisht_sleefpirates),
    (Base.tanh, tanh_sleefpirates), (NNlib.tanh_fast, tanh_fast_sleefpirates))
    @eval __sleefpirates_activation(::typeof($fbase)) = $ffast
end
__sleefpirates_activation(f::F) where {F} = f
