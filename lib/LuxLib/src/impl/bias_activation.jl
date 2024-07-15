# Helper to add bias and apply activation function
## This is only meant to be used inside rrules
function __apply_bias_activation!!(
        σ::F, x, bias::Optional{<:AbstractArray}, ::Val{cache}) where {F, cache}
    if σ === identity
        bias === nothing && return x
        return __nonuniform_fast_broadcast!(+, x, bias)
    end
    if !cache
        bias === nothing && return __fast_broadcast!(σ, x)
        return __nonuniform_fast_broadcast!(σ ∘ +, x, bias)
    end
    bias === nothing && return __fast_broadcast(σ, x), x
    x = __nonuniform_fast_broadcast!(+, x, bias)
    return __fast_broadcast(σ, x), x
end

function __fast_broadcast(f::F, x, args...) where {F}
    fast_scalar_indexing(x) && return @.. f(x, args...)
    return @. f(x, args...)
end
function __fast_broadcast!(f::F, x, args...) where {F}
    if fast_scalar_indexing(x)
        @.. x = f(x, args...)
    elseif __fails_inplace_bcast_gpu(f) && length(args) == 1
        y = first(args)
        @. x = f.outer(f.inner(x, y))
    else
        @. x = f(x, args...)
    end
    return x
end
function __nonuniform_fast_broadcast!(f::F, x, args...) where {F}
    if fast_scalar_indexing(x)
        if maximum(length, (x, args...)) > 100_000
            bc = Broadcast.instantiate(Broadcast.broadcasted(f, x, args...))
            @simd ivdep for I in eachindex(bc)
                @inbounds x[I] = bc[I]
            end
        else
            @. x = f(x, args...)
        end
    elseif __fails_inplace_bcast_gpu(f) && length(args) == 1
        y = first(args)
        @. x = f.outer(f.inner(x, y))
    else
        @. x = f(x, args...)
    end
    return x
end

__fails_inplace_bcast_gpu(::ComposedFunction{typeof(sigmoid_fast), typeof(+)}) = true
__fails_inplace_bcast_gpu(::ComposedFunction{typeof(swish), typeof(+)}) = true
__fails_inplace_bcast_gpu(::F) where {F} = false

__apply_bias_activation(σ::F, x, bias::AbstractArray) where {F} = @. σ(x + bias)
__apply_bias_activation(::typeof(identity), x, bias::AbstractArray) = @. x + bias
__apply_bias_activation(σ::F, x, ::Nothing) where {F} = @. σ(x)
__apply_bias_activation(::typeof(identity), x, ::Nothing) = x

__added_bias_gradient(::Nothing, _) = NoTangent()
function __added_bias_gradient(b::AbstractArray, Δ)
    ∂b = similar(b, promote_type(eltype(b), eltype(Δ)))
    sum!(∂b, Δ)
    return ∂b
end

function __activation_gradient(Δ, out, act::F, x) where {F}
    if fast_scalar_indexing(out)
        return @.. Δ * only_derivative(out, act, x)
    end
    return @. Δ * only_derivative(out, act, x)
end

function __activation_gradient_simple(Δ, out, act::F, x) where {F}
    return @. Δ * only_derivative(out, act, x)
end

# Needed for reverse over reverse mode AD
function CRC.rrule(cfg::RuleConfig{>:HasReverseMode},
        ::typeof(__activation_gradient), Δ, out, act::F, x) where {F}
    return CRC.rrule_via_ad(cfg, __activation_gradient_simple, Δ, out, act, x)
end
