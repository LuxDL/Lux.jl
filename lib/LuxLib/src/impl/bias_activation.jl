__reshape_bias_into_xdims(::AbstractArray, ::Nothing) = nothing
__reshape_bias_into_xdims(::AbstractVector, bias::AbstractVector) = bias
function __reshape_bias_into_xdims(
        ::AbstractArray{<:Number, N}, bias::AbstractVector) where {N}
    return reshape(bias, ntuple(i -> ifelse(i == N - 1, length(bias), 1), N))
end

## Needed for type stability
function CRC.rrule(::typeof(__reshape_bias_into_xdims), x::AbstractArray{<:Number, N},
        bias::AbstractVector{<:Number}) where {N}
    bias_r = __reshape_bias_into_xdims(x, bias)
    proj_bias = CRC.ProjectTo(bias)
    return bias_r, Δ -> (∂∅, ∂∅, proj_bias(vec(Δ)))
end

function __generic_bias_activation(
        ::typeof(identity), x::AbstractArray{<:Number}, bias::AbstractVector{<:Number})
    return broadcast(+, x, __reshape_bias_into_xdims(x, bias))
end
__generic_bias_activation(::typeof(identity), x::AbstractArray{<:Number}, ::Nothing) = x
__generic_bias_activation(σ::F, x::AbstractArray{<:Number}, ::Nothing) where {F} = σ.(x)
function __generic_bias_activation(
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    bias_ = __reshape_bias_into_xdims(x, bias)
    return @. σ(x + bias_)
end

# Entry Points to the implementation
## Prevent Ambiguity
__bias_activation_impl(::typeof(identity), x::AbstractVector{<:Number}, ::Nothing) = x
for bType in (Nothing, AbstractVector{<:Number})
    @eval function __bias_activation_impl(
            σ::F, x::AbstractVector{<:Number}, bias::$(bType)) where {F}
        return vec(__bias_activation_impl(σ, reshape(x, :, 1), bias))
    end
end

__bias_activation_impl(::typeof(identity), x::AbstractArray{<:Number}, ::Nothing) = x
function __bias_activation_impl(σ::F, x::AbstractArray{<:Number}, ::Nothing) where {F}
    return _fast_activation(σ, x)
end
@stable default_mode="disable" function __bias_activation_impl(
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    if unrolled_all(fast_scalar_indexing, (x, bias))
        y = similar(x, __get_concrete_fba_output_eltype(σ, x, bias))
        __bias_activation_impl!(y, σ, x, bias)
        return y
    end
    return __generic_bias_activation(σ, x, bias)
end

function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(__bias_activation_impl), σ::F,
        x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    return CRC.rrule_via_ad(cfg, __generic_bias_activation, σ, x, bias)
end

CRC.@opt_out rrule(::typeof(__bias_activation_impl), ::F, ::AbstractVector{<:Number},
    ::Optional{<:AbstractVector{<:Number}}) where {F}

## Prevent Ambiguity
__bias_activation_impl!!(::typeof(identity), x::AbstractVector{<:Number}, ::Nothing) = x
for bType in (Nothing, AbstractVector{<:Number})
    @eval function __bias_activation_impl!!(
            σ::F, x::AbstractVector{<:Number}, bias::$(bType)) where {F}
        return vec(__bias_activation_impl!!(σ, reshape(x, :, 1), bias))
    end
end

__bias_activation_impl!!(::typeof(identity), x::AbstractArray{<:Number}, ::Nothing) = x
function __bias_activation_impl!!(σ::F, x::AbstractArray{<:Number}, ::Nothing) where {F}
    return fast_activation!!(σ, x)
end
@stable default_mode="disable" function __bias_activation_impl!!(
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    can_setindex(x) || return __bias_activation_impl(σ, x, bias)
    __bias_activation_impl!(x, σ, x, bias)
    return x
end

function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(__bias_activation_impl!!), σ::F,
        x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    T = __get_concrete_fba_output_eltype(σ, x, bias)

    if __no_intermediate_needed(σ, T)
        y = __bias_activation_impl!!(σ, x, bias)
        proj_x_no_cached = CRC.ProjectTo(x)
        prob_b_no_cached = CRC.ProjectTo(bias)
        ∇__bias_activation_impl_no_cached = @closure Δ -> begin
            ∂x = __activation_gradient(CRC.unthunk(Δ), y, σ, NotaNumber())
            ∂b = __added_bias_gradient(bias, ∂x)
            return ∂∅, ∂∅, proj_x_no_cached(∂x), prob_b_no_cached(∂b)
        end
        return y, ∇__bias_activation_impl_no_cached
    end

    if __needs_intermediate_but_has_rrule(σ, T)
        y, z = __apply_bias_activation_cached!!(σ, x, bias)
        proj_x_cached = CRC.ProjectTo(x)
        proj_b_cached = CRC.ProjectTo(bias)
        ∇__bias_activation_impl_cached_crc = @closure Δ -> begin
            ∂x = __activation_gradient(CRC.unthunk(Δ), z, σ, y)
            ∂b = __added_bias_gradient(bias, ∂x)
            return ∂∅, ∂∅, proj_x_cached(∂x), proj_b_cached(∂b)
        end
        return y, ∇__bias_activation_impl_cached_crc
    end

    return CRC.rrule_via_ad(cfg, __bias_activation_impl, σ, x, bias)
end

CRC.@opt_out rrule(::typeof(__bias_activation_impl!!), ::F, ::AbstractVector{<:Number},
    ::Optional{<:AbstractVector{<:Number}}) where {F}

## Most functions should never call this outside of this file
function __bias_activation_impl!(
        y::AbstractArray{<:Number, N}, σ::F, x::AbstractArray{<:Number, N},
        bias::AbstractVector{<:Number}) where {F, N}
    opmode = internal_operation_mode((y, x, bias))
    bias_ = __reshape_bias_into_xdims(x, bias)
    if opmode isa LoopedArrayOp
        bc = Broadcast.instantiate(Broadcast.broadcasted(σ ∘ +, x, bias_))
        @simd ivdep for I in eachindex(bc)
            @inbounds y[I] = bc[I]
        end
        return y
    end
    if σ === identity
        broadcast!(+, y, x, bias_)
        return y
    end
    broadcast!(σ ∘ +, y, x, bias_)
    return y
end

# Useful in some of the rrule implementations
function __apply_bias_activation_cached!!(
        σ::F, x, bias::Optional{<:AbstractVector{<:Number}}) where {F}
    @assert σ !== identity
    bias === nothing && return _fast_activation(σ, x), x
    bias_ = __reshape_bias_into_xdims(x, bias)
    if can_setindex(x)
        opmode = internal_operation_mode((x, bias))
        if opmode isa LoopedArrayOp
            bc = Broadcast.instantiate(Broadcast.broadcasted(+, x, bias_))
            @simd ivdep for I in eachindex(bc)
                @inbounds x[I] = bc[I]
            end
            return _fast_activation(σ, x), x
        end
        broadcast!(+, x, x, bias_)
        return _fast_activation(σ, x), x
    end
    y = broadcast(+, x, bias_)
    return _fast_activation(σ, y), y
end
