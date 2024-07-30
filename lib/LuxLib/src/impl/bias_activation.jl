__reshape_bias_into_xdims(::AbstractArray, ::Nothing) = nothing
__reshape_bias_into_xdims(::AbstractVector, bias::AbstractVector) = bias
__reshape_bias_into_xdims(::AbstractVector, bias::StaticVector) = bias
function __reshape_bias_into_xdims(x::AbstractArray, bias::AbstractVector)
    return reshape(bias, ntuple(i -> ifelse(i == ndims(x) - 1, length(bias), 1), ndims(x)))
end
function __reshape_bias_into_xdims(x::AbstractArray, bias::StaticVector)
    return StaticArraysCore.SArray{
        Tuple{ntuple(i -> ifelse(i == ndims(x) - 1, length(bias), 1), ndims(x))...},
        eltype(bias), ndims(x), length(bias)}(bias.data)
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
    if opmode isa LoopedArrayOp
        x_ = reshape(x, :, size(x, N - 1), size(x, N))
        y_ = reshape(y, :, size(y, N - 1), size(y, N))
        @tturbo for K in indices(x_, 3),
            J in indices((x_, bias), (2, 1)),
            I in indices(y_, 1)

            y_[I, J, K] = x_[I, J, K] + bias[J]
        end
        _fast_activation!(σ, y) # NOTE: don't fuse into the above loop
        return y
    end
    bias_ = __reshape_bias_into_xdims(x, bias)
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
    if can_setindex(x)
        opmode = internal_operation_mode((x, bias))
        if opmode isa LoopedArrayOp
            x_ = reshape(x, :, size(x, N - 1), size(x, N))
            @tturbo for K in indices(x_, 3),
                J in indices((x_, bias), (2, 1)),
                I in indices(x_, 1)

                x_[I, J, K] = x_[I, J, K] + bias[J]
            end
            return _fast_activation(σ, x), x
        end
        broadcast!(+, x, x, __reshape_bias_into_xdims(x, bias))
        return _fast_activation(σ, x), x
    end
    y = broadcast(+, x, __reshape_bias_into_xdims(x, bias))
    return _fast_activation(σ, y), y
end
