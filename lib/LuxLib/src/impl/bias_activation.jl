__resize_bias_into_xdims(::AbstractArray, ::Nothing) = nothing
__resize_bias_into_xdims(::AbstractVector, bias::AbstractVector) = bias
function __resize_bias_into_xdims(
        ::AbstractArray{<:Number, N}, bias::AbstractVector) where {N}
    return reshape(bias, ntuple(i -> i == N - 1 ? length(bias) : 1, N))
end

function __generic_bias_activation(
        ::typeof(identity), x::AbstractArray, bias::AbstractVector)
    return broadcast(+, x, bias)
end
function __generic_bias_activation(
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector) where {F, N}
    bias_ = __resize_bias_into_xdims(x, bias)
    # TODO: Call broadcast(σ ∘ +, x, bias) once https://github.com/FluxML/NNlib.jl/pull/597 lands
    return @. σ(x + bias_)
end

# Entry Points to the implementation
function __bias_activation_impl(
        σ::F, x::AbstractVector, bias::Optional{<:AbstractVector}) where {F}
    return vec(__bias_activation_impl(σ, reshape(x, :, 1), bias))
end

__bias_activation_impl(::typeof(identity), x::AbstractArray, ::Nothing) = x
__bias_activation_impl(σ::F, x::AbstractArray, ::Nothing) where {F} = _fast_activation(σ, x)
@stable default_mode="warn" function __bias_activation_impl(
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector) where {F, N}
    if unrolled_all(fast_scalar_indexing, (x, bias))
        y = similar(x, __get_concrete_fba_output_eltype(σ, x, bias))
        __bias_activation_impl!(y, σ, x, bias)
        return y
    end
    return __generic_bias_activation(σ, x, bias)
end

@stable default_mode="warn" function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(__bias_activation_impl),
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector) where {F, N}
    return CRC.rrule_via_ad(cfg, __generic_bias_activation, σ, x, bias)
end

CRC.@opt_out rrule(::typeof(__bias_activation_impl), ::F, ::AbstractVector,
    ::Optional{<:AbstractVector}) where {F}

function __bias_activation_impl!!(
        σ::F, x::AbstractVector, bias::Optional{<:AbstractVector}) where {F}
    return vec(__bias_activation_impl!!(σ, reshape(x, :, 1), bias))
end

__bias_activation_impl!!(::typeof(identity), x::AbstractArray, ::Nothing) = x
function __bias_activation_impl!!(σ::F, x::AbstractArray, ::Nothing) where {F}
    return fast_activation!!(σ, x)
end
@stable default_mode="warn" function __bias_activation_impl!!(
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector) where {F, N}
    can_setindex(x) || return __bias_activation_impl(σ, x, bias)
    __bias_activation_impl!(x, σ, x, bias)
    return x
end

@stable default_mode="warn" function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(__bias_activation_impl!!),
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector) where {F, N}
    T = __get_concrete_fba_output_eltype(σ, x, bias)

    if __no_intermediate_needed(σ, T)
        y = __bias_activation_impl!!(σ, x, bias)
        proj_x_no_cached = CRC.ProjectTo(x)
        prob_b_no_cached = CRC.ProjectTo(bias)
        ∇__bias_activation_impl_no_cached = @closure Δ -> begin
            ∂x = __activation_gradient(CRC.unthunk(Δ), y, σ, NotaNumber())
            ∂b = __added_bias_gradient(bias, ∂x)
            return NoTangent(), NoTangent(), proj_x_no_cached(∂x), prob_b_no_cached(∂b)
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
            return NoTangent(), NoTangent(), proj_x_cached(∂x), proj_b_cached(∂b)
        end
        return y, ∇__bias_activation_impl_cached_crc
    end

    return CRC.rrule_via_ad(cfg, __bias_activation_impl, σ, x, bias)
end

CRC.@opt_out rrule(::typeof(__bias_activation_impl!!), ::F,
    ::AbstractVector, ::Optional{<:AbstractVector}) where {F}

## Most functions should never call this outside of this file
function __bias_activation_impl!(y::AbstractArray{<:Number, N}, σ::F,
        x::AbstractArray{<:Number, N}, bias::AbstractVector) where {F, N}
    if unrolled_all(fast_scalar_indexing, (x, bias))
        __bias_activation_impl_loop!(y, σ, x, bias)
        return y
    end
    bias_ = __resize_bias_into_xdims(x, bias)
    if σ === identity
        broadcast!(+, y, x, bias_)
        return y
    end
    # TODO: Call broadcast!(σ ∘ +, y, x, bias) once https://github.com/FluxML/NNlib.jl/pull/597 lands
    @. y = σ(x + bias_)
    return y
end
function __bias_activation_impl_loop!(y::AbstractArray{<:Number, N}, σ::F,
        x::AbstractArray{<:Number, N}, bias::AbstractVector) where {F, N}
    sz_fn = Base.Fix1(size, x)
    x̃_dims = (prod(sz_fn, 1:(N - 2); init=1), sz_fn(N - 1), sz_fn(N))
    x̃ = reshape(x, x̃_dims)
    if σ === identity
        ỹ = reshape(y, x̃_dims)
        @simd ivdep for j in axes(ỹ, 2)
            for i in axes(ỹ, 1), k in axes(ỹ, 3)
                @inbounds ỹ[i, j, k] = x̃[i, k, j] + bias[j]
            end
        end
    else
        ỹ = reshape(y, x̃_dims)
        @simd ivdep for j in axes(ỹ, 2)
            for i in axes(ỹ, 1), k in axes(ỹ, 3)
                @inbounds ỹ[i, j, k] = σ(x̃[i, k, j] + bias[j])
            end
        end
    end
end

# Useful in some of the rrule implementations
function __apply_bias_activation_cached!!(
        σ::F, x, bias::Optional{<:AbstractVector}) where {F}
    @assert σ !== identity
    bias === nothing && return _fast_activation(σ, x), x
    if can_setindex(x)
        if unrolled_all(fast_scalar_indexing, (x, bias))
            __bias_activation_impl_loop!(x, identity, x, bias)
            return _fast_activation(σ, x), x
        end
        bias_ = __resize_bias_into_xdims(x, bias)
        broadcast!(+, x, x, bias_)
        return _fast_activation(σ, x), x
    end
    y = broadcast(+, x, __resize_bias_into_xdims(x, bias))
    return _fast_activation(σ, y), y
end
