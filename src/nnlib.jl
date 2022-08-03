# TODO(@avik-pal): Eventually we want to move all these functions and their adjoints to NNlib.jl

# Normalization Implementation
@inline function update_statistics(x::AbstractArray{T, N},
                                   running_mean::AbstractArray{T, N},
                                   running_var::AbstractArray{T, N},
                                   batchmean::AbstractArray{T, N},
                                   batchvar::AbstractArray{T, N}, momentum::T,
                                   reduce_dims) where {T, N}
    sx = size(x)
    m = T(prod((sx[i] for i in reduce_dims)))
    if reduce_dims[end] != N
        batchmean = mean(batchmean; dims=N)
        batchvar = mean(batchvar; dims=N)
    end
    running_mean = @. (1 - momentum) * running_mean + momentum * batchmean
    running_var = @. (1 - momentum) * running_var + momentum * batchvar * (m / (m - one(m)))
    return (running_mean, running_var)
end

"""
    normalization(x, running_mean, running_var, scale, bias, activation, reduce_dims,
                  ::Val{training}, momentum, epsilon)

Performs BatchNorm/GroupNorm/InstanceNorm based on input configuration

!!! note
    
    Detailed docs are WIP
"""
@inline function normalization(x::AbstractArray{T, N},
                               running_mean::Union{Nothing, AbstractVector{T}},
                               running_var::Union{Nothing, AbstractVector{T}},
                               scale::Union{Nothing, AbstractVector{T}},
                               bias::Union{Nothing, AbstractVector{T}}, activation,
                               reduce_dims, t::Val, momentum::T=T(0.1),
                               epsilon::T=T(1e-5)) where {T, N}
    running_mean_reshaped = _reshape_into_proper_shape(running_mean, x)
    running_var_reshaped = _reshape_into_proper_shape(running_var, x)
    scale_reshaped = _reshape_into_proper_shape(scale, x)
    bias_reshaped = _reshape_into_proper_shape(bias, x)
    x_norm, running_mean_, running_var_ = normalization_forward(x, running_mean_reshaped,
                                                                running_var_reshaped,
                                                                scale_reshaped,
                                                                bias_reshaped, activation,
                                                                reduce_dims, t, momentum,
                                                                epsilon)
    return x_norm, _safe_vec(running_mean_), _safe_vec(running_var_)
end

@generated function normalization_forward(x::AbstractArray{T, N}, running_mean::RM,
                                          running_var::RV, scale::S, bias::B, activation::A,
                                          reduce_dims, ::Val{training},
                                          momentum::T=T(0.1f0),
                                          epsilon::T=T(1.0f-5)) where {RM, RV, S, B, T, N,
                                                                       A, training}
    calls = []
    if !training
        if RM == Nothing
            expr = :(batchmean = mean(x; dims=reduce_dims);
                     batchvar = var(x; mean=batchmean, dims=reduce_dims, corrected=false))
        else
            expr = :(batchmean = running_mean;
                     batchvar = running_var)
        end
        push!(calls, expr)
    else
        expr = :(batchmean = mean(x; dims=reduce_dims);
                 batchvar = var(x; mean=batchmean, dims=reduce_dims, corrected=false))
        push!(calls, expr)

        if RM != Nothing
            push!(calls,
                  :((running_mean, running_var) = update_statistics(x, running_mean,
                                                                    running_var, batchmean,
                                                                    batchvar, momentum,
                                                                    reduce_dims)))
        end
    end

    expr = if S != Nothing
        if A == typeof(identity)
            :(result = scale .* (x .- batchmean) ./ sqrt.(batchvar .+ epsilon) .+ bias)
        else
            :(result = activation.(scale .* (x .- batchmean) ./
                                   sqrt.(batchvar .+ epsilon) .+ bias))
        end
    else
        if A == typeof(identity)
            :(result = (x .- batchmean) ./ sqrt.(batchvar .+ epsilon))
        else
            :(result = activation.((x .- batchmean) ./ sqrt.(batchvar .+ epsilon)))
        end
    end
    push!(calls, expr)
    push!(calls, :(return (result, running_mean, running_var)))

    return Expr(:block, calls...)
end

# Convolution
@inline conv_wrapper(x, weight, cdims) = conv(x, weight, cdims)

@inline function conv_wrapper(x::SubArray{T, N, <:CuArray}, weight, cdims) where {T, N}
    return conv(copy(x), weight, cdims)
end

# Dropout
@inline _dropout_shape(s, ::Colon) = size(s)
@inline function _dropout_shape(s, dims)
    return tuple((i ∉ dims ? 1 : si for (i, si) in enumerate(size(s)))...)
end

@inline _dropout_kernel(y::T, p, q) where {T} = y > p ? q : zero(T)

@inline function generate_dropout_mask(rng::AbstractRNG, x, p, q; dims=:)
    realfptype = float(real(eltype(x)))
    y = rand!(rng, similar(x, realfptype, _dropout_shape(x, dims)))
    y .= _dropout_kernel.(y, p, q)
    return y
end

"""
    dropout(rng::AbstractRNG, x, p, q, dims, ::Val{training})
    dropout(rng::AbstractRNG, x, mask, p, q, dims, t::Val{training}, ::Val{update_mask})

If `training` then dropout is applied on `x` with probability `p` along `dims`. If `mask` is
passed it is used if `update_mask` is false. If `update_mask` is true then the mask is
generated and used.
"""
@inline @generated function dropout(rng::AbstractRNG, x, p, q, dims,
                                    ::Val{training}) where {training}
    if training
        return :(rng = replicate(rng);
                 mask = generate_dropout_mask(rng, x, p, q; dims);
                 return (x .* ignore_derivatives(mask), mask, rng))
    else
        return :(return (x, x, rng))
    end
end

@inline @generated function dropout(rng::AbstractRNG, x, mask, p, q, dims, t::Val{training},
                                    ::Val{update_mask}) where {training, update_mask}
    if update_mask
        return :((y, mask, rng) = dropout(rng, x, p, q, dims, t);
                 return (y, mask, rng, Val(false)))
    else
        if training
            return :(size(x, ndims(x)) != size(mask, ndims(x)) &&
                         return (dropout(rng, x, p, q, dims, t)..., Val(false));
                     return (x .* ignore_derivatives(mask), mask, rng, Val(false)))
        else
            return :(return (x, mask, rng, Val(false)))
        end
    end
end

# Adaptive Pooling
@inline function compute_adaptive_pooling_dims(x::AbstractArray, outsize)
    insize = size(x)[1:(end - 2)]
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    pad = 0
    return PoolDims(x, k; padding=pad, stride=stride)
end
