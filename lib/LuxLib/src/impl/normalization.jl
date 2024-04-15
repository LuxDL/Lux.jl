# Generic Normalization Implementation
function _update_normalization_statistics(
        x::AbstractArray{T1, N}, running_mean::AbstractArray{T2, N},
        running_var::AbstractArray{T3, N}, batchmean::AbstractArray{T4, N},
        batchvar::AbstractArray{T5, N}, momentum::Real,
        ::Val{reduce_dims}) where {N, reduce_dims, T1, T2, T3, T4, T5}
    m = eltype(x)(prod(Base.Fix1(size, x), reduce_dims))
    m_ = m / (m - one(m))
    if last(reduce_dims) != N
        batchmean = mean(batchmean; dims=N)
        batchvar = mean(batchvar; dims=N)
    end
    running_mean = @. (1 - momentum) * running_mean + momentum * batchmean
    running_var = @. (1 - momentum) * running_var + momentum * batchvar * m_
    return (running_mean, running_var)
end

@generated function _get_batch_statistics(
        x::AbstractArray, running_mean::R, running_var::R, r::Val{rdims},
        ::Val{training}, momentum::Union{Real, Nothing}) where {R, rdims, training}
    calls = []
    if !training
        if R == Nothing
            push!(calls, :(batchmean = mean(x; dims=rdims)))
            push!(calls, :(batchvar = var(x; corrected=false, mean=batchmean, dims=rdims)))
        else
            push!(calls, :((batchmean, batchvar) = (running_mean, running_var)))
        end
    else
        push!(calls, :(batchmean = mean(x; dims=rdims)))
        push!(calls, :(batchvar = var(x; corrected=false, mean=batchmean, dims=rdims)))

        if R != Nothing
            push!(calls,
                :(_stats = _update_normalization_statistics(
                    x, running_mean, running_var, batchmean, batchvar, momentum, r)))
            push!(calls, :((running_mean, running_var) = _stats))
        end
    end
    push!(calls, :(return ((batchmean, batchvar), (running_mean, running_var))))
    return Expr(:block, calls...)
end

@generated function _affine_normalize(x::AbstractArray, xmean::ST, xvar::ST,
        scale::A, bias::A, epsilon::Real) where {ST, A}
    if A != Nothing
        return quote
            x_norm = (x .- xmean) ./ sqrt.(xvar .+ epsilon)
            return scale .* x_norm .+ bias
        end
    else
        return :(return (x .- xmean) ./ sqrt.(xvar .+ epsilon))
    end
end

function _normalization_impl(x::AbstractArray, running_mean::R, running_var::R,
        scale::A, bias::A, r::Val{reduce_dims}, training::Val,
        momentum::Union{Real, Nothing}, epsilon::Real) where {R, A, reduce_dims}
    _stats = _get_batch_statistics(x, running_mean, running_var, r, training, momentum)
    (batchmean, batchvar), (running_mean, running_var) = _stats
    x_norm = _affine_normalize(x, batchmean, batchvar, scale, bias, epsilon)
    return (x_norm, running_mean, running_var)
end

function _normalization(x::AbstractArray, running_mean::Union{Nothing, <:AbstractVector},
        running_var::Union{Nothing, <:AbstractVector},
        scale::Union{Nothing, <:AbstractVector},
        bias::Union{Nothing, <:AbstractVector}, reduce_dims::Val,
        training::Val, momentum::Union{Real, Nothing}, epsilon::Real)
    rm_ = _reshape_into_proper_shape(running_mean, x)
    rv_ = _reshape_into_proper_shape(running_var, x)
    s_ = _reshape_into_proper_shape(scale, x)
    b_ = _reshape_into_proper_shape(bias, x)
    x_, rm, rv = _normalization_impl(
        x, rm_, rv_, s_, b_, reduce_dims, training, momentum, epsilon)
    return x_, _vec(rm), _vec(rv)
end
