# Matrix-Matrix & Matrix-Vector Multiplication
"""
    fast_matmul(A, B)
    fast_matmul!(C, A, B)

Dispatch to Octavian for CPU and CUBLAS for GPU
"""
fast_matmul

@inbounds Base.@pure function fast_matmul(A::AbstractMatrix{T1}, B::AbstractArray{T2,N}) where {T1,T2,N}
    return reshape(fast_matmul(A, reshape(B, size(B, 1), :)), :, size(B)[2:end]...)
end

@inbounds Base.@pure function fast_matmul(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}) where {T1,T2}
    size(A, 2) != size(B, 1) && throw(DimensionMismatch("$(size(A, 2)) != $(size(B, 1)) for Matrix-Matrix Multiply"))
    return fast_matmul!(similar(A, promote_type(T1, T2), (size(A, 1), size(B, 2))), A, B)
end

@inbounds Base.@pure function fast_matmul(A::AbstractMatrix{T1}, b::AbstractVector{T2}) where {T1,T2}
    size(A, 2) != length(b) && throw(DimensionMismatch("$(size(A, 2)) != $(length(b)) for Matrix-Vector Multiply"))
    return fast_matmul!(similar(A, promote_type(T1, T2), size(A, 1)), A, b)
end

fast_matmul!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat) = matmul!(C, A, B)

function fast_matmul!(
    C::CuVecOrMat,
    A::Union{<:CuMatrix{T1},<:Adjoint{T1,<:CuVecOrMat{T1}},<:Transpose{T1,<:CuVecOrMat{T1}}},
    B::Union{<:CuVecOrMat{T2},<:Adjoint{T2,<:CuVecOrMat{T2}},<:Transpose{T2,<:CuVecOrMat{T2}}},
) where {T1,T2}
    return mul!(C, A, B)
end

# Normalization Implementation
function get_stats!(
    ::Val{track_stats}, ::Val{active}, μ, σ², x::AbstractArray{T,N}, reduce_dims, momentum::T
) where {track_stats,active,T,N}
    if track_stats
        if active
            # Training
            μ_batch = mean(x; dims=reduce_dims)
            σ²_batch = std(x; mean=μ_batch, dims=reduce_dims)
            _update_stats!(reduce_dims, N, μ_batch, σ²_batch, momentum, μ, σ²)
            return μ_batch, σ²_batch
        else
            # Testing
            stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
            return reshape(μ, stats_shape), reshape(σ², stats_shape)
        end
    else
        # No Statistics Tracking
        μ = mean(x; dims=reduce_dims)
        return μ, std(x; mean=μ, dims=reduce_dims)
    end
end

function _update_stats!(reduce_dims, N, _μ, _σ², momentum, μ, σ²)
    μnew = vec(N == reduce_dims[end] ? _μ : mean(_μ; dims=N))
    σ²new = vec(N == reduce_dims[end] ? _σ² : mean(_σ²; dims=N))
    @. μ = (1 - momentum) * μ + momentum * μnew
    @. σ² = (1 - momentum) * σ² + momentum * σ²new
    return nothing
end

function norm_forward(
    l::AbstractNormalizationLayer{affine,track_stats},
    ps::NamedTuple,
    states::NamedTuple,
    x::AbstractArray{T,N},
    reduce_dims,
    affine_shape,
) where {T,N,affine,track_stats}
    μ, σ² = get_stats!(
        Val(track_stats),
        Val(states.training == :auto ? istraining() : states.training),
        states.μ,
        states.σ²,
        x,
        reduce_dims,
        l.momentum,
    )
    if affine
        γ = reshape(ps.γ, affine_shape)
        β = reshape(ps.β, affine_shape)
        return @. l.λ(γ * (x - μ) / sqrt(σ² + l.ϵ) + β)
    else
        return @. l.λ((x - μ) / sqrt(σ² + l.ϵ))
    end
end

function get_stats!(
    ::Val{track_stats},
    ::Val{active},
    μ::AbstractArray{T,1},
    σ²::AbstractArray{T,1},
    x::AbstractArray{T,N},
    momentum::T,
    μ_batch::AbstractArray{T,N},
    σ²_batch::AbstractArray{T,N},
    σ²_batch_intermediate::AbstractArray{T,N},
    l::AbstractNormalizationLayer,
) where {track_stats,active,T,N}
    if track_stats
        if active
            # Training
            mean!(μ_batch, x)
            var!(σ²_batch, σ²_batch_intermediate, x, μ_batch)
            _update_stats!!(μ_batch, σ²_batch, momentum, μ, σ², l)
        else
            # Testing
            selectdim(μ_batch, N - 1, :) .= μ
            selectdim(σ²_batch, N - 1, :) .= σ²
        end
    else
        # No Statistics Tracking
        mean!(μ_batch, x)
        var!(σ²_batch, σ²_batch_intermediate, x, μ_batch)
    end
    return μ_batch, σ²_batch
end

function _update_stats!!(μnew, σ²new, momentum, μ, σ², ::BatchNorm)
    @. μ = (1 - momentum) * μ + momentum * μnew
    @. σ² = (1 - momentum) * σ² + momentum * σ²new
    return nothing
end

function norm_forward!(
    y::AbstractArray{T,N},
    μ::AbstractArray{T,N},
    σ²::AbstractArray{T,N},
    σ²_intermediate::AbstractArray{T,N},
    γ::Union{AbstractArray{T,N},Nothing},
    β::Union{AbstractArray{T,N},Nothing},
    l::AbstractNormalizationLayer{affine,track_stats},
    ps::NamedTuple,
    states::NamedTuple,
    x::AbstractArray{T,N},
) where {T,N,affine,track_stats}
    get_stats!(
        Val(track_stats),
        Val(states.training == :auto ? istraining() : states.training),
        states.μ,
        states.σ²,
        x,
        l.momentum,
        μ,
        σ²,
        σ²_intermediate,
        l,
    )
    if affine
        selectdim(γ, N - 1, :) .= ps.γ
        selectdim(β, N - 1, :) .= ps.β
        @. y = l.λ(γ * (x - μ) / sqrt(σ² + l.ϵ) + β)
    else
        @. y = l.λ((x - μ) / sqrt(σ² + l.ϵ))
    end
    return y
end

# Convolution
function fast_conv_bias_act(
    x::AbstractArray{xT,N},
    w::AbstractArray{wT,N},
    cdims::ConvDims,
    b::AbstractArray{bT,N},
    λ=identity;
    kwargs...
) where {xT, wT, bT, N}
    y = similar(x, promote_type(xT, wT, bT), NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x,N))
    return fast_conv_bias_act!(y, x, w, cdims, b, λ, kwargs...)
end

function fast_conv_bias_act!(
    y::AbstractArray{yT,N},
    x::AbstractArray{xT,N},
    w::AbstractArray{wT,N},
    cdims::ConvDims,
    b::AbstractArray{bT,N},
    λ=identity;
    kwargs...
) where {yT, xT, wT, bT, N}
    conv!(y, x, w, cdims)
    if λ == identity
        @. y += b
    else
        @. y = λ(y + b)
    end
    return y
end
