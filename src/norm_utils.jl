function get_stats(::Val{true}, ::Val{false}, μ, σ², x::AbstractArray{T,N}, reduce_dims, momentum::T) where {T,N}
    # testmode with tracked stats
    stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return reshape(μ, stats_shape), reshape(σ², stats_shape)
end

function get_stats(::Val{false}, active, μ, σ², x, reduce_dims, momentum::T) where {T}
    # trainmode or testmode without tracked stats
    μ = mean(x; dims=reduce_dims)
    return μ, sum(abs2, x .- μ; dims=reduce_dims) ./ mapreduce(i -> size(x, i), *, unique(reduce_dims); init=1)
end

function get_stats(::Val{true}, active::Val{true}, μ, σ², x::AbstractArray{T,N}, reduce_dims, momentum::T) where {T,N}
    # trainmode with tracked stats
    _μ, _σ² = get_stats(Val(false), active, μ, σ², x, reduce_dims, momentum)
    Zygote.ignore() do
        m = prod(size(x)[reduce_dims])  # needed for computing corrected var
        μnew = vec(N ∈ reduce_dims ? _μ : mean(_μ; dims=N))
        σ²new = vec(N ∈ reduce_dims ? _σ² : mean(_σ²; dims=N))
        @. μ = (1 - momentum) * μ + momentum * μnew
        @. σ² = (1 - momentum) * σ² + momentum * (m / (m - one(eltype(σ²)))) * σ²new
        return nothing
    end
    return _μ, _σ²
end

function norm_forward(
    l::ExplicitLayer, ps::NamedTuple, states::NamedTuple, x::AbstractArray{T,N}, reduce_dims, affine_shape
) where {T,N}
    μ, σ² = get_stats(Val(l.track_stats), Val(states.training), states.μ, states.σ², x, reduce_dims, l.momentum)
    if l.affine
        γ = reshape(ps.γ, affine_shape)
        β = reshape(ps.β, affine_shape)
        return @. l.λ(γ * (x - μ) / sqrt(σ² + l.ϵ) + β)
    else
        return @. l.λ((x - μ) / sqrt(σ² + l.ϵ))
    end
end
