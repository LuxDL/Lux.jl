"""
    fast_chunk(x::AbstractArray, ::Val{n}, ::Val{dim})

Type-stable and faster version of `MLUtils.chunk`
"""
@inline fast_chunk(h::Int, n::Int) = (1:h) .+ h * (n - 1)
@inline function fast_chunk(x::AbstractArray, h::Int, n::Int, ::Val{dim}) where {dim}
    selectdim(x, dim, fast_chunk(h, n))
end
@inline function fast_chunk(x::AbstractArray, ::Val{N}, d::Val{D}) where {N, D}
    fast_chunk.((x,), size(x, D) ÷ N, 1:N, d)
end

@inline function flatten_spatial(x::AbstractArray{T, 4}) where {T}
    return permutedims(reshape(x, (:, size(x, 3), size(x, 4))), (2, 1, 3))
end

@inline seconddimmean(x) = dropdims(mean(x, dims=2); dims=2)

@inline function normalise(x::AbstractArray, ::typeof(identity); dims=ndims(x),
                           epsilon=ofeltype(x, 1e-5))
    xmean = mean(x, dims=dims)
    xstd = std(x, dims=dims, mean=xmean, corrected=false)
    return @. (x - xmean) / (xstd + epsilon)
end

@inline function normalise(x::AbstractArray, activation; dims=ndims(x),
                           epsilon=ofeltype(x, 1e-5))
    xmean = mean(x, dims=dims)
    xstd = std(x, dims=dims, mean=xmean, corrected=false)
    return @. activation((x - xmean) / (xstd + epsilon))
end
