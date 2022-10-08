"""
    fast_chunk(x::AbstractArray, ::Val{n}, ::Val{dim})

Type-stable and faster version of `MLUtils.chunk`
"""
@inline fast_chunk(h::Int, n::Int) = (1:h) .+ h * (n - 1)
@inline function fast_chunk(x::AbstractArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return selectdim(x, dim, fast_chunk(h, n))
end
@inline function fast_chunk(x::CuArray, h::Int, n::Int, ::Val{dim}) where {dim}
    # NOTE(@avik-pal): Most CuArray dispatches rely on a contiguous memory layout. Copying
    #                  might be slow but allows us to use the faster and more reliable
    #                  dispatches.
    return copy(selectdim(x, dim, fast_chunk(h, n)))
end
@inline function fast_chunk(x::AbstractArray, ::Val{N}, d::Val{D}) where {N, D}
    return fast_chunk.((x,), size(x, D) ÷ N, 1:N, d)
end

"""
    flatten_spatial(x::AbstractArray{T, 4})

Flattens the first 2 dimensions of `x`, and permutes the remaining dimensions to (2, 1, 3)
"""
@inline function flatten_spatial(x::AbstractArray{T, 4}) where {T}
    return permutedims(reshape(x, (:, size(x, 3), size(x, 4))), (2, 1, 3))
end

"""
    seconddimmean(x)

Computes the mean of `x` along dimension `2`
"""
@inline seconddimmean(x) = dropdims(mean(x; dims=2); dims=2)

"""
    normalise(x::AbstractArray, activation; dims=ndims(x), epsilon=ofeltype(x, 1e-5))

Normalises the array `x` to have a mean of 0 and standard deviation of 1, and applies the
activation function `activation` to the result.
"""
@inline function normalise(x::AbstractArray, ::typeof(identity); dims=ndims(x),
                           epsilon=ofeltype(x, 1e-5))
    xmean = mean(x; dims=dims)
    xstd = std(x; dims=dims, mean=xmean, corrected=false)
    return @. (x - xmean) / (xstd + epsilon)
end

@inline function normalise(x::AbstractArray, activation; dims=ndims(x),
                           epsilon=ofeltype(x, 1e-5))
    xmean = mean(x; dims=dims)
    xstd = std(x; dims=dims, mean=xmean, corrected=false)
    return @. activation((x - xmean) / (xstd + epsilon))
end

# Model construction utilities
function assert_name_present_in(name, possibilities)
    @assert name in possibilities "`name` must be one of $(possibilities)"
end

# TODO(@avik-pal): Starting v0.2 we should be storing only the parameters and some of the
#                  states. Fields like rng don't need to be stored explicitly.
get_pretrained_weights_path(name::Symbol) = get_pretrained_weights_path(string(name))
function get_pretrained_weights_path(name::String)
    try
        return @artifact_str(name)
    catch LoadError
        throw(ArgumentError("No pretrained weights available for `$name`"))
    end
end

function initialize_model(name::Symbol, model; pretrained::Bool=false, rng=nothing, seed=0,
                          kwargs...)
    if pretrained
        path = get_pretrained_weights_path(name)
        ps = load(joinpath(path, "$name.jld2"), "parameters")
        st = load(joinpath(path, "$name.jld2"), "states")
    else
        if rng === nothing
            rng = Random.default_rng()
            Random.seed!(rng, seed)
        end

        ps, st = Lux.setup(rng, model)
    end
    return model, ps, st
end
