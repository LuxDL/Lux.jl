"""
    _fast_chunk(x::AbstractArray, ::Val{n}, ::Val{dim})

Type-stable and faster version of `MLUtils.chunk`
"""
@inline _fast_chunk(h::Int, n::Int) = (1:h) .+ h * (n - 1)
@inline function _fast_chunk(x::AbstractArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return selectdim(x, dim, _fast_chunk(h, n))
end
# NOTE(@avik-pal): Most CuArray dispatches rely on a contiguous memory layout. Copying
#                  might be slow but allows us to use the faster and more reliable
#                  dispatches.
@inline function _fast_chunk(x::CuArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return copy(selectdim(x, dim, _fast_chunk(h, n)))
end
@inline function _fast_chunk(x::AbstractArray, ::Val{N}, d::Val{D}) where {N, D}
    return _fast_chunk.((x,), size(x, D) ÷ N, 1:N, d)
end

"""
    _flatten_spatial(x::AbstractArray{T, 4})

Flattens the first 2 dimensions of `x`, and permutes the remaining dimensions to (2, 1, 3)
"""
@inline function _flatten_spatial(x::AbstractArray{T, 4}) where {T}
    return permutedims(reshape(x, (:, size(x, 3), size(x, 4))), (2, 1, 3))
end

"""
    _seconddimmean(x)

Computes the mean of `x` along dimension `2`
"""
@inline _seconddimmean(x) = dropdims(mean(x; dims=2); dims=2)

# Model construction utilities
function assert_name_present_in(name, possibilities)
    @assert name in possibilities "`name` must be one of $(possibilities)"
end

# TODO(@avik-pal): Starting v0.2 we should be storing only the parameters and some of the
#                  states. Fields like rng don't need to be stored explicitly.
_get_pretrained_weights_path(name::Symbol) = _get_pretrained_weights_path(string(name))
function _get_pretrained_weights_path(name::String)
    try
        return @artifact_str(name)
    catch LoadError
        throw(ArgumentError("no pretrained weights available for `$name`"))
    end
end

function _initialize_model(name::Symbol, model; pretrained::Bool=false, rng=nothing, seed=0,
                           kwargs...)
    if pretrained
        path = _get_pretrained_weights_path(name)
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
