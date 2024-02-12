module WeightInitializersCUDAExt

using WeightInitializers, CUDA
import WeightInitializers: __partial_apply, NUM_TO_FPOINT, identity_init, sparse_init, orthogonal

const AbstractCuRNG = Union{CUDA.RNG, CURAND.RNG}

for T in ("16", "32", "64", "C16", "C32", "C64"), fname in (:ones, :zeros)
    name = Symbol(fname, T)
    TP = NUM_TO_FPOINT[Symbol(T)]
    @eval begin
        function WeightInitializers.$(name)(rng::AbstractCuRNG, dims::Integer...; kwargs...)
            return CUDA.$(fname)($TP, dims...; kwargs...)
        end
    end

    @eval function WeightInitializers.$(name)(rng::AbstractCuRNG; kwargs...)
        return __partial_apply($name, (rng, (; kwargs...)))
    end
end

function orthogonal(rng::AbstractCuRNG, ::Type{T}, dims::Integer...;
        gain::Number=T(1.0)) where {T <: Number}
   @assert length(dims)>1 "Creating vectors (length(dims) == 1) is not allowed"

    if length(dims) == 2
        rows, cols = dims
    else
        rows = prod(dims[1:(end - 1)])
        cols = dims[end]
    end

    if rows < cols
        return CUDA.permutedims(orthogonal(rng, T, cols, rows; gain))
    end

    mat = randn(rng, T, rows, cols)
    Q, R = CUDA.qr(mat)
    mat .= Q * sign.(CUDA.diag(R)) .* T(gain)

    if length(dims) > 2
        return CUDA.reshape(mat, dims)
    else
        return mat
    end
end


function identity_init(rng::AbstractCuRNG, ::Type{T}, dims::Integer...;
        gain::Number=1, shift::Integer=0) where {T <: Number}
    if length(dims) == 1
        # Bias initialization
        return CUDA.zeros(T, dims...)
    elseif length(dims) == 2
        # Matrix multiplication
        rows, cols = dims
        mat = CUDA.zeros(T, rows, cols)
        diag_indices = 1:min(rows, cols)
        CUDA.fill!(view(mat, diag_indices, diag_indices), gain)
        return CUDA.circshift(mat, shift)
    else
        # Convolution or more dimensions
        nin, nout = dims[end - 1], dims[end]
        centers = map(d -> cld(d, 2), dims[1:(end - 2)])
        weights = CUDA.zeros(T, dims...)
        #we should really find a better way to do this
        CUDA.@allowscalar for i in 1:min(nin, nout)
            index = (centers..., i, i)
            weights[index...] = gain
        end
        return CUDA.circshift(weights, (ntuple(d -> 0, length(dims) - 2)..., shift, shift))
    end
end

function sparse_init(rng::AbstractCuRNG, ::Type{T}, dims::Integer...;
        sparsity::Number, std::Number=T(0.01)) where {T <: Number}
    if length(dims) != 2
        throw(ArgumentError("Only 2-dimensional outputs are supported for sparse initialization."))
    end

    rows, cols = dims
    prop_zero = min(1.0, sparsity)
    num_zeros = ceil(Integer, prop_zero * rows)
    sparse_array = randn(rng, T, dims...) .* std
    sparse_array[1:num_zeros, :] .= CUDA.zero(T)

    for col in 1:cols
        sparse_array[:, col] = CUDA.shuffle(rng, sparse_array[:, col])
    end

    return sparse_array
end

for initializer in (:sparse_init, :identity_init)
    @eval function ($initializer)(rng::AbstractCuRNG, dims::Integer...; kwargs...)
        return $initializer(rng, Float32, dims...; kwargs...)
    end

    @eval function ($initializer)(rng::AbstractCuRNG; kwargs...)
        return __partial_apply($initializer, (rng, (; kwargs...)))
    end
    @eval function ($initializer)(rng::AbstractCuRNG,
            ::Type{T}; kwargs...) where {T <: Number}
        return __partial_apply($initializer, ((rng, T), (; kwargs...)))
    end
end

end
