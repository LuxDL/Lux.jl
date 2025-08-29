module Utils

using Random: Xoshiro
using SpecialFunctions: erf

nfan() = 1, 1 # fan_in, fan_out
nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:(end - 2)]) .* (dims[end - 1], dims[end]) # In case of convolution kernels

norm_cdf(x::T) where {T} = T(0.5) * (1 + T(erf(x / √2))) # erf often doesn't respect the type

safe_type_conversion(::Type{T}, x) where {T} = T(x)
safe_type_conversion(::Type{Complex{T}}, x) where {T} = complex(T(x), T(x))

default_rng() = Xoshiro(1234)

#! format: off
const NAME_TO_DIST = Dict(
    :zeros => "an AbstractArray of zeros",
    :ones  => "an AbstractArray of ones",
    :randn => "random numbers from a standard normal distribution",
    :rand  => "random numbers from a uniform distribution"
)
const NUM_TO_FPOINT = Dict(
    Symbol(16) => Float16,
    Symbol(32) => Float32,
    Symbol(64) => Float64,
    :C16       => ComplexF16,
    :C32       => ComplexF32,
    :C64       => ComplexF64
)
#! format: on

function function_name(fname::String)
    fp = fname[(end - 2):end]
    Symbol(fp) in keys(NUM_TO_FPOINT) && return fname[1:(end - 3)], fp
    return fname[1:(end - 2)], fname[(end - 1):end]
end

function generic_docstring(fname::String)
    funcname, fp = function_name(fname)
    name = NAME_TO_DIST[Symbol(funcname)]
    dist_type = NUM_TO_FPOINT[Symbol(fp)]
    return """
        $fname([::AbstractRNG=Utils.default_rng()], size...;
            kwargs...) -> AbstractArray{$(dist_type), length(size)}

    Return an `AbstractArray{$(dist_type)}` of the given `size` containing $(name).
    """
end

end

module DeviceAgnostic

using Random: AbstractRNG, randn!, rand!

# Different RNG types should override this method
function get_backend_array(::AbstractRNG, ::Type{T}, dims::Integer...) where {T}
    return Array{T}(undef, dims...)
end

ones!(rng, x::AbstractArray{T}) where {T} = fill!(x, one(T))
zeros!(rng, x::AbstractArray{T}) where {T} = fill!(x, zero(T))

# Helpers for device agnostic initializers
for f in (:zeros, :ones, :rand, :randn)
    f! = Symbol(f, "!")
    @eval function $(f)(rng::AbstractRNG, ::Type{T}, dims::Integer...) where {T}
        x = get_backend_array(rng, T, dims...)
        $(f!)(rng, x)
        return x
    end
end

## Certain backends don't support sampling Complex numbers, so we avoid hitting those
## dispatches
for f in (:rand, :randn)
    f! = Symbol(f, "!")
    @eval function $(f)(rng::AbstractRNG, ::Type{Complex{T}}, args::Integer...) where {T}
        x = get_backend_array(rng, Complex{T}, args...)
        $(f!)(rng, reinterpret(T, x))
        return x
    end
end

end
