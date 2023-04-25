# Shorthand Types
const AA = AbstractArray
const AV = AbstractVector
const NOrAVR = Union{Nothing, AbstractVector{<:Real}}
const FP_32_64 = Union{Float32, Float64}
const ∂∅ = NoTangent()

# Utilities
_div_idx(idx, n) = div(idx - 1, n) + 1
_mod_idx(idx, n) = mod(idx - 1, n) + 1

_get_backend(::Nothing) = nothing
function _get_backend(d)
    return hasmethod(KA.get_backend, (typeof(d),)) ? KA.get_backend(d) : nothing
end
_get_backend(t::Tuple) = _get_backend.(t)

function __check_all_same_or_nothing(x::Union{AbstractVector, Tuple})
    for i in 1:length(x)
        x[i] === nothing && continue
        for j in (i + 1):length(x)
            x[j] === nothing && continue
            x[i] != x[j] && return false
        end
    end
    return true
end

CRC.@non_differentiable _get_backend(::Any)

_assert_same_backend(args...) = _assert_same_backend([args...])
function _assert_same_backend(xs)
    devs = _get_backend.(xs)
    if !__check_all_same_or_nothing(devs)
        throw(ArgumentError("All arguments must be on the same backend. This error is encountered if you are calling a function with a mix of CPU and GPU arrays."))
    end
    return
end

CRC.@non_differentiable _assert_same_backend(::Any...)

@inline @generated _vec(x::T) where {T} = hasmethod(vec, (T,)) ? :(vec(x)) : :x

@inline @inbounds function _get_reshape_dims(sx::NTuple{N, <:Int}, ly::Int) where {N}
    if ly == sx[N - 1]
        return ntuple(i -> i == N - 1 ? ly : 1, N)
    elseif N > 2 && ly == sx[N - 1] * sx[N - 2]
        return ntuple(i -> i == (N - 1) || i == (N - 2) ? sx[i] : 1, N)
    else
        throw(ArgumentError("Invalid Dimensions!"))
    end
end

CRC.@non_differentiable _get_reshape_dims(::Any...)

@inline _reshape_into_proper_shape(::Nothing, y) = nothing
@inline _reshape_into_proper_shape(x, y) = reshape(x, _get_reshape_dims(size(y), length(x)))

# Copy and don't allow gradient propagation
_copy_autodiff_barrier(x) = copy(x)
_copy_autodiff_barrier(::Nothing) = nothing

CRC.@non_differentiable _copy_autodiff_barrier(::Any)

_replicate(rng::AbstractRNG) = copy(rng)

CRC.@non_differentiable _replicate(::Any)

# Var Implementation
## Using the default version from Statistics causes issues with Tracker.jl
function _var(x, ::Val{corrected}, _mean, ::Val{dims}) where {corrected, dims}
    return sum((x .- _mean) .^ 2; dims) ./ (prod(Base.Fix1(size, x), dims) - corrected)
end

# Meta Programming Utilities
__is_tracked(x) = x == :TrackedArray || x == :TrackedVector
__is_tracked(args...) = any(__is_tracked, args)
