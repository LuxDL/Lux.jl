# Utilities
@inline _div_idx(idx, n) = div(idx - 1, n) + 1
@inline _mod_idx(idx, n) = mod(idx - 1, n) + 1

@inline _get_backend(::Nothing) = nothing
@inline function _get_backend(d)
    return hasmethod(KA.get_backend, (typeof(d),)) ? KA.get_backend(d) : nothing
end
@inline _get_backend(t::Tuple) = _get_backend.(t)

function __check_all_same_or_nothing(x::Union{AbstractVector, Tuple})
    @inbounds for i in eachindex(x)
        x[i] === nothing && continue
        for j in (i + 1):length(x)
            x[j] === nothing && continue
            x[i] != x[j] && return false
        end
    end
    return true
end

CRC.@non_differentiable _get_backend(::Any)

@inline _assert_same_backend(args...) = _assert_same_backend([args...])
@inline function _assert_same_backend(xs)
    devs = _get_backend.(xs)
    if !__check_all_same_or_nothing(devs)
        throw(ArgumentError("All arguments must be on the same backend. This error is \
                             encountered if you are calling a function with a mix of CPU \
                             and GPU arrays."))
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

# Meta Programming Utilities
__is_tracked(x) = x == :TrackedArray || x == :TrackedVector
__is_tracked(args...) = any(__is_tracked, args)

# Droping ForwardDiff Gradients
function _drop_forwarddiff_partials end

_drop_forwarddiff_partials(x::AbstractArray) = x
_drop_forwarddiff_partials(::Nothing) = nothing
_drop_forwarddiff_partials(x::Tuple) = _drop_forwarddiff_partials.(x)
function _drop_forwarddiff_partials(x::NamedTuple{N}) where {N}
    return NamedTuple{N}(map(_drop_forwarddiff_partials, values(x)))
end

# Maybe typecast the array
@inline _oftype_array(::Type{T}, x::AbstractArray{T}) where {T} = x
@inline _oftype_array(::Type{T}, x::AbstractArray) where {T} = T.(x)

# Import chain rules to tracker with a syntax similar to ReverseDiff's
# `@grad_from_chainrules`. Needs Tracker.jl to be explicit loaded
macro tracker_grad_from_chainrules(expr)
    return __tracker_grad_from_chainrules(__source__, __module__, expr)
end

function __tracker_grad_from_chainrules end
