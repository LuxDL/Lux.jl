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
@inline _oftype_array(::Type{T}, ::Nothing) where {T} = nothing

## This part is taken from NNlib.jl
# This just saves typing `only.(only.(` many times:
@inline only_derivative(y, f::F, x) where {F} = only(only(CRC.derivatives_given_output(
    y, f, x)))

# This has no methods, used for testing whether `derivatives_given_output(Ω, f, x)`
# is independent of `x`, as `_return_type` says `Union{}` when calling is an error.
struct NotaNumber <: Real end

# Check no setindexing
@inline __is_immutable_array(x::AbstractArray) = !ArrayInterface.can_setindex(x)
@inline __is_immutable_array(::Nothing) = false
@inline __is_immutable_array_val(x) = Val(__is_immutable_array(x))

CRC.@non_differentiable __is_immutable_array_val(::Any...)

@inline __has_dual(x) = false
@inline __is_immutable_array_or_dual_val(x) = Val(__is_immutable_array(x) || __has_dual(x))

CRC.@non_differentiable __is_immutable_array_or_dual_val(::Any...)

@inline function __expand_conv_bias_dims(
        bias::AbstractVector, ::AbstractArray{T, N}) where {T, N}
    @assert N ≥ 2
    return reshape(bias, (ntuple(Returns(1), N - 2)..., length(bias), 1))
end

@inline function __get_concrete_fba_output_eltype(
        act::F, ::AbstractArray{Tw}, ::AbstractArray{Tx},
        b::Union{Nothing, <:AbstractArray}) where {F, Tw, Tx}
    if b === nothing
        Ty = promote_type(Tw, Tx)
        Tact = Core.Compiler._return_type(act, Tuple{Ty})
        return isconcretetype(Tact) ? promote_type(Ty, Tact) : Ty
    end
    Ty = promote_type(Tw, Tx, eltype(b))
    Tact = Core.Compiler._return_type(act, Tuple{Ty})
    return isconcretetype(Tact) ? promote_type(Ty, Tact) : Ty
end

CRC.@non_differentiable __get_concrete_fba_output_eltype(::Any...)

# Helper to add bias and apply activation function
## This is only meant to be used inside rrules
@inline function __apply_bias_activation!!(
        σ::F, x, bias::Union{Nothing, AbstractArray}, ::Val{cache}) where {F, cache}
    if σ === identity
        bias === nothing && return x
        return __nonuniform_fast_broadcast!(+, x, bias)
    end
    if !cache
        bias === nothing && return __fast_broadcast!(σ, x)
        return __nonuniform_fast_broadcast!(σ ∘ +, x, bias)
    end
    bias === nothing && return __fast_broadcast(σ, x), x
    x = __nonuniform_fast_broadcast!(+, x, bias)
    return __fast_broadcast(σ, x), x
end

@inline function __fast_broadcast(f::F, x, args...) where {F}
    ArrayInterface.fast_scalar_indexing(x) && return @.. f(x, args...)
    return @. f(x, args...)
end
@inline function __fast_broadcast!(f::F, x, args...) where {F}
    if ArrayInterface.fast_scalar_indexing(x)
        if maximum(length, (x, args...)) > 100_000
            @.. thread=true x=f(x, args...)
        else
            @.. x = f(x, args...)
        end
    elseif f === ComposedFunction(sigmoid_fast, +) && length(args) == 1
        y = first(args)
        @. x = sigmoid_fast(x + y) # Has GPU Compilation Problems
    else
        @. x = f(x, args...)
    end
    return x
end
@inline function __nonuniform_fast_broadcast!(f::F, x, args...) where {F}
    if ArrayInterface.fast_scalar_indexing(x)
        if maximum(length, (x, args...)) > 100_000
            bc = Broadcast.instantiate(Broadcast.broadcasted(f, x, args...))
            @batch for I in eachindex(bc)
                @inbounds x[I] = bc[I]
            end
        else
            @. x = f(x, args...)
        end
    elseif f === ComposedFunction(sigmoid_fast, +) && length(args) == 1
        y = first(args)
        @. x = sigmoid_fast(x + y) # Has GPU Compilation Problems
    else
        @. x = f(x, args...)
    end
    return x
end

@inline __apply_bias_activation(σ::F, x, bias::AbstractArray) where {F} = @. σ(x + bias)
@inline __apply_bias_activation(::typeof(identity), x, bias::AbstractArray) = @. x + bias
@inline __apply_bias_activation(σ::F, x, ::Nothing) where {F} = @. σ(x)
@inline __apply_bias_activation(::typeof(identity), x, ::Nothing) = x

@inline __added_bias_gradient(::Nothing, _) = CRC.NoTangent()
@inline function __added_bias_gradient(b::AbstractArray, Δ)
    ∂b = similar(b, promote_type(eltype(b), eltype(Δ)))
    sum!(∂b, Δ)
    return ∂b
end

@inline function __activation_gradient(Δ, out, act::F, x) where {F}
    if ArrayInterface.fast_scalar_indexing(out)
        return @.. Δ * only_derivative(out, act, x)
    end
    return @. Δ * only_derivative(out, act, x)
end

# Reduce BLAS threads if we are going to use a native Julia implementation
@inline function __maybe_reduce_BLAS_threads(x::AbstractArray)::Int
    if ArrayInterface.fast_scalar_indexing(x)
        old_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        return old_threads
    end
    return -1
end

CRC.@non_differentiable __maybe_reduce_BLAS_threads(::AbstractArray)

@inline function __reset_BLAS_threads(old_threads::Int)
    old_threads ≥ 1 && BLAS.set_num_threads(old_threads)
    return nothing
end

CRC.@non_differentiable __reset_BLAS_threads(::Int)

# Defined in ext/LuxLibCUDAExt.jl
function _cublaslt_matmul_fused! end

@inline __materialize_subarray(x::AbstractArray) = x
@inline __materialize_subarray(x::SubArray) = copy(x)
