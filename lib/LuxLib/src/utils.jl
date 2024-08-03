const Optional{T} = Union{Nothing, T}
const Numeric = Union{AbstractArray{<:T}, T} where {T <: Number}
const ∂∅ = NoTangent()

# Bias Gradient -- can't be used inside gradient rules
__added_bias_gradient(::Nothing, Δ::AbstractArray) = ∂∅
function __added_bias_gradient(
        b::AbstractArray{<:Number, N}, Δ::AbstractArray{<:Number, N}) where {N}
    return __reduce_sum(b, Δ)
end
function __added_bias_gradient(b::AbstractVector{<:Number}, Δ::AbstractArray{<:Number})
    b_ = __reshape_bias_into_xdims(Δ, b)
    return vec(__reduce_sum(b_, Δ))
end

# Operations that most AD won't be able to differentiate
__reduce_sum(::Nothing, ::NoTangent) = ∂∅
function __reduce_sum(x::AbstractArray, y::AbstractArray)
    z = similar(x, promote_type(eltype(x), eltype(y)))
    sum!(z, y)
    return z
end

# Simple Operations -- no rrules needed
@generated _vec(x::T) where {T} = hasmethod(vec, (T,)) ? :(vec(x)) : :x

## Maybe typecast the array
_ofeltype_array(::Type{T}, x::AbstractArray{T}) where {T} = x
_ofeltype_array(::Type{T}, x::AbstractArray) where {T} = convert(AbstractArray{T}, x)
_ofeltype_array(::Type{T}, ::Nothing) where {T} = nothing

__materialize_subarray(x::AbstractArray) = x
__materialize_subarray(x::SubArray) = copy(x)

remove_tracking(x::Number) = x
remove_tracking(x::AbstractArray) = x
remove_tracking(::Type{T}) where {T <: Number} = T
remove_tracking(x::ForwardDiff.Dual) = ForwardDiff.value(x)
remove_tracking(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)
remove_tracking(::Type{<:ForwardDiff.Dual{Tag, T}}) where {Tag, T} = remove_tracking(T)
remove_tracking(::Nothing) = nothing

__reshape(x::AbstractArray, dims...) = reshape(x, dims)
__reshape(::Nothing, dims...) = nothing

# Non-differentiable functions
## Reduce BLAS threads if we are going to use a native Julia implementation
function __maybe_reduce_BLAS_threads(x::AbstractArray)
    __maybe_reduce_BLAS_threads(get_device_type(x))
end
__maybe_reduce_BLAS_threads(::Type{T}) where {T} = -1
function __maybe_reduce_BLAS_threads(::Type{CPUDevice})::Int
    old_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    return old_threads
end

CRC.@non_differentiable __maybe_reduce_BLAS_threads(::AbstractArray)
EnzymeRules.inactive_noinl(::typeof(__maybe_reduce_BLAS_threads), ::AbstractArray) = nothing

function __reset_BLAS_threads(old_threads::Int)
    old_threads ≥ 1 && BLAS.set_num_threads(old_threads)
    return nothing
end

CRC.@non_differentiable __reset_BLAS_threads(::Int)
EnzymeRules.inactive_noinl(::typeof(__reset_BLAS_threads), ::Int) = nothing

function __get_concrete_fba_output_eltype(act::F, ::AbstractArray{Tw}, ::AbstractArray{Tx},
        b::Optional{<:AbstractVector}) where {F, Tw, Tx}
    if b === nothing
        Ty = promote_type(Tw, Tx)
        Tact = Core.Compiler._return_type(act, Tuple{Ty})
        return ifelse(isconcretetype(Tact), Tact, Ty)
    end
    Ty = promote_type(Tw, Tx, eltype(b))
    Tact = Core.Compiler._return_type(act, Tuple{Ty})
    return ifelse(isconcretetype(Tact), Tact, Ty)
end

function __get_concrete_fba_output_eltype(
        act::F, x::AbstractArray, b::Optional{<:AbstractVector}) where {F}
    return __get_concrete_fba_output_eltype(act, x, x, b)
end

CRC.@non_differentiable __get_concrete_fba_output_eltype(::Any...)
EnzymeRules.inactive_noinl(::typeof(__get_concrete_fba_output_eltype), ::Any...) = nothing

## Copy and don't allow gradient propagation
_copy_autodiff_barrier(x) = copy(remove_tracking(x))
_copy_autodiff_barrier(::Nothing) = nothing

CRC.@non_differentiable _copy_autodiff_barrier(::Any)
EnzymeRules.inactive_noinl(::typeof(_copy_autodiff_barrier), ::Any...) = nothing

## depwarn but marked non-differentiable to prevent type instability
__depwarn(msg::String, f::Symbol) = Base.depwarn(msg, f)

CRC.@non_differentiable __depwarn(::Any...)

__eltype(::AbstractArray{T}) where {T} = T
__eltype(::T) where {T <: Number} = T
__eltype(::Nothing) = Bool

CRC.@non_differentiable __eltype(::Any)
EnzymeRules.inactive_noinl(::typeof(__eltype), ::Any) = nothing

__default_epsilon(::Type{T}) where {T} = T(eps(T)^(5 / 7))
__default_epsilon(::AbstractArray{T}) where {T} = __default_epsilon(T)

CRC.@non_differentiable __default_epsilon(::Any...)
EnzymeRules.inactive_noinl(::typeof(__default_epsilon), ::Any...) = nothing

__unsafe_free!(x) = nothing
__unsafe_free!(x::AbstractArray) = KA.unsafe_free!(x)

CRC.@non_differentiable __unsafe_free!(::Any)
EnzymeRules.inactive_noinl(::typeof(__unsafe_free!), ::Any) = nothing

# Meta Programming Utilities
__is_tracked(x) = x == :TrackedArray || x == :TrackedVector
__is_tracked(args...) = any(__is_tracked, args)

## This part is taken from NNlib.jl
# This just saves typing `only.(only.(` many times:
only_derivative(y, f::F, x) where {F} = only(only(CRC.derivatives_given_output(y, f, x)))

# This has no methods, used for testing whether `derivatives_given_output(Ω, f, x)`
# is independent of `x`, as `_return_type` says `Union{}` when calling is an error.
struct NotaNumber <: Real end

# How to take activation gradients?
# See https://github.com/FluxML/NNlib.jl/blob/d85402aa39ddc6386d194e0dad88ab2e514ec5ea/src/bias_act.jl#L59-L60
function __no_intermediate_needed(f::F, ::Type{T}) where {F, T}
    f === identity && return true
    return isconcretetype(Core.Compiler._return_type(
        only_derivative, Tuple{T, F, NotaNumber}))
end

function __needs_intermediate_but_has_rrule(f::F, ::Type{T}) where {F, T}
    return isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
end

# Switches function `foo` with function `bar`. To be used when Enzyme cannot differentiate
# through `foo` but supports `bar`. Use with caution, avoid multiple dispatch on `foo`.
# Also the function should always return `nothing`
macro enzyme_reverse_alternative(f₁, f₂)
    return esc(quote
        function EnzymeRules.augmented_primal(
                ::EnzymeRules.ConfigWidth, ::EnzymeCore.Const{typeof($(f₁))},
                ::Type{RT}, args...) where {RT}
            fwd, rev = EnzymeCore.autodiff_thunk(
                EnzymeCore.ReverseSplitWithPrimal, EnzymeCore.Const{typeof($(f₂))},
                EnzymeCore.Const, typeof.(args)...)

            tape, result, shadow_result = fwd(EnzymeCore.Const($(f₂)), args...)

            return EnzymeRules.AugmentedReturn(result, shadow_result, (tape, rev))
        end

        function EnzymeRules.reverse(
                ::EnzymeRules.ConfigWidth, ::EnzymeCore.Const{typeof($(f₁))},
                ::Type{RT}, (tape, rev), args...) where {RT}
            return only(rev(EnzymeCore.Const($(f₂)), args..., tape))
        end
    end)
end

# UnrolledUtilities.jl has these functions. But we need to support Static so we make some
# specialized versions
inferred_length(::Type{<:NTuple{N, Any}}) where {N} = N

@generated function unrolled_any(f::F, xs) where {F}
    L = inferred_length(xs)
    L == 1 && return :(f(xs[1]))
    return Expr(:call, :|, (:(f(xs[$i])) for i in 1:L)...)
end
@generated function unrolled_all(f::F, xs) where {F}
    L = inferred_length(xs)
    L == 1 && return :(f(xs[1]))
    return Expr(:call, :&, (:(f(xs[$i])) for i in 1:L)...)
end
