module Utils

using ChainRulesCore: ChainRulesCore
using EnzymeCore: EnzymeCore, EnzymeRules
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: LinearAlgebra, BLAS
using MLDataDevices: get_device_type, CPUDevice
using NNlib: NNlib
using Static: Static, False

using ..LuxLib: Optional

const CRC = ChainRulesCore
const KA = KernelAbstractions

is_extension_loaded(::Val) = False()

# Simple Operations -- no rrules needed
vec(x::Number) = x
vec(x::AbstractArray) = Base.vec(x)
vec(::Nothing) = nothing

ofeltype_array(::Type{T}, x::AbstractArray{T}) where {T} = x
ofeltype_array(::Type{T}, x::AbstractArray) where {T} = convert(AbstractArray{T}, x)
ofeltype_array(::Type{T}, ::Nothing) where {T} = nothing

contiguous(x::AbstractArray) = x
contiguous(x::SubArray) = copy(x)

reshape(x::AbstractArray, dims...) = Base.reshape(x, dims)
reshape(::Nothing, dims...) = nothing

remove_tracking(x::Number) = x
remove_tracking(x::AbstractArray) = x
remove_tracking(::Type{T}) where {T <: Number} = T
remove_tracking(x::ForwardDiff.Dual) = ForwardDiff.value(x)
remove_tracking(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)
remove_tracking(::Type{<:ForwardDiff.Dual{Tag, T}}) where {Tag, T} = remove_tracking(T)
remove_tracking(::Nothing) = nothing

## This part is taken from NNlib.jl
# This has no methods, used for testing whether `derivatives_given_output(Ω, f, x)`
# is independent of `x`, as `_return_type` says `Union{}` when calling is an error.
struct NotaNumber <: Real end

# This just saves typing `only.(only.(` many times:
only_derivative(y, f::F, x) where {F} = only(only(CRC.derivatives_given_output(y, f, x)))

# Non-differentiable functions
## Reduce BLAS threads if we are going to use a native Julia implementation
maybe_reduce_BLAS_threads(x::AbstractArray) = maybe_reduce_BLAS_threads(get_device_type(x))
maybe_reduce_BLAS_threads(::Type{T}) where {T} = -1
function maybe_reduce_BLAS_threads(::Type{CPUDevice})::Int
    old_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    return old_threads
end

CRC.@non_differentiable maybe_reduce_BLAS_threads(::AbstractArray)

function reset_BLAS_threads(old_threads::Int)
    old_threads ≥ 1 && BLAS.set_num_threads(old_threads)
    return nothing
end

CRC.@non_differentiable reset_BLAS_threads(::Int)

unsafe_free!(_) = nothing
unsafe_free!(x::AbstractArray) = KA.unsafe_free!(x)

CRC.@non_differentiable unsafe_free!(::Any)

known(x) = Static.known(x)  # will drop gradients. needed for type stability in Zygote

CRC.@non_differentiable known(::Any)

## depwarn but marked non-differentiable to prevent type instability
depwarn(msg::String, f::Symbol) = Base.depwarn(msg, f)

CRC.@non_differentiable depwarn(::Any...)

eltype(::AbstractArray{T}) where {T} = T
eltype(::T) where {T <: Number} = T
eltype(::Nothing) = Bool

CRC.@non_differentiable eltype(::Any)

default_epsilon(::Type{T}) where {T} = T(eps(T)^(5 / 7))
default_epsilon(::AbstractArray{T}) where {T} = default_epsilon(T)

CRC.@non_differentiable default_epsilon(::Any...)

function concrete_bias_act_output_eltype(act::F, ::AbstractArray{Tw}, ::AbstractArray{Tx},
        b::Optional{<:AbstractVector}) where {F, Tw, Tx}
    Ty = promote_type(Tw, Tx, eltype(b))
    Tact = Core.Compiler._return_type(act, Tuple{Ty})
    return ifelse(isconcretetype(Tact), Tact, Ty)
end

function concrete_bias_act_output_eltype(
        act::F, x::AbstractArray, b::Optional{<:AbstractVector}) where {F}
    return concrete_bias_act_output_eltype(act, x, x, b)
end

CRC.@non_differentiable concrete_bias_act_output_eltype(::Any...)

## Copy and don't allow gradient propagation
copy_drop_gradients(x) = copy(remove_tracking(x))
copy_drop_gradients(::Nothing) = nothing

CRC.@non_differentiable copy_drop_gradients(::Any)
EnzymeRules.inactive_noinl(::typeof(copy_drop_gradients), ::Any...) = nothing

# Meta Programming Utilities
is_tracked(x) = x == :TrackedArray || x == :TrackedVector
is_tracked(args...) = unrolled_any(is_tracked, args)

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

# Working with batches
batchview(x::AbstractArray{<:Any, 3}, k::Int) = view(x, :, :, k)
batchview(x::NNlib.BatchedTranspose, k::Int) = transpose(batchview(parent(x), k))
batchview(x::NNlib.BatchedAdjoint, k::Int) = adjoint(batchview(parent(x), k))

batchview(x::AbstractArray{<:Any, 3}) = map(Base.Fix1(batchview, x), 1:size(x, 3))

expand_batchdim(x::AbstractMatrix) = reshape(x, size(x)..., 1)
function expand_batchdim(x::LinearAlgebra.Adjoint)
    return NNlib.BatchedAdjoint(reshape(parent(x), size(parent(x))..., 1))
end
function expand_batchdim(x::LinearAlgebra.Transpose)
    return NNlib.BatchedTranspose(reshape(parent(x), size(parent(x))..., 1))
end

function CRC.rrule(::typeof(expand_batchdim), x::AbstractMatrix)
    proj_x = CRC.ProjectTo(x)
    ∇expand_batchdim = @closure Δ -> begin
        return ∂∅, proj_x(view(Δ, :, :, 1))
    end
    return expand_batchdim(x), ∇expand_batchdim
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

end
