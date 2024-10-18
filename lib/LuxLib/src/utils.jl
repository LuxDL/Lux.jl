module Utils

using ChainRulesCore: ChainRulesCore
using EnzymeCore: EnzymeCore, EnzymeRules
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: LinearAlgebra, BLAS
using MLDataDevices: get_device_type, CPUDevice
using NNlib: NNlib
using Static: Static, StaticBool, False, True, static
using StaticArraysCore: SVector, SMatrix

using ..LuxLib: Optional, ∂∅, DISABLE_LOOP_VECTORIZATION

const CRC = ChainRulesCore
const KA = KernelAbstractions

is_extension_loaded(::Val) = False()

CRC.@non_differentiable is_extension_loaded(::Any...)
EnzymeRules.inactive_noinl(::typeof(is_extension_loaded), ::Any...) = nothing

# Simple Operations -- no rrules needed
ofeltype_array(::Type{T}, x::AbstractArray{T}) where {T} = x
function ofeltype_array(
        ::Type{T}, x::AbstractArray{<:ForwardDiff.Dual{Tag, T, N}}) where {Tag, T, N}
    return x
end
ofeltype_array(::Type{T}, x::AbstractArray) where {T} = T.(x)
function ofeltype_array(
        ::Type{T}, x::AbstractArray{<:ForwardDiff.Dual{Tag, T2, N}}) where {Tag, T, T2, N}
    return ForwardDiff.Dual{Tag, T, N}.(x)
end
ofeltype_array(::Type{T}, ::Nothing) where {T} = nothing

contiguous(x::AbstractArray) = x
contiguous(x::SubArray) = copy(x)

safe_reshape(x::AbstractArray, dims...) = reshape(x, dims...)
safe_reshape(::Nothing, dims...) = nothing

remove_tracking(x) = x
remove_tracking(x::AbstractArray) = x
remove_tracking(::Type{T}) where {T} = T
remove_tracking(x::ForwardDiff.Dual) = ForwardDiff.value(x)
remove_tracking(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)
remove_tracking(::Type{<:ForwardDiff.Dual{Tag, T}}) where {Tag, T} = remove_tracking(T)
remove_tracking(::Nothing) = nothing

safe_vec(x) = x
safe_vec(x::AbstractArray) = vec(x)
safe_vec(::Nothing) = nothing

## This part is taken from NNlib.jl
# This has no methods, used for testing whether `derivatives_given_output(Ω, f, x)`
# is independent of `x`, as `_return_type` says `Union{}` when calling is an error.
struct NotaNumber <: Real end

# This just saves typing `only.(only.(` many times:
only_derivative(y, f::F, x) where {F} = only(only(CRC.derivatives_given_output(y, f, x)))

# Non-differentiable functions
eltype_mismatch(::Type, ::Type) = True()
eltype_mismatch(::Type{T}, ::Type{T}) where {T} = False()
function eltype_mismatch(::Type{T}, ::Type{<:ForwardDiff.Dual{Tag, T, N}}) where {Tag, T, N}
    return False()
end
function eltype_mismatch(::Type{<:ForwardDiff.Dual{Tag, T, N}}, ::Type{T}) where {Tag, T, N}
    return False()
end

CRC.@non_differentiable eltype_mismatch(::Any...)

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

unsafe_known(x) = Static.known(x)  # will drop gradients. needed for type stability in Zygote

CRC.@non_differentiable unsafe_known(::Any)

## depwarn but marked non-differentiable to prevent type instability
depwarn(msg::String, f::Symbol) = Base.depwarn(msg, f)

CRC.@non_differentiable depwarn(::Any...)

safe_eltype(::AbstractArray{T}) where {T} = T
safe_eltype(::T) where {T} = T
safe_eltype(::Nothing) = Bool

CRC.@non_differentiable safe_eltype(::Any)

default_epsilon(::Type{T}) where {T} = T(eps(T)^(5 / 7))
default_epsilon(::AbstractArray{T}) where {T} = default_epsilon(T)

CRC.@non_differentiable default_epsilon(::Any...)

function concrete_bias_act_output_eltype(act::F, ::AbstractArray{Tw}, ::AbstractArray{Tx},
        b::Optional{<:AbstractVector}) where {F, Tw, Tx}
    Ty = promote_type(Tw, Tx, safe_eltype(b))
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

inferred_length(::Type{<:NTuple{N, Any}}) where {N} = N
@generated static_length(itr) = return :($(Val(inferred_length(itr))))

@generated function unrolled_any(f::F, xs) where {F}
    L = inferred_length(xs)
    L == 1 && return :(f(xs[1]))
    return Expr(:call, :|, (:(f(xs[$i])) for i in 1:L)...)
end

@generated function unrolled_map(f::F, xs) where {F}
    L = inferred_length(xs)
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:inbounds, true))
        res = $(Expr(:tuple, (:(f(xs[$i])) for i in 1:L)...))
        $(Expr(:inbounds, :pop))
        return res
    end
end

function unrolled_mapreduce(f::F, op::O, itr) where {F, O}
    return unrolled_mapreduce(f, op, itr, static_length(itr))
end

function unrolled_mapreduce(::F, ::O, _, ::Val{0}) where {F, O}
    error("Cannot unroll over an empty iterator.")
end

unrolled_mapreduce(f::F, ::O, itr, ::Val{1}) where {F, O} = f(only(itr))

@generated function unrolled_mapreduce(f::F, op::O, itr, ::Val{N}) where {F, O, N}
    syms = [gensym("f_itr_$(i)") for i in 1:N]
    op_syms = [gensym("op_$(i)") for i in 1:(N - 1)]
    f_applied = [:($(syms[i]) = f(itr[$i])) for i in 1:N]
    combine_expr = [:($(op_syms[1]) = op($(syms[1]), $(syms[2])))]
    for i in 2:(N - 1)
        push!(combine_expr, :($(op_syms[i]) = op($(op_syms[i - 1]), $(syms[i + 1]))))
    end
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:inbounds, true))
        $(Expr(:block, f_applied...))
        $(Expr(:inbounds, :pop))
        $(Expr(:block, combine_expr...))
        return $(op_syms[end])
    end
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
expand_batchdim(x::AbstractVector) = reshape(x, :, 1)
expand_batchdim(x::SVector{L, T}) where {L, T} = SMatrix{L, 1, T}(x)

function CRC.rrule(::typeof(expand_batchdim), x::AbstractMatrix)
    proj_x = CRC.ProjectTo(x)
    ∇expand_batchdim = @closure Δ -> begin
        return ∂∅, proj_x(view(Δ, :, :, 1))
    end
    return expand_batchdim(x), ∇expand_batchdim
end

function safe_warning(msg::String, maxlog::Int)
    if maxlog < 0
        @warn msg
    else
        @warn msg maxlog=maxlog
    end
end

CRC.@non_differentiable safe_warning(::Any...)

function safe_minimum(x::AbstractArray, default)
    length(x) == 0 && return default
    return minimum(x)
end

CRC.@non_differentiable safe_minimum(::Any...)

# Switches function `foo` with function `bar`. To be used when Enzyme cannot differentiate
# through `foo` but supports `bar`. Use with caution, avoid multiple dispatch on `foo`.
# Also the function should always return `nothing`
macro enzyme_alternative(f₁, f₂)
    return esc(quote
        function EnzymeRules.augmented_primal(
                ::EnzymeRules.RevConfig, ::EnzymeCore.Const{typeof($(f₁))},
                ::Type{RT}, args...) where {RT}
            fwd, rev = EnzymeCore.autodiff_thunk(
                EnzymeCore.ReverseSplitWithPrimal, EnzymeCore.Const{typeof($(f₂))},
                EnzymeCore.Const, typeof.(args)...)

            tape, result, shadow_result = fwd(EnzymeCore.Const($(f₂)), args...)

            return EnzymeRules.AugmentedReturn(result, shadow_result, (tape, rev))
        end

        function EnzymeRules.reverse(
                ::EnzymeRules.RevConfig, ::EnzymeCore.Const{typeof($(f₁))},
                ::Type{RT}, (tape, rev), args...) where {RT}
            return only(rev(EnzymeCore.Const($(f₂)), args..., tape))
        end

        function EnzymeRules.forward(cfg::EnzymeRules.FwdConfig,
                ::EnzymeCore.Const{typeof($(f₁))}, ::Type{RT}, args...) where {RT}
            EnzymeCore.autodiff(EnzymeCore.Forward, EnzymeCore.Const($(f₂)), RT, args...)
            return
        end
    end)
end

@inline function run_ka_kernel(f::F, backend, workgroupsize, ndrange, args...) where {F}
    if workgroupsize === nothing
        kernel = f(backend)
        kernel(args...; ndrange)
        return
    end
    kernel = f(backend, KA.StaticSize(workgroupsize), KA.StaticSize(ndrange))
    kernel(args...)
    return
end

within_autodiff_vararg(args...) = unrolled_any(within_autodiff, args)

function within_autodiff(_)
    unsafe_known(is_extension_loaded(Val(:Enzyme))) &&
        return static(EnzymeCore.within_autodiff())
    return False()
end
within_autodiff(::ForwardDiff.Dual) = True()
within_autodiff(::AbstractArray{<:ForwardDiff.Dual}) = True()

CRC.rrule(::typeof(within_autodiff), x) = True(), _ -> (∂∅, ∂∅)

static_training_mode(::Nothing, args...) = within_autodiff_vararg(args...)

function static_training_mode(
        training::Union{Bool, Val{true}, Val{false}, StaticBool}, args...)
    return static_training_mode_check(
        training, static(training), within_autodiff_vararg(args...))
end

function CRC.rrule(::typeof(static_training_mode), ::Nothing, args...)
    return True(), _ -> ntuple(Returns(∂∅), length(args) + 2)
end

function CRC.rrule(::typeof(static_training_mode),
        training::Union{Bool, Val{true}, Val{false}, StaticBool}, args...)
    res = static_training_mode_check(training, static(training), True())
    return res, _ -> ntuple(Returns(∂∅), length(args) + 2)
end

static_training_mode_check(_, ::True, ::True) = True()
static_training_mode_check(_, ::False, ::False) = False()

function static_training_mode_check(training, ::True, ::False)
    @warn "`training` is set to `$(training)` but is not being used within an autodiff \
           call (gradient, jacobian, etc...). This will be slow. If you are using a \
           `Lux.jl` model, set it to inference (test) mode using `LuxCore.testmode`. \
           Reliance on this behavior is discouraged, and is not guaranteed by Semantic \
           Versioning, and might be removed without a deprecation cycle. It is recommended \
           to fix this issue in your code." maxlog=1
    return True()
end

function static_training_mode_check(training, ::False, ::True)
    @warn "`training` is set to `$(training)` but is being used within an autodiff call \
           (gradient, jacobian, etc...). This might lead to incorrect results. If you are \
           using a `Lux.jl` model, set it to training mode using \
           `LuxCore.trainmode`." maxlog=1
    return False()
end

CRC.@non_differentiable static_training_mode_check(::Any...)

@static if DISABLE_LOOP_VECTORIZATION
    @inline can_loopvec_args(args...) = false
else
    @inline function can_loopvec_args(args...)
        return can_loopvec_args_check(is_extension_loaded(Val(:LoopVectorization)), args...)
    end
end

@inline can_loopvec_args_check(::False, args...) = false

CRC.@non_differentiable can_loopvec_args_check(::Any...)

EnzymeRules.inactive_noinl(::typeof(can_loopvec_args_check), ::Any...) = nothing

end
