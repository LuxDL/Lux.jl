module LuxLibTrackerExt

using ChainRulesCore: ChainRulesCore
using FastClosures: @closure
using LuxLib: LuxLib
using NNlib: NNlib, batched_mul, batched_adjoint
using Tracker: Tracker, @grad, TrackedArray, TrackedVector, TrackedReal

const CRC = ChainRulesCore

# Macro to load chainrules to Tracker
function LuxLib.__tracker_grad_from_chainrules(__source__, __module__, fcall)
    Meta.isexpr(fcall, :call) && length(fcall.args) ≥ 2 ||
        error("`@tracked_grad_from_chainrules` has to be applied to a function signature")
    f = fcall.args[1]
    kws_var = Meta.isexpr(fcall.args[2], :parameters) ? fcall.args[2].args[1].args[1] : :()
    rem_args = Meta.isexpr(fcall.args[2], :parameters) ? fcall.args[3:end] :
               fcall.args[2:end]
    xs = map(rem_args) do x
        Meta.isexpr(x, :(::)) || return x
        length(x.args) == 1 && return :($(gensym())::$(x.args[1])) # ::T without var name
        @assert length(x.args) == 2
        return :($(x.args[1])::$(x.args[2])) # x::T
    end
    xs_untyped = map(xs) do x
        Meta.isexpr(x, :(::)) || return x
        return x.args[1]
    end
    tracked_args = Int[]
    foreach(enumerate(xs)) do (i, x)
        Meta.isexpr(x, :(::)) || return
        x.args[2] in (:TrackedArray, :TrackedVector, :TrackedMatrix) || return
        push!(tracked_args, i)
    end
    @assert length(tracked_args) > 0 "No tracked arguments found."
    return esc(quote
        function $(f)($(xs...); $(kws_var)...)
            return Tracker.track($(f), $(xs_untyped...); $(kws_var)...)
        end
    end)
end

# NNlib: batched_mul
for T1 in (:AbstractArray, :TrackedArray), T2 in (:AbstractArray, :TrackedArray)
    LuxLib.__is_tracked(T1, T2) || continue

    @eval NNlib.batched_mul(x::$T1, y::$T2) = Tracker.track(batched_mul, x, y)
end

@grad function NNlib.batched_mul(
        A::AbstractArray{T1, 3}, B::AbstractArray{T2, 3}) where {T1, T2}
    ∇batched_mul = @closure Δ -> begin
        tmp = batched_mul(Δ, batched_adjoint(Tracker.data(B)))
        ∂A = size(A, 3) == 1 ? sum(tmp; dims=3) : tmp
        tmp = batched_mul(batched_adjoint(Tracker.data(A)), Δ)
        ∂B = size(B, 3) == 1 ? sum(tmp; dims=3) : tmp
        return Tracker.nobacksies(:batched_mul, (∂A, ∂B))
    end
    return batched_mul(Tracker.data(A), Tracker.data(B)), ∇batched_mul
end

# NNlib: gather
function NNlib.gather!(dst::AbstractArray, src::TrackedArray, idx::AbstractArray)
    return Tracker.track(NNlib.gather!, dst, src, idx)
end

@grad function NNlib.gather!(dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    ∇gather! = @closure Δ -> begin
        ∂src = NNlib.∇gather_src(Δ, size(src), idx)
        return Tracker.nobacksies(:gather, (nothing, ∂src, nothing))
    end
    return NNlib.gather!(dst, Tracker.data(src), idx), ∇gather!
end

# Base.repeat
Base.repeat(x::TrackedArray, counts...) = Tracker.track(Base.repeat, x, counts...)

@grad function Base.repeat(x, counts...)
    y, ∇repeat_cr = CRC.rrule(Base.repeat, Tracker.data(x), counts...)
    ∇repeat = @closure Δ -> begin
        res = ∇repeat_cr(Δ)[2:(2 + length(counts))]
        return Tracker.nobacksies(
            :repeat, map(x -> x isa CRC.NoTangent ? nothing : CRC.unthunk(x), res))
    end
    return y, ∇repeat
end

# Base.selectdim
Base.selectdim(x::TrackedArray, d::Integer, i) = Tracker.track(selectdim, x, d, i)

@grad function Base.selectdim(x::AbstractArray, d::Integer, i)
    x_ = Tracker.data(x)
    y = selectdim(x_, d, i)
    ∇selectdim = @closure Δ -> begin
        ∂x = zero(x_)
        selectdim(∂x, d, i) .= Tracker.data(Δ)
        return ∂x, nothing, nothing
    end
    return y, ∇selectdim
end

# utils.jl
function LuxLib._copy_autodiff_barrier(x::Union{TrackedArray, TrackedReal})
    return LuxLib._copy_autodiff_barrier(Tracker.data(x))
end

LuxLib._get_backend(x::TrackedArray) = LuxLib._get_backend(Tracker.data(x))

# api/dropout.jl
LuxLib._dropout_fptype(x::TrackedArray) = LuxLib._dropout_fptype(Tracker.data(x))

# api/groupnorm.jl
for T1 in (:TrackedArray, :AbstractArray),
    T2 in (:TrackedVector, :AbstractVector),
    T3 in (:TrackedVector, :AbstractVector)

    LuxLib.__is_tracked(T1, T2, T3) || continue

    @eval function LuxLib.groupnorm(
            x::$T1{<:Union{Float32, Float64}, 4}, scale::$T2{<:Union{Float32, Float64}},
            bias::$T3{<:Union{Float32, Float64}}; groups::Int, epsilon::Real)
        return Tracker.track(LuxLib.groupnorm, x, scale, bias; groups, epsilon)
    end
end

@grad function LuxLib.groupnorm(x::AbstractArray{<:Union{Float32, Float64}, 4},
        scale::AbstractVector{<:Union{Float32, Float64}},
        bias::AbstractVector{<:Union{Float32, Float64}}; groups::Int, epsilon::Real)
    LuxLib._assert_same_backend(Tracker.data(x), Tracker.data(scale), Tracker.data(bias))
    if length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("Length of `scale` and `bias` must be equal to the number of \
                             channels (N - 1 dim of the input array)."))
    end
    if size(x, 3) % groups != 0
        throw(ArgumentError("Number of channels $(size(x, 3)) must be divisible by the \
                             number of groups $groups."))
    end

    y, μ, σ⁻¹ = LuxLib._groupnorm(
        Tracker.data(x), groups, Tracker.data(scale), Tracker.data(bias), epsilon)
    ∇groupnorm = @closure Δ -> begin
        dx, dscale, dbias = LuxLib._∇groupnorm(
            Δ, y, Tracker.data(x), groups, Tracker.data(scale), Tracker.data(bias), μ, σ⁻¹)
        return Tracker.nobacksies(:groupnorm, (dx, dscale, dbias))
    end
    return y, ∇groupnorm
end

end
