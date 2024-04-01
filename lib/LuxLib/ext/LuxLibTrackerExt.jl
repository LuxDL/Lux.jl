module LuxLibTrackerExt

using ChainRulesCore: ChainRulesCore
using FastClosures: @closure
using LuxLib: LuxLib
using NNlib: NNlib, batched_mul, batched_adjoint
using Tracker: Tracker, @grad, TrackedArray, TrackedVector, TrackedReal

const CRC = ChainRulesCore

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
        _, res... = ∇repeat_cr(Δ)
        return nobacksies(
            :repeat, map(x -> x == CRC.NoTangent() ? nothing : CRC.unthunk(x), res))
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
        return nobacksies(:groupnorm, (dx, dscale, dbias))
    end
    return y, ∇groupnorm
end

end
