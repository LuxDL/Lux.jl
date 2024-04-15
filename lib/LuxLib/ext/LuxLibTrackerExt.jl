module LuxLibTrackerExt

using ChainRulesCore: ChainRulesCore
using FastClosures: @closure
using LuxLib: LuxLib
using NNlib: NNlib
using Tracker: Tracker, TrackedArray, TrackedVector, TrackedReal

const CRC = ChainRulesCore

# Macro to load chainrules to Tracker
@inline __no_crctangent(::CRC.NoTangent) = nothing
@inline __no_crctangent(::CRC.ZeroTangent) = nothing
@inline __no_crctangent(x::CRC.AbstractThunk) = CRC.unthunk(x)
@inline __no_crctangent(x) = x

## TODO: Upstream to Tracker.jl repo
function LuxLib.__tracker_grad_from_chainrules(__source__, __module__, fcall)
    @assert isdefined(__module__, :Tracker) "Tracker not found in module $__module__. Please load `Tracker.jl`."
    Meta.isexpr(fcall, :call) && length(fcall.args) ≥ 2 ||
        error("`@tracked_grad_from_chainrules` has to be applied to a function signature")
    f = fcall.args[1]
    kws_var = Meta.isexpr(fcall.args[2], :parameters) ? fcall.args[2].args[1].args[1] :
              nothing
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

    untrack_args = map(enumerate(xs)) do (i, x)
        Meta.isexpr(x, :(::)) || return (x, nothing)
        name, type = x.args
        Meta.isexpr(type, :curly) && (type = type.args[1])
        type in (:TrackedArray, :TrackedVector, :TrackedMatrix) || return (name, nothing)
        xdata = gensym(name)
        return xdata, :($(xdata) = $(Tracker.data)($(name)))
    end
    untrack_calls = filter(Base.Fix2(!==, nothing), last.(untrack_args))
    @assert length(untrack_calls)>0 "No tracked arguments found."
    var_names = first.(untrack_args)

    f_sym = Meta.quot(Symbol(f))

    if kws_var === nothing
        return esc(quote
            $(f)($(xs...)) = $(Tracker.track)($(f), $(xs_untyped...))
            function Tracker._forward(::typeof($(f)), $(xs...))
                $(untrack_calls...)
                y, pb_f = $(CRC.rrule)($(f), $(var_names...))
                ∇internal_generated = let pb_f = pb_f
                    Δ -> return Tracker.nobacksies(
                        $(f_sym), $(__no_crctangent).(pb_f(Δ)[2:end]))
                end
                return y, ∇internal_generated
            end
        end)
    end
    return esc(quote
        function $(f)($(xs...); $(kws_var)...)
            return Tracker.track($(f), $(xs_untyped...); $(kws_var)...)
        end
        function Tracker._forward(::typeof($(f)), $(xs...); $(kws_var)...)
            $(untrack_calls...)
            y, pb_f = $(CRC.rrule)($(f), $(var_names...); $(kws_var)...)
            ∇internal_generated = let pb_f = pb_f
                Δ -> Tracker.nobacksies($(f_sym), $(__no_crctangent).(pb_f(Δ)[2:end]))
            end
            return y, ∇internal_generated
        end
    end)
end

# NNlib: batched_mul
for T1 in (:AbstractArray, :TrackedArray), T2 in (:AbstractArray, :TrackedArray)
    LuxLib.__is_tracked(T1, T2) || continue

    @eval LuxLib.@tracker_grad_from_chainrules NNlib.batched_mul(x::$T1, y::$T2)
end

# NNlib: gather
LuxLib.@tracker_grad_from_chainrules NNlib.gather!(
    dst::AbstractArray, src::TrackedArray, idx::AbstractArray)

# Base.repeat
LuxLib.@tracker_grad_from_chainrules Base.repeat(x::TrackedArray, counts...)

# Base.selectdim -- Needed for GPUArrays
Base.selectdim(x::TrackedArray, d::Integer, i) = Tracker.track(selectdim, x, d, i)

Tracker.@grad function Base.selectdim(x::AbstractArray, d::Integer, i)
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

    @eval LuxLib.@tracker_grad_from_chainrules LuxLib.groupnorm(
        x::$T1{<:Union{Float32, Float64}, 4}, scale::$T2{<:Union{Float32, Float64}},
        bias::$T3{<:Union{Float32, Float64}}; groups::Int, epsilon::Real)
end

end
