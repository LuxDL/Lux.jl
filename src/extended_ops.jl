"""
    LuxOps

This module is a part of `Lux.jl`. It contains operations that are useful in DL context.
Additionally certain operations here alias Base functions to behave more sensibly with
GPUArrays.
"""
module LuxOps

using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @thunk
using EnzymeCore: EnzymeCore, EnzymeRules
using FastClosures: @closure

const CRC = ChainRulesCore

# `xlogx` and `xlogy`
## We don't use `LogExpFunctions` since they don't support GPU broadcasting. See
## https://github.com/LuxDL/Lux.jl/pull/796. Additionally we have special broadcast rrules.
"""
    xlogx(x::Number)

Return `x * log(x)` for `x ≥ 0`, handling `x == 0` by taking the limit from above, to get
zero.
"""
function xlogx(x::Number)
    result = x * log(x)
    return ifelse(iszero(x), zero(result), result)
end

∇xlogx(Δ::Number, logx::Number) = Δ * (logx + true)

function CRC.rrule(::typeof(xlogx), x::Number)
    iszero(x) && return x, Returns((NoTangent(), ZeroTangent()))
    logx = log(x)
    ∇xlogx_internal = @closure Δ -> (NoTangent(), @thunk(∇xlogx(Δ, logx)))
    return x * logx, ∇xlogx_internal
end

function CRC.rrule(
        ::typeof(Broadcast.broadcasted), ::typeof(xlogx), x::AbstractArray{<:Number})
    logx = log.(x)
    ∇xlogx_internal = @closure Δ -> (NoTangent(), NoTangent(), @thunk(∇xlogx.(Δ, logx)))
    return .*(x, logx), ∇xlogx_internal
end

"""
    xlogy(x::Number, y::Number)

Return `x * log(y)` for `y > 0`, and zero when `x == 0`.
"""
@inline function xlogy(x::Number, y::Number)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

∇₁xlogy(Δ::Number, logy::Number) = Δ * logy
∇₂xlogy(Δ::Number, x::Number, y::Number) = Δ * x / y

function CRC.rrule(::typeof(xlogy), x::Number, y::Number)
    iszero(x) && return x, Returns((NoTangent(), ZeroTangent()))
    logy = log(y)
    ∇xlogy_internal = @closure Δ -> (
        NoTangent(), @thunk(∇₁xlogy(Δ, logy)), @thunk(∇₂xlogy(Δ, x, y)))
    return x * logy, ∇xlogy_internal
end

function CRC.rrule(::typeof(Broadcast.broadcasted), ::typeof(xlogy),
        x::AbstractArray{<:Number}, y::AbstractArray{<:Number})
    logy = log.(y)
    ∇xlogy_internal = @closure Δ -> (
        NoTangent(), NoTangent(), @thunk(∇₁xlogy.(Δ, logy)), @thunk(∇₂xlogy.(Δ, x, y)))
    return .*(x, logy), ∇xlogy_internal
end

export xlogx, xlogy

end
