"""
    LuxOps

This module is a part of `Lux.jl`. It contains operations that are useful in DL context.
Additionally certain operations here alias Base functions to behave more sensibly with
GPUArrays.
"""
module LuxOps

using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @thunk
using Compat: @compat
using EnzymeCore: EnzymeCore, EnzymeRules
using FastClosures: @closure
using MLDataDevices: get_device_type, AbstractGPUDevice, AbstractDevice

using ..Utils: Utils

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

"""
    getproperty(x, ::Val{v})

Similar to `Base.getproperty` but requires a `Val`. Additionally if `v` is not present in
`x`, then `nothing` is returned.
"""
function getproperty(x, ::Val{v}) where {v}
    return v ∈ Base.propertynames(x) ? Base.getproperty(x, v) : nothing
end
@generated function getproperty(x::NamedTuple{names}, ::Val{v}) where {names, v}
    return v ∈ names ? :(x.$v) : :(nothing)
end

"""
    eachslice(x, dims::Val)

Same as `Base.eachslice` but doesn't produce a `SubArray` for the slices if `x` is a
GPUArray.

Additional dispatches for RNN helpers are also provided for `TimeLastIndex` and
`BatchLastIndex`.
"""
function eachslice(x::AbstractArray, ::Val{dims}) where {dims <: Integer}
    return eachslice(get_device_type(x), x, dims)
end
function eachslice(
        ::Type{<:AbstractGPUDevice}, x::AbstractArray, ::Val{dims}) where {dims <: Integer}
    return [Utils.contiguous(selectdim(x, dims, i)) for i in axes(x, dims)]
end
function eachslice(
        ::Type{<:AbstractDevice}, x::AbstractArray, ::Val{dims}) where {dims <: Integer}
    return [selectdim(x, dims, i) for i in axes(x, dims)]
end

function ∇eachslice(Δ′, x::AbstractArray, ::Val{dims}) where {dims <: Integer}
    Δs = CRC.unthunk(Δ′)
    idx = findfirst(Base.Fix2(isa, AbstractArray), Δs)
    idx === nothing && return zero.(x)
    Δ = similar(x)
    for i in axes(x, dims)
        Δᵢ = selectdim(Δ, dims, i)
        copyto!(Δᵢ, Δs[i])
    end
    return CRC.ProjectTo(x)(Δ)
end

function CRC.rrule(
        ::typeof(eachslice), x::AbstractArray, d::Val{dims}) where {dims <: Integer}
    ∇eachslice_internal = @closure Δ -> (NoTangent(), ∇eachslice(Δ, x, d), NoTangent())
    return eachslice(x, d), ∇eachslice_internal
end

"""
    foldl_init(op, x)
    foldl_init(op, x, init)

Exactly same as `foldl(op, x; init)` in the forward pass. But, gives gradients wrt `init`
in the backward pass.
"""
foldl_init(op, x) = foldl_init(op, x, nothing)
foldl_init(op, x, init) = foldl(op, x; init)

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(foldl_init), op::G, x::Tuple, init) where {G}
    x_arr = [x...]
    y, ∇foldl_init_internal = CRC.rrule_via_ad(cfg, foldl_init, op, x_arr, init)
    ∇foldl_init = @closure Δ -> begin
        ∂foldl_init, ∂op, ∂x, ∂init = ∇foldl_init_internal(Δ)
        return ∂foldl_init, ∂op, Tuple(∂x), ∂init
    end
    return y, ∇foldl_init
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(foldl_init), op::G, x::AbstractArray, init) where {G}
    list, start = x, init
    accum_func = @closure (a, b) -> CRC.rrule_via_ad(cfg, op, first(a), b)
    accum_func_inner = @closure (x1, x2) -> begin
        (_d1, dc, _d3) = x1
        (_val, back) = x2
        return back(dc)
    end
    hobbits = Vector{Any}(undef, length(list))  # Unfornately Zygote needs this for CUDA
    accumulate!(accum_func, hobbits, list; init=(start, nothing))
    y = first(last(hobbits))
    ax = axes(x)
    project = CRC.ProjectTo.(x)
    ∇foldl_init = @closure Δ -> begin
        trio = accumulate(accum_func_inner, reverse(hobbits); init=(0, Δ, 0))
        ∂op = sum(first, trio)
        ∂x = reshape(map(last, reverse(trio)), ax)
        return (NoTangent(), ∂op,
            [proj(∂xᵢ) for (proj, ∂xᵢ) in zip(project, ∂x)], last(trio)[2])
    end
    return y, ∇foldl_init
end

"""
    multigate(x::AbstractArray, ::Val{N})

Split up `x` into `N` equally sized chunks (along dimension `1`).
"""
function multigate(x::AbstractArray, ::Val{N}) where {N}
    return ntuple(i -> Utils.gate(x, size(x, 1) ÷ N, i), N)
end

function ∇multigate(Δ::AbstractArray, x::AbstractArray, ::Val{N}) where {N}
    ∂x = similar(x, eltype(x), axes(x))
    foreach(multigate(∂x, Val(N)), Δ) do ∂xᵢ, Δᵢ
        if Δᵢ isa CRC.AbstractZero
            fill!(∂xᵢ, false)
        else
            copyto!(∂xᵢ, Δᵢ)
        end
    end
    return CRC.ProjectTo(x)(∂x)
end

function CRC.rrule(::typeof(multigate), x::AbstractArray, c::Val{N}) where {N}
    ∇multigate_internal = @closure Δ -> (
        NoTangent(), @thunk(∇multigate(Δ, x, c)), NoTangent())
    return multigate(x, c), ∇multigate_internal
end

# Public API

@compat(public, (xlogx, xlogy, getproperty, eachslice, foldl_init, multigate))

end
