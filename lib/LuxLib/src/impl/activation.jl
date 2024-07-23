# Used inside rrules
__activation_gradient(Δ, out, ::typeof(identity), x) = Δ
function __activation_gradient(Δ, out, act::F, x) where {F}
    opmode = internal_operation_mode((Δ, out, x))
    if opmode isa LoopedArrayOp  # All sizes are same
        y = similar(out)
        if x isa NotaNumber
            @simd ivdep for i in eachindex(Δ, out)
                @inbounds y[i] = only_derivative(out[i], act, x) * Δ[i]
            end
        else
            @simd ivdep for i in eachindex(Δ, out, x)
                @inbounds y[i] = only_derivative(out[i], act, x[i]) * Δ[i]
            end
        end
        return y
    end
    only_deriv = @closure (Δᵢ, oᵢ, xᵢ) -> Δᵢ * only_derivative(oᵢ, act, xᵢ)
    return broadcast(only_deriv, Δ, out, x)
end

function _fast_activation!(
        ::LoopedArrayOp, y::AbstractArray, σ::F, x::AbstractArray) where {F}
    σ_sleef = sleefpirates_activation(σ)
    @simd ivdep for I in eachindex(y, x)
        @inbounds y[I] = σ_sleef(x[I])
    end
end
function _fast_activation!(opmode, y::AbstractArray, σ::F, x::AbstractArray) where {F}
    broadcast!(σ, y, x)
    return
end

# Entry Points to the implementation
_fast_activation(::typeof(identity), x::AbstractArray) = x

@stable default_mode="disable" function _fast_activation(σ::F, x::AbstractArray) where {F}
    return _fast_activation(internal_operation_mode(x), σ, x)
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(_fast_activation),
        σ::F, x::AbstractArray{T}) where {F, T}
    return CRC.rrule_via_ad(cfg, broadcast, σ, x)
end

_fast_activation(opmode, σ::F, x::AbstractArray) where {F} = broadcast(σ, x)

function _fast_activation(opmode::LoopedArrayOp, σ::F, x::AbstractArray) where {F}
    RT = Core.Compiler._return_type(σ, Tuple{eltype(x)})
    y = similar(x, ifelse(isconcretetype(RT), RT, eltype(x)))
    _fast_activation!(opmode, y, σ, x)
    return y
end

_fast_activation!(::typeof(identity), x::AbstractArray) = nothing

@stable default_mode="disable" function _fast_activation!(σ::F, x::AbstractArray) where {F}
    _fast_activation!(internal_operation_mode(x), x, σ, x)
    return nothing
end

# Define rrule for `fast_activation!!`
function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(fast_activation!!),
        σ::F, x::AbstractArray{T}) where {F, T}
    can_setindex(typeof(x)) || return CRC.rrule_via_ad(cfg, _fast_activation, σ, x)

    σ === identity && return x, @closure(Δ->(∂∅, ∂∅, Δ))

    if __no_intermediate_needed(σ, T)
        _fast_activation!(σ, x) # Safe to overwrite x
        proj_x_no_cached = CRC.ProjectTo(x)
        ∇__fast_activation_impl_no_cached = @closure Δ -> begin
            ∂x = __activation_gradient(Δ, x, σ, NotaNumber())
            return ∂∅, ∂∅, proj_x_no_cached(∂x)
        end
        return x, ∇__fast_activation_impl_no_cached
    end

    if __needs_intermediate_but_has_rrule(σ, T)
        y = _fast_activation(σ, x)
        proj_x_cached = CRC.ProjectTo(x)
        ∇__fast_activation_impl_cached_crc = @closure Δ -> begin
            ∂x = __activation_gradient(CRC.unthunk(Δ), y, σ, x)
            return ∂∅, ∂∅, proj_x_cached(∂x)
        end
        return y, ∇__fast_activation_impl_cached_crc
    end

    return CRC.rrule_via_ad(cfg, _fast_activation, σ, x)
end

# Specialized functions that use SLEEFPirates.jl to speed up the activation functions
sigmoid_fast_sleefpirates(x::Number) = SLEEFPirates.sigmoid_fast(x)
softplus_sleefpirates(x::Number) = SLEEFPirates.softplus(x)
logsigmoid_sleefpirates(x::Number) = -softplus_sleefpirates(-x)
elu_sleefpirates(x::Number, α=1) = SLEEFPirates.Elu(α)(x)
gelu_sleefpirates(x::Number) = SLEEFPirates.gelu(x)
swish_sleefpirates(x::Number) = Base.FastMath.mul_fast(x, sigmoid_fast_sleefpirates(x))
lisht_sleefpirates(x::Number) = Base.FastMath.mul_fast(x, tanh_fast_sleefpirates(x))
tanh_sleefpirates(x::Number) = SLEEFPirates.tanh(x)
tanh_fast_sleefpirates(x::Number) = SLEEFPirates.tanh_fast(x)

# TODO: Add scalar rules for these functions via ChainRules and Enzyme

# Convert to SLEEFPirates.jl
function sleefpirates_activation(f::F, x::AbstractArray{T}) where {F, T}
    internal_operation_mode(x) isa LoopedArrayOp || return f
    return sleefpirates_activation(f, T)
end

sleefpirates_activation(f::F, ::Type{T}) where {F, T} = f
sleefpirates_activation(f::F, ::Type{Float32}) where {F} = sleefpirates_activation(f)
sleefpirates_activation(f::F, ::Type{Float64}) where {F} = sleefpirates_activation(f)

for (fbase, ffast) in ((NNlib.sigmoid_fast, sigmoid_fast_sleefpirates),
    (NNlib.softplus, softplus_sleefpirates), (NNlib.logsigmoid, logsigmoid_sleefpirates),
    (NNlib.elu, elu_sleefpirates), (NNlib.gelu, gelu_sleefpirates),
    (NNlib.swish, swish_sleefpirates), (NNlib.lisht, lisht_sleefpirates),
    (Base.tanh, tanh_sleefpirates), (NNlib.tanh_fast, tanh_fast_sleefpirates))
    @eval sleefpirates_activation(::typeof($fbase)) = $ffast
end
sleefpirates_activation(f::F) where {F} = f
