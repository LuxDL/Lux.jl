# Entry Points
function activation!!(σ::F, x::AbstractArray) where {F}
    return activation!!(
        Traits.attempt_fast_implementation(x), select_fastest_activation(σ, x), x)
end

activation!(::typeof(identity), ::AbstractArray) = nothing
function activation!(σ::F, x::AbstractArray) where {F}
    activation!(Traits.attempt_fast_implementation(x), select_fastest_activation(σ, x), x)
    return nothing
end

activation(::typeof(identity), x::AbstractArray) = x
function activation(σ::F, x::AbstractArray) where {F}
    return activation(
        Traits.attempt_fast_implementation(x), select_fastest_activation(σ, x), x)
end

# Core Implementation
activation!!(::False, σ::F, x::AbstractArray) where {F} = activation(False(), σ, x)
function activation!!(::True, σ::F, x::AbstractArray) where {F}
    return activation!!(True(), Traits.is_mutable_array(x), σ, x)
end
activation!!(::True, ::False, σ::F, x::AbstractArray) where {F} = activation(True(), σ, x)
@stable default_mode="disable" function activation!!(
        ::True, ::True, σ::F, x::AbstractArray) where {F}
    activation!(True(), σ, x)
    return x
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(activation!!),
        ::True, ::True, σ::F, x::AbstractArray{T}) where {F, T}
    if Utils.known(Traits.activation_intermediate_not_needed(σ, T))
        activation!(True(), σ, x)
        𝒫x_no_intermediate = CRC.ProjectTo(x)
        ∇activation_no_intermediate_rrule = @closure Δ -> begin
            ∂x = ∇activation(CRC.unthunk(Δ), x, σ, Utils.NotaNumber())
            return ∂∅, ∂∅, ∂∅, ∂∅, 𝒫x_no_intermediate(∂x)
        end
        return x, ∇activation_no_intermediate_rrule
    end

    if Utils.known(Traits.activation_has_rrule(σ, T))
        y = activation(True(), σ, x)
        𝓟x_cached = CRC.ProjectTo(x)
        ∇activation_rrule = @closure Δ -> begin
            ∂x = ∇activation(CRC.unthunk(Δ), y, σ, x)
            return ∂∅, ∂∅, ∂∅, ∂∅, 𝓟x_cached(∂x)
        end
        return y, ∇activation_rrule
    end

    res, ∇activation_from_ad = CRC.rrule_via_ad(cfg, activation, True(), σ, x)
    ∇activation_fallback = @closure Δ -> begin
        ∂f, _, ∂σ, ∂x = ∇activation_from_ad(Δ)
        return ∂f, ∂∅, ∂∅, ∂σ, ∂x
    end
    return res, ∇activation_fallback
end

activation(::False, σ::F, x::AbstractArray) where {F} = broadcast(σ, x)
function activation(::True, σ::F, x::AbstractArray) where {F}
    return activation(internal_operation_mode(x), σ, x)
end

function activation(::AbstractInternalArrayOpMode, σ::F, x::AbstractArray) where {F}
    return broadcast(σ, x)
end
@stable default_mode="disable" function activation(
        opmode::LoopedArrayOp, σ::F, x::AbstractArray{T}) where {F, T}
    RT = Core.Compiler._return_type(σ, Tuple{T})
    y = similar(x, ifelse(isconcretetype(RT), RT, T))
    activation!(opmode, y, σ, x)
    return y
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(activation),
        opmode::LoopedArrayOp, σ::F, x::AbstractArray{T}) where {F, T}
    if Utils.known(Traits.activation_has_rrule(σ, T))
        y = activation(opmode, σ, x)
        𝓟x = CRC.ProjectTo(x)
        ∇activation_rrule = @closure Δ -> begin
            ∂x = ∇activation(CRC.unthunk(Δ), y, σ, x)
            return ∂∅, ∂∅, ∂∅, 𝓟x(∂x)
        end
        return y, ∇activation_rrule
    end

    z, ∇broadcast = CRC.rrule_via_ad(cfg, broadcast, σ, x)
    ∇activation_fallback = @closure Δ -> begin
        ∂f, ∂σ, ∂x = ∇broadcast(Δ)
        return ∂f, ∂∅, ∂σ, ∂x
    end
    return z, ∇activation_fallback
end

function activation!(::False, σ::F, x::AbstractArray) where {F}
    broadcast!(σ, x, x)
    return
end
function activation!(::True, σ::F, x::AbstractArray) where {F}
    return activation!(internal_operation_mode(x), x, σ, x)
end

function activation!(
        ::AbstractInternalArrayOpMode, y::AbstractArray, σ::F, x::AbstractArray) where {F}
    broadcast!(σ, y, x)
    return
end
function activation!(::LoopedArrayOp, y::AbstractArray, σ::F, x::AbstractArray) where {F}
    if LV.check_args(y, x)
        @tturbo for I in indices((y, x))
            y[I] = σ(x[I])
        end
    else
        @batch for I in indices((y, x))
            y[I] = σ(x[I])
        end
    end
end

function activation_no_turbo!(
        ::LoopedArrayOp, y::AbstractArray, σ::F, x::AbstractArray) where {F}
    @simd ivdep for I in eachindex(y, x)
        y[I] = σ(x[I])
    end
end

function EnzymeRules.augmented_primal(
        cfg::EnzymeRules.ConfigWidth{1}, ::EnzymeCore.Const{typeof(activation!)},
        ::Type{EnzymeCore.Const{Nothing}}, opmode::EnzymeCore.Const{LoopedArrayOp},
        y::EnzymeCore.Duplicated{<:AbstractArray}, σ::EnzymeCore.Const{F},
        x::EnzymeCore.Duplicated{<:AbstractArray}) where {F}
    dx = one.(x.val)
    dy = zero.(y.val)
    EnzymeCore.autodiff(EnzymeCore.Forward, activation_no_turbo!, opmode,
        EnzymeCore.Duplicated(y.val, dy), σ, EnzymeCore.Duplicated(x.val, dx))
    return EnzymeRules.AugmentedReturn(nothing, nothing, (dy,))
end

function EnzymeRules.reverse(
        ::EnzymeRules.ConfigWidth{1}, ::EnzymeCore.Const{typeof(activation!)},
        ::Type{EnzymeCore.Const{Nothing}}, (dy,), opmode::EnzymeCore.Const{LoopedArrayOp},
        y::EnzymeCore.Duplicated{<:AbstractArray}, σ::EnzymeCore.Const{F},
        x::EnzymeCore.Duplicated{<:AbstractArray}) where {F}
    if LV.check_args(y.dval, x.dval, dy)
        @tturbo for I in indices((y.dval, x.dval, dy))
            x.dval[I] = y.dval[I] * dy[I]
        end
    else
        @batch for I in indices((y.dval, x.dval, dy))
            x.dval[I] = y.dval[I] * dy[I]
        end
    end

    x.dval !== y.dval && fill!(y.dval, false)

    return nothing, nothing, nothing, nothing
end

# Gradient for activations
∇activation(Δ, _, ::typeof(identity), x) = Δ
function ∇activation(Δ, out, act::F, x) where {F}
    return ∇activation(internal_operation_mode((Δ, out)), Δ, out, act, x)
end
function ∇activation(::AbstractInternalArrayOpMode, Δ, out, act::F, x) where {F}
    ∇act = @closure (Δᵢ, oᵢ, xᵢ) -> Δᵢ * Utils.only_derivative(oᵢ, act, xᵢ)
    return broadcast(∇act, Δ, out, x)
end
function ∇activation(::LoopedArrayOp, Δ, out, act::F, x) where {F}
    y = similar(out)
    if x isa Utils.NotaNumber
        @simd ivdep for i in eachindex(Δ, out)
            @inbounds y[i] = Utils.only_derivative(out[i], act, x) * Δ[i]
        end
    else
        @batch for i in eachindex(Δ, out)
            @inbounds y[i] = Utils.only_derivative(out[i], act, x[i]) * Δ[i]
        end
    end
    return y
end

# Switch some of the activations to use SLEEFPirates.jl if needed
function select_fastest_activation(f::F, xs...) where {F}
    return select_fastest_activation(
        f, internal_operation_mode(xs), unrolled_mapreduce(Utils.eltype, promote_type, xs))
end

select_fastest_activation(f::F, ::AbstractInternalArrayOpMode, ::Type{T}) where {F, T} = f

function select_fastest_activation(f::F, ::LoopedArrayOp, ::Type{T}) where {F, T}
    return SLEEFActivations.fast_act(f, T)
end

CRC.@non_differentiable select_fastest_activation(::Any...)

# Fast activations via SLEEFPirates.jl
module SLEEFActivations

using ChainRulesCore: ChainRulesCore
using EnzymeCore: EnzymeCore, EnzymeRules
using NNlib: NNlib
using SLEEFPirates: SLEEFPirates

using ....LuxLib: Numeric

const CRC = ChainRulesCore

sigmoid_fast(x::Number) = SLEEFPirates.sigmoid_fast(x)
softplus(x::Number) = SLEEFPirates.softplus(x)
logsigmoid(x::Number) = -softplus(-x)
gelu(x::Number) = SLEEFPirates.gelu(x)
swish(x::Number) = Base.FastMath.mul_fast(x, sigmoid_fast(x))
lisht(x::Number) = Base.FastMath.mul_fast(x, tanh_fast(x))
tanh(x::Number) = SLEEFPirates.tanh(x)
tanh_fast(x::Number) = SLEEFPirates.tanh_fast(x)

const gelu_λ = √(2 / π)
const gelu_2λ = √(8 / π)

function ∇gelu(x::Number)
    α = oftype(x, 0.044715)
    α2 = oftype(x, 0.08943)
    λλ = oftype(x, gelu_2λ)
    x2 = Base.FastMath.mul_fast(x, x)
    t = muladd(x2, α, one(x))
    Ω = sigmoid_fast(λλ * x * t)
    dσ = conj(Ω * (1 - Ω))
    return muladd(dσ * λλ * muladd(x2, α2, t), x, Ω)
end

for (f, dfdx) in [
    #! format: off
    (:sigmoid_fast, :(conj(Base.FastMath.mul_fast(Ω, Base.FastMath.sub_fast(1, Ω))))),
    (:softplus, :(sigmoid_fast(x))),
    (:logsigmoid, :(sigmoid_fast(-x))),
    (:gelu, :(∇gelu(x))),
    (:swish, :(Base.FastMath.add_fast(Ω, Base.FastMath.mul_fast(sigmoid_fast(x), Base.FastMath.sub_fast(1, Ω))))),
    (:tanh, :(conj(Base.FastMath.sub_fast(1, Base.FastMath.mul_fast(Ω, Ω))))),
    (:tanh_fast, :(conj(Base.FastMath.sub_fast(1, Base.FastMath.mul_fast(Ω, Ω)))))
    #! format: on
]
    @eval CRC.@scalar_rule($f(x), $dfdx)

    ∇f = Symbol(:∇broadcasted_, f)
    @eval function CRC.rrule(::typeof(Broadcast.broadcasted), ::typeof($f),
            x::Union{Numeric, Broadcast.Broadcasted})
        Ω = $f.(x)
        function $∇f(dΩ)
            ∂x = CRC.InplaceableThunk(dx -> @.(dx+=dΩ * $dfdx), CRC.@thunk @.(dΩ*$dfdx))
            return CRC.NoTangent(), CRC.NoTangent(), ∂x
        end
        return Ω, $∇f
    end
end

# Enzyme works for all of these except `gelu`.
# See https://github.com/EnzymeAD/Enzyme.jl/issues/1671
function EnzymeRules.augmented_primal(
        cfg::EnzymeRules.ConfigWidth{1}, func::EnzymeCore.Const{typeof(gelu)},
        ::Type{<:EnzymeCore.Active}, x::EnzymeCore.Active{<:Number})
    primal = EnzymeRules.needs_primal(cfg) ? func.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(::EnzymeRules.ConfigWidth{1}, ::EnzymeCore.Const{typeof(gelu)},
        dret::EnzymeCore.Active, ::Nothing, x::EnzymeCore.Active{<:Number})
    return (dret.val * ∇gelu(x.val),)
end

function EnzymeRules.forward(
        ::EnzymeCore.Const{typeof(gelu)}, ::Type{<:EnzymeCore.Duplicated},
        x::EnzymeCore.Duplicated{<:Number})
    return EnzymeCore.Duplicated(gelu(x.val), x.dval * ∇gelu(x.val))
end

fast_act(f::F, ::Type{T}) where {F, T} = f
fast_act(f::F, ::Type{Float32}) where {F} = fast_act(f)

for (fbase, ffast) in [
    #! format: off
    (NNlib.sigmoid_fast, sigmoid_fast),
    (NNlib.softplus, softplus),
    (NNlib.logsigmoid, logsigmoid),
    (NNlib.gelu, gelu),
    (NNlib.swish, swish),
    (NNlib.lisht, lisht),
    (Base.tanh, tanh),
    (NNlib.tanh_fast, tanh_fast)
    #! format: on
]
    @eval fast_act(::typeof($fbase)) = $ffast
end

CRC.@non_differentiable fast_act(::Any...)

end
