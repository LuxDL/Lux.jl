# Entry Points
function activation!!(σ::F, x::AbstractArray) where {F}
    return activation!!(internal_operation_mode(x), is_mutable_array(x), σ, x)
end

activation!(::typeof(identity), ::AbstractArray) = nothing
function activation!(σ::F, x::AbstractArray) where {F}
    activation!(x, internal_operation_mode(x), σ, x)
    return nothing
end

activation(::typeof(identity), x::AbstractArray) = x
activation(σ::F, x::AbstractArray) where {F} = activation(internal_operation_mode(x), σ, x)

# Core Implementation
function activation!!(
        opmode::AbstractInternalArrayOpMode, ::False, σ::F, x::AbstractArray) where {F}
    return activation(opmode, σ, x)
end
@stable default_mode="disable" function activation!!(
        opmode::AbstractInternalArrayOpMode, ::True, σ::F, x::AbstractArray) where {F}
    activation!(x, opmode, σ, x)
    return x
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(activation!!),
        opmode::AbstractInternalArrayOpMode, ::True,
        σ::F, x::AbstractArray{T}) where {F, T}
    if unsafe_known(activation_intermediate_not_needed(σ, T))
        activation!(x, opmode, σ, x)
        𝒫x_no_intermediate = CRC.ProjectTo(x)
        ∇activation_no_intermediate_rrule = @closure Δ -> begin
            ∂x = ∇activation(CRC.unthunk(Δ), x, σ, NotaNumber())
            return ∂∅, ∂∅, ∂∅, ∂∅, 𝒫x_no_intermediate(∂x)
        end
        return x, ∇activation_no_intermediate_rrule
    end

    if unsafe_known(activation_has_rrule(σ, T))
        y = activation(opmode, σ, x)
        𝓟x_cached = CRC.ProjectTo(x)
        ∇activation_rrule = @closure Δ -> begin
            ∂x = ∇activation(CRC.unthunk(Δ), y, σ, x)
            return ∂∅, ∂∅, ∂∅, ∂∅, 𝓟x_cached(∂x)
        end
        return y, ∇activation_rrule
    end

    res, ∇activation_from_ad = CRC.rrule_via_ad(cfg, activation, opmode, σ, x)
    ∇activation_fallback = @closure Δ -> begin
        _, ∂opmode, ∂σ, ∂x = ∇activation_from_ad(Δ)
        return ∂∅, ∂opmode, ∂∅, ∂σ, ∂x
    end
    return res, ∇activation_fallback
end

function activation(::AbstractInternalArrayOpMode, σ::F, x::AbstractArray) where {F}
    return broadcast(σ, x)
end
@stable default_mode="disable" function activation(
        opmode::LoopedArrayOp, σ::F, x::AbstractArray{T}) where {F, T}
    RT = Core.Compiler._return_type(σ, Tuple{T})
    y = similar(x, ifelse(isconcretetype(RT), RT, T))
    activation!(y, opmode, σ, x)
    return y
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(activation),
        opmode::LoopedArrayOp, σ::F, x::AbstractArray{T}) where {F, T}
    if unsafe_known(activation_has_rrule(σ, T))
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

function activation!(
        y::AbstractArray, ::AbstractInternalArrayOpMode, σ::F, x::AbstractArray) where {F}
    broadcast!(σ, y, x)
    return
end
function activation!(y::AbstractArray, ::LoopedArrayOp, σ::F, x::AbstractArray) where {F}
    activation_simd_loop!(y, σ, x)
    return
end

function activation_simd_loop!(y::AbstractArray, σ::F, x::AbstractArray) where {F}
    @simd ivdep for I in eachindex(y, x)
        @inbounds y[I] = σ(x[I])
    end
end

# Gradient for activations
∇activation(Δ, _, ::typeof(identity), x) = Δ
function ∇activation(Δ, out, act::F, x) where {F}
    return ∇activation(internal_operation_mode((Δ, out)), Δ, out, act, x)
end
function ∇activation(::AbstractInternalArrayOpMode, Δ, out, act::F, x) where {F}
    return @. Δ * only_derivative(out, act, x)
end
@inbounds function ∇activation(::LoopedArrayOp, Δ, out, act::F, x) where {F}
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

# Switch some of the activations to use SLEEFPirates.jl if needed
function select_fastest_activation(f::F, xs...) where {F}
    return select_fastest_activation(
        f, internal_operation_mode(xs), unrolled_mapreduce(safe_eltype, promote_type, xs))
end

select_fastest_activation(f::F, ::AbstractInternalArrayOpMode, ::Type{T}) where {F, T} = f

function select_fastest_activation(f::F, ::LoopedArrayOp, ::Type{T}) where {F, T}
    return sleefpirates_fast_act(f, T)
end

CRC.@non_differentiable select_fastest_activation(::Any...)

sleefpirates_fast_act(f::F, ::Type{T}) where {F, T} = f
sleefpirates_fast_act(f::F, ::Type{Float32}) where {F} = sleefpirates_fast_act(f)
sleefpirates_fast_act(f::F) where {F} = f

CRC.@non_differentiable sleefpirates_fast_act(::Any...)
