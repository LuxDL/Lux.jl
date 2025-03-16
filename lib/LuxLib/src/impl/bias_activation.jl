# Entry Points
bias_activation(::typeof(identity), x::AbstractVector, ::Nothing) = x
for bType in (Nothing, AbstractVector)
    @eval function bias_activation(σ::F, x::AbstractVector, bias::$(bType)) where {F}
        return vec(bias_activation(σ, expand_batchdim(x), bias))
    end
end

bias_activation(::typeof(identity), x::AbstractArray, ::Nothing) = x
function bias_activation(σ::F, x::AbstractArray{xT,N}, ::Nothing) where {F,N,xT}
    return activation(σ, x)
end
function bias_activation(
    σ::F, x::AbstractArray{xT,N}, bias::AbstractVector{bT}
) where {F,N,xT,bT}
    return bias_activation(internal_operation_mode((x, bias)), σ, x, bias)
end

## General Implementation
function bias_activation(
    ::GenericBroadcastOp,
    ::typeof(identity),
    x::AbstractArray{T1,N},
    bias::AbstractVector{T2},
) where {N,T1,T2}
    return x .+ reshape_bias(x, bias)
end
function bias_activation(
    ::GenericBroadcastOp, σ::F, x::AbstractArray{T1,N}, bias::AbstractVector
) where {F,N,T1}
    return σ.(x .+ reshape_bias(x, bias))
end

function bias_activation(
    ::AbstractInternalArrayOpMode,
    ::typeof(identity),
    x::AbstractArray{xT,N},
    bias::AbstractVector,
) where {N,xT}
    return x .+ reshape_bias(x, bias)
end
function bias_activation(
    ::AbstractInternalArrayOpMode, σ::F, x::AbstractArray{xT,N}, bias::AbstractVector
) where {F,N,xT}
    return broadcast(σ ∘ +, x, reshape_bias(x, bias))
end

# Prevent ambiguity
@stable default_mode = "disable" function bias_activation(
    opmode::LoopedArrayOp, ::typeof(identity), x::AbstractArray{xT,N}, bias::AbstractVector
) where {N,xT}
    y = similar(x, concrete_bias_act_output_eltype(identity, x, bias))
    bias_activation!(y, opmode, identity, x, bias)
    return y
end
@stable default_mode = "disable" function bias_activation(
    opmode::LoopedArrayOp, σ::F, x::AbstractArray{xT,N}, bias::AbstractVector
) where {F,N,xT}
    y = similar(x, concrete_bias_act_output_eltype(σ, x, bias))
    bias_activation!(y, opmode, σ, x, bias)
    return y
end

function CRC.rrule(
    cfg::RuleConfig{>:HasReverseMode},
    ::typeof(bias_activation),
    opmode::AbstractInternalArrayOpMode,
    σ::F,
    x::AbstractArray{xT,N},
    bias::AbstractVector,
) where {F,N,xT}
    T = concrete_bias_act_output_eltype(σ, x, bias)

    if unsafe_known(activation_intermediate_not_needed(σ, T))
        y = bias_activation(opmode, σ, x, bias)
        ∇bias_activation_no_intermediate = @closure Δ -> begin
            ∂x = CRC.ProjectTo(x)(∇activation(recursive_unthunk(Δ), y, σ, NotaNumber()))
            ∂b = CRC.@thunk CRC.ProjectTo(bias)(∇bias_add(bias, ∂x))
            return ∂∅, ∂∅, ∂∅, ∂x, ∂b
        end
        return y, ∇bias_activation_no_intermediate
    end

    if unsafe_known(activation_has_rrule(σ, T))
        tmp = similar(x, T)
        bias_add!(tmp, opmode, x, bias)
        y = activation(opmode, σ, tmp)
        ∇bias_activation_rrule = @closure Δ -> begin
            ∂x = CRC.ProjectTo(x)(∇activation(recursive_unthunk(Δ), y, σ, tmp))
            ∂b = CRC.@thunk CRC.ProjectTo(bias)(∇bias_add(bias, ∂x))
            return ∂∅, ∂∅, ∂∅, ∂x, ∂b
        end
        return y, ∇bias_activation_rrule
    end

    y, ∇broadcast = CRC.rrule_via_ad(
        cfg, broadcast_bias_activation_generic, σ, x, reshape_bias(x, bias)
    )
    ∇bias_activation_rrule = @closure Δ -> begin
        _, _, ∂x, ∂bias = ∇broadcast(Δ)
        return ∂∅, ∂∅, ∂∅, CRC.ProjectTo(x)(∂x), CRC.ProjectTo(bias)(vec(∂bias))
    end
    return y, ∇bias_activation_rrule
end

@inline broadcast_bias_activation_generic(σ::F, x, b) where {F} = σ.(x .+ b)

bias_activation!!(::typeof(identity), x::AbstractVector, ::Nothing) = x
for bType in (Nothing, AbstractVector)
    @eval function bias_activation!!(σ::F, x::AbstractVector, bias::$(bType)) where {F}
        return vec(bias_activation!!(σ, expand_batchdim(x), bias))
    end
end

bias_activation!!(::typeof(identity), x::AbstractArray, ::Nothing) = x
function bias_activation!!(σ::F, x::AbstractArray{xT,N}, ::Nothing) where {F,N,xT}
    return activation!!(σ, x)
end
function bias_activation!!(
    σ::F, x::AbstractArray{xT,N}, bias::AbstractVector
) where {F,N,xT}
    return bias_activation!!(
        internal_operation_mode((x, bias)), is_mutable_array(x), σ, x, bias
    )
end

function bias_activation!!(
    opmode::AbstractInternalArrayOpMode,
    ::False,
    σ::F,
    x::AbstractArray{xT,N},
    bias::AbstractVector,
) where {F,N,xT}
    return bias_activation(opmode, σ, x, bias)
end

function bias_activation!!(
    opmode::GenericBroadcastOp, ::True, σ::F, x::AbstractArray{xT,N}, bias::AbstractVector
) where {F,N,xT}
    return bias_activation(opmode, σ, x, bias)
end

@stable default_mode = "disable" function bias_activation!!(
    opmode::AbstractInternalArrayOpMode,
    ::True,
    σ::F,
    x::AbstractArray{xT,N},
    bias::AbstractVector,
) where {F,N,xT}
    bias_activation!(x, opmode, σ, x, bias)
    return x
end

function CRC.rrule(
    cfg::RuleConfig{>:HasReverseMode},
    ::typeof(bias_activation!!),
    opmode::AbstractInternalArrayOpMode,
    ::True,
    σ::F,
    x::AbstractArray{xT,N},
    bias::AbstractVector,
) where {F,N,xT}
    T = concrete_bias_act_output_eltype(σ, x, bias)

    if unsafe_known(activation_intermediate_not_needed(σ, T))
        bias_activation!(x, opmode, σ, x, bias)
        ∇bias_activation_no_intermediate = @closure Δ -> begin
            ∂x = CRC.ProjectTo(x)(∇activation(recursive_unthunk(Δ), x, σ, NotaNumber()))
            ∂b = CRC.@thunk CRC.ProjectTo(bias)(∇bias_add(bias, ∂x))
            return ∂∅, ∂∅, ∂∅, ∂∅, ∂x, ∂b
        end
        return x, ∇bias_activation_no_intermediate
    end

    if unsafe_known(activation_has_rrule(σ, T))
        y, tmp = bias_activation_cached!!(σ, x, bias)
        ∇bias_activation_rrule = @closure Δ -> begin
            ∂x = CRC.ProjectTo(x)(∇activation(recursive_unthunk(Δ), y, σ, tmp))
            ∂b = CRC.@thunk CRC.ProjectTo(bias)(∇bias_add(bias, ∂x))
            return ∂∅, ∂∅, ∂∅, ∂∅, ∂x, ∂b
        end
        return y, ∇bias_activation_rrule
    end

    res, ∇bias_activation_from_ad = CRC.rrule_via_ad(
        cfg, bias_activation, opmode, σ, x, bias
    )
    ∇bias_activation_fallback = @closure Δ -> begin
        _, _, _, ∂x, ∂b = ∇bias_activation_from_ad(Δ)
        return ∂∅, ∂∅, ∂∅, ∂∅, CRC.ProjectTo(x)(∂x), CRC.ProjectTo(bias)(∂b)
    end
    return res, ∇bias_activation_fallback
end

# Core Implementation
function bias_activation!(
    y::AbstractArray{yT,N},
    opmode::AbstractInternalArrayOpMode,
    σ::F,
    x::AbstractArray{xT,N},
    ::Nothing,
) where {F,N,xT,yT}
    activation!(y, opmode, σ, x)
    return nothing
end

function bias_activation!(
    y::AbstractArray{yT,N},
    opmode::AbstractInternalArrayOpMode,
    σ::F,
    x::AbstractArray{xT,N},
    bias::AbstractVector,
) where {F,N,xT,yT}
    if σ === identity
        bias_add!(y, opmode, x, bias)
    else
        broadcast!(σ ∘ +, y, x, reshape_bias(x, bias))
    end
    return nothing
end

function bias_activation!(
    y::AbstractArray{yT,N},
    ::LoopedArrayOp,
    σ::F,
    x::AbstractArray{xT,N},
    bias::AbstractVector,
) where {F,N,xT,yT}
    bias_activation_cpu!(
        reshape(y, flattened_bias_dims(y), size(y, N - 1), size(y, N)),
        fuse_cpu_activation(σ),
        σ,
        reshape(x, flattened_bias_dims(x), size(x, N - 1), size(x, N)),
        bias,
    )
    return nothing
end

function bias_activation_cpu!(
    y::AbstractArray{yT,3}, ::True, σ::F, x::AbstractArray{xT,3}, bias::AbstractVector
) where {F,xT,yT}
    bias_activation_simd_loop!(y, σ, x, bias)
    return nothing
end

function bias_activation_cpu!(
    y::AbstractArray{yT,3}, ::False, σ::F, x::AbstractArray{xT,3}, bias::AbstractVector
) where {F,xT,yT}
    bias_activation_simd_loop!(y, σ, x, bias)
    return nothing
end

function bias_activation_simd_loop!(
    y::AbstractArray{yT,3}, σ::F, x::AbstractArray{xT,3}, bias::AbstractVector
) where {F,xT,yT}
    if size(y, 1) == 1
        for K in axes(x, 3)
            @simd ivdep for J in axes(x, 2)
                @inbounds y[1, J, K] = σ(x[1, J, K] + bias[J])
            end
        end
    else
        for K in axes(x, 3), J in axes(x, 2)
            @simd ivdep for I in axes(x, 1)
                @inbounds y[I, J, K] = σ(x[I, J, K] + bias[J])
            end
        end
    end
    return nothing
end

function bias_add!(
    y::AbstractArray{yT,N},
    ::AbstractInternalArrayOpMode,
    x::AbstractArray{xT,N},
    bias::AbstractVector,
) where {N,xT,yT}
    broadcast!(+, y, x, reshape_bias(x, bias))
    return nothing
end

function bias_add!(
    y::AbstractArray{yT,N}, ::LoopedArrayOp, x::AbstractArray{xT,N}, bias::AbstractVector
) where {N,xT,yT}
    bias_add_loop!(
        reshape(y, flattened_bias_dims(y), size(y, N - 1), size(y, N)),
        reshape(x, flattened_bias_dims(x), size(x, N - 1), size(x, N)),
        bias,
    )
    return nothing
end

function bias_add_loop!(
    y::AbstractArray{yT,3}, x::AbstractArray{xT,3}, bias::AbstractVector
) where {xT,yT}
    return if size(y, 1) == 1
        for K in axes(x, 3)
            @simd ivdep for J in axes(x, 2)
                @inbounds y[1, J, K] = x[1, J, K] + bias[J]
            end
        end
    else
        for K in axes(x, 3), J in axes(x, 2)
            @simd ivdep for I in axes(y, 1)
                @inbounds y[I, J, K] = x[I, J, K] + bias[J]
            end
        end
    end
end

# Some helper functions for the rrule
function bias_activation_cached!!(
    σ::F, x::AbstractArray{xT,N}, bias::Optional{<:AbstractVector}
) where {F,N,xT}
    @assert σ !== identity
    bias === nothing && return activation(σ, x), x
    return bias_activation_cached!!(
        internal_operation_mode((x, bias)), is_mutable_array(x), σ, x, bias
    )
end

function bias_activation_cached!!(
    ::AbstractInternalArrayOpMode,
    ::False,
    σ::F,
    x::AbstractArray{xT,N},
    bias::Optional{<:AbstractVector},
) where {F,N,xT}
    y = broadcast(+, x, reshape_bias(x, bias))
    return activation(σ, y), y
end

function bias_activation_cached!!(
    ::AbstractInternalArrayOpMode,
    ::True,
    σ::F,
    x::AbstractArray{xT,N},
    bias::Optional{<:AbstractVector},
) where {F,N,xT}
    broadcast!(+, x, x, reshape_bias(x, bias))
    return activation(σ, x), x
end

function bias_activation_cached!!(
    ::LoopedArrayOp, ::True, σ::F, x::AbstractArray{xT,N}, bias::Optional{<:AbstractVector}
) where {F,N,xT}
    x′ = reshape(x, flattened_bias_dims(x), size(x, N - 1), size(x, N))
    bias_add_loop!(x′, x′, bias)
    x′′ = reshape(x′, size(x))
    return activation(σ, x′′), x′′
end

flattened_bias_dims(x::AbstractArray{T,N}) where {T,N} = prod(size(x)[1:(N - 2)]; init=1)

CRC.@non_differentiable flattened_bias_dims(::Any...)
