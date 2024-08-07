# Entry Points
bias_activation(::typeof(identity), x::AbstractVector{<:Number}, ::Nothing) = x
for bType in (Nothing, AbstractVector{<:Number})
    @eval function bias_activation(
            σ::F, x::AbstractVector{<:Number}, bias::$(bType)) where {F}
        return vec(bias_activation(σ, reshape(x, :, 1), bias))
    end
end

bias_activation(::typeof(identity), x::AbstractArray{<:Number}, ::Nothing) = x
function bias_activation(σ::F, x::AbstractArray{<:Number, N}, ::Nothing) where {F, N}
    return activation(σ, x)
end
function bias_activation(
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    return bias_activation(internal_operation_mode((x, bias)), σ, x, bias)
end

## General Implementation
function bias_activation(::AbstractInternalArrayOpMode, ::typeof(identity),
        x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {N}
    return broadcast(+, x, reshape_bias(x, bias))
end
function bias_activation(::AbstractInternalArrayOpMode, σ::F, x::AbstractArray{<:Number, N},
        bias::AbstractVector{<:Number}) where {F, N}
    return broadcast(σ ∘ +, x, reshape_bias(x, bias))
end

@stable default_mode="disable" function bias_activation(
        opmode::LoopedArrayOp, σ::F, x::AbstractArray{<:Number, N},
        bias::AbstractVector{<:Number}) where {F, N}
    y = similar(x, Utils.concrete_bias_act_output_eltype(σ, x, bias))
    bias_activation!(y, opmode, σ, x, bias)
    return y
end

function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(bias_activation), opmode::LoopedArrayOp,
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    T = Utils.concrete_bias_act_output_eltype(σ, x, bias)

    if Utils.known(Traits.activation_intermediate_not_needed(σ, T))
        y = bias_activation(opmode, σ, x, bias)
        𝒫x_no_intermediate = CRC.ProjectTo(x)
        𝒫bias_no_intermediate = CRC.ProjectTo(bias)
        ∇bias_activation_no_intermediate = @closure Δ -> begin
            ∂x = ∇activation(CRC.unthunk(Δ), y, σ, Utils.NotaNumber())
            ∂b = ∇bias_add(bias, ∂x)
            return ∂∅, ∂∅, ∂∅, 𝒫x_no_intermediate(∂x), 𝒫bias_no_intermediate(∂b)
        end
        return y, ∇bias_activation_no_intermediate
    end

    if Utils.known(Traits.activation_has_rrule(σ, T))
        tmp = similar(x, T)
        bias_activation!(tmp, opmode, σ, x, bias)
        y = activation(opmode, σ, x)
        𝓟x_cached = CRC.ProjectTo(x)
        𝓟bias_cached = CRC.ProjectTo(bias)
        ∇bias_activation_rrule = @closure Δ -> begin
            ∂x = ∇activation(CRC.unthunk(Δ), y, σ, tmp)
            ∂b = ∇bias_add(bias, ∂x)
            return ∂∅, ∂∅, ∂∅, 𝓟x_cached(∂x), 𝓟bias_cached(∂b)
        end
        return y, ∇bias_activation_rrule
    end

    return CRC.rrule_via_ad(cfg, bias_activation, GenericBroadcastOp(), σ, x, bias)
end

bias_activation!!(::typeof(identity), x::AbstractVector{<:Number}, ::Nothing) = x
for bType in (Nothing, AbstractVector{<:Number})
    @eval function bias_activation!!(
            σ::F, x::AbstractVector{<:Number}, bias::$(bType)) where {F}
        return vec(bias_activation!!(σ, reshape(x, :, 1), bias))
    end
end

bias_activation!!(::typeof(identity), x::AbstractArray{<:Number}, ::Nothing) = x
function bias_activation!!(σ::F, x::AbstractArray{<:Number, N}, ::Nothing) where {F, N}
    return activation!!(σ, x)
end
function bias_activation!!(
        σ::F, x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    return bias_activation!!(
        internal_operation_mode((x, bias)), Traits.is_mutable_array(x), σ, x, bias)
end

function bias_activation!!(opmode::AbstractInternalArrayOpMode, ::False, σ::F,
        x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    return bias_activation(opmode, σ, x, bias)
end

@stable default_mode="disable" function bias_activation!!(
        opmode::AbstractInternalArrayOpMode, ::True, σ::F,
        x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    bias_activation!(x, opmode, σ, x, bias)
    return x
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(bias_activation!!),
        opmode::AbstractInternalArrayOpMode, ::True, σ::F,
        x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    T = Utils.concrete_bias_act_output_eltype(σ, x, bias)

    if Utils.known(Traits.activation_intermediate_not_needed(σ, T))
        bias_activation!(x, opmode, σ, x, bias)
        𝒫x_no_intermediate = CRC.ProjectTo(x)
        𝒫bias_no_intermediate = CRC.ProjectTo(bias)
        ∇bias_activation_no_intermediate = @closure Δ -> begin
            ∂x = ∇activation(CRC.unthunk(Δ), x, σ, Utils.NotaNumber())
            ∂b = ∇bias_add(bias, ∂x)
            return ∂∅, ∂∅, ∂∅, 𝒫x_no_intermediate(∂x), 𝒫bias_no_intermediate(∂b)
        end
        return x, ∇bias_activation_no_intermediate
    end

    if Utils.known(Traits.activation_has_rrule(σ, T))
        y, tmp = bias_activation_cached!!(σ, x, bias)
        𝓟x_cached = CRC.ProjectTo(x)
        𝓟bias_cached = CRC.ProjectTo(bias)
        ∇bias_activation_rrule = @closure Δ -> begin
            ∂x = ∇activation(CRC.unthunk(Δ), y, σ, tmp)
            ∂b = ∇bias_add(bias, ∂x)
            return ∂∅, ∂∅, ∂∅, 𝓟x_cached(∂x), 𝓟bias_cached(∂b)
        end
        return y, ∇bias_activation_rrule
    end

    res, ∇bias_activation_from_ad = CRC.rrule_via_ad(
        cfg, bias_activation, opmode, σ, x, bias)
    ∇bias_activation_fallback = @closure Δ -> begin
        _, ∂opmode, ∂σ, ∂x, ∂b = ∇bias_activation_from_ad(Δ)
        return ∂∅, ∂opmode, ∂∅, ∂σ, ∂x, ∂b
    end
    return res, ∇bias_activation_fallback
end

# Core Implementation
function bias_activation!(
        y::AbstractArray{<:Number, N}, opmode::AbstractInternalArrayOpMode, σ::F,
        x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    if σ === identity
        bias_add!(y, opmode, x, bias)
    else
        broadcast!(σ ∘ +, y, x, reshape_bias(x, bias))
    end
    return
end

function bias_activation!(y::AbstractArray{<:Number, N}, opmode::LoopedArrayOp, σ::F,
        x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {F, N}
    bias_add!(y, opmode, x, bias)
    activation!(y, opmode, σ, y)
    return
end

function bias_add!(y::AbstractArray{<:Number, N}, ::AbstractInternalArrayOpMode,
        x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {N}
    broadcast!(+, y, x, reshape_bias(x, bias))
    return
end

function bias_add!(y::AbstractArray{<:Number, N}, ::LoopedArrayOp,
        x::AbstractArray{<:Number, N}, bias::AbstractVector{<:Number}) where {N}
    y_ = reshape(y, :, size(y, N - 1), size(y, N))
    x_ = reshape(x, :, size(x, N - 1), size(x, N))
    if LV.check_args(y_, x_, bias)
        @tturbo for K in indices(x_, 3),
            J in indices((x_, bias), (2, 1)),
            I in indices(y_, 1)

            y_[I, J, K] = x_[I, J, K] + bias[J]
        end
    else
        @batch for K in indices(x_, 3), J in indices((x_, bias), (2, 1))
            @simd ivdep for I in indices(y_, 1)
                y_[I, J, K] = x_[I, J, K] + bias[J]
            end
        end
    end
end

function EnzymeRules.augmented_primal(
        cfg::EnzymeRules.ConfigWidth, ::EnzymeCore.Const{typeof(bias_add!)},
        ::Type{EnzymeCore.Const{Nothing}}, y::EnzymeCore.Duplicated{<:AbstractArray},
        opmode::EnzymeCore.Const{LoopedArrayOp}, x::EnzymeCore.Duplicated{<:AbstractArray},
        bias::EnzymeCore.Duplicated{<:AbstractVector})
    if typeof(y) <: EnzymeCore.Duplicated || typeof(y) <: EnzymeCore.BatchDuplicated
        bias_add!(y.val, opmode.val, x.val, bias.val)
    end
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        cfg::EnzymeRules.ConfigWidth, ::EnzymeCore.Const{typeof(bias_add!)},
        ::Type{EnzymeCore.Const{Nothing}}, ::Nothing,
        y::EnzymeCore.Duplicated{<:AbstractArray},
        opmode::EnzymeCore.Const{LoopedArrayOp}, x::EnzymeCore.Duplicated{<:AbstractArray},
        bias::EnzymeCore.Duplicated{<:AbstractVector})
    dys = y.dval
    dxs = x.dval
    dbs = bias.dval

    if EnzymeRules.width(cfg) == 1
        dys = (dys,)
        dxs = (dxs,)
        dbs = (dbs,)
    end

    for (dy, dx, db) in zip(dys, dxs, dbs)
        if !(typeof(y) <: EnzymeCore.Const) && dy !== y.val
            if !(typeof(x) <: EnzymeCore.Const) && dx !== x.val && dx !== dy
                copyto!(dx, dy)
            end

            if !(typeof(bias) <: EnzymeCore.Const) && db !== bias.val
                dy_ = reshape(dy, :, size(dy, N - 1), size(dy, N))
                if LV.check_args(dy_, bias)
                    @turbo for K in indices(dy_, 3),
                        J in indices((dy_, db), (2, 1)),
                        I in indices(dy_, 1)

                        db[J] += dy_[I, J, K]
                    end
                else
                    db_ = reshape(db, 1, :, 1)
                    sum!(db_, dy_)
                end
            end

            dx !== dy && fill!(dy, false)
        end
    end

    return nothing, nothing, nothing, nothing
end

# Soem helper functions for the rrule
function bias_activation_cached!!(σ::F, x::AbstractArray{<:Number, N},
        bias::Optional{<:AbstractVector{<:Number}}) where {F, N}
    @assert σ !== identity
    bias === nothing && return activation(σ, x), x
    return bias_activation_cached!!(
        internal_operation_mode((x, bias)), Traits.is_mutable_array(x), σ, x, bias)
end

function bias_activation_cached!!(
        ::AbstractInternalArrayOpMode, ::False, σ::F, x::AbstractArray{<:Number, N},
        bias::Optional{<:AbstractVector{<:Number}}) where {F, N}
    y = broadcast(+, x, reshape_bias(x, bias))
    return activation(σ, y), y
end

function bias_activation_cached!!(
        ::AbstractInternalArrayOpMode, ::True, σ::F, x::AbstractArray{<:Number, N},
        bias::Optional{<:AbstractVector{<:Number}}) where {F, N}
    broadcast!(+, x, x, reshape_bias(x, bias))
    return activation(σ, x), x
end

function bias_activation_cached!!(
        opmode::LoopedArrayOp, ::False, σ::F, x::AbstractArray{<:Number, N},
        bias::Optional{<:AbstractVector{<:Number}}) where {F, N}
    x_ = reshape(x, :, size(x, N - 1), size(x, N))
    if LV.check_args(x_, bias)
        @tturbo for K in indices(x_, 3),
            J in indices((x_, bias), (2, 1)),
            I in indices(x_, 1)

            x_[I, J, K] = x_[I, J, K] + bias[J]
        end
    else
        @batch for K in indices(x_, 3), J in indices((x_, bias), (2, 1))
            @simd ivdep for I in indices(x_, 1)
                x_[I, J, K] = x_[I, J, K] + bias[J]
            end
        end
    end
    return activation(σ, x), x
end
