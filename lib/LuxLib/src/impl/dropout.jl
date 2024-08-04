_dropout_shape(s, ::Colon) = size(s)
function _dropout_shape(s, dims)
    return ntuple(@closure(i->ifelse(i ∈ dims, size(s, i), 1)), ndims(s))
end

CRC.@non_differentiable _dropout_shape(::Any...)

function _alpha_dropout_kernel(noise::AbstractArray, p, x::AbstractArray, α, A, B)
    return _alpha_dropout_kernel(internal_operation_mode((noise, x)), noise, p, x, α, A, B)
end

@stable default_mode="disable" function _alpha_dropout_kernel(
        ::AbstractBroadcastOpMode, noise::AbstractArray,
        p::Real, x::AbstractArray, α::Real, A::Real, B::Real)
    A′, B′, α = eltype(x)(A), eltype(x)(B), eltype(x)(α)
    return @. muladd(ifelse(noise > p, x, α), A′, B′)
end

@stable default_mode="disable" function _alpha_dropout_kernel(
        opmode::LoopedArrayOp, noise::AbstractArray, p::Real,
        x::AbstractArray, α::Real, A::Real, B::Real)
    res = similar(x, promote_type(typeof(p), typeof(α)))
    _alpha_dropout_kernel!(res, opmode, noise, p, x, α, A, B)
    return res
end

function _alpha_dropout_kernel!(res::AbstractArray, ::LoopedArrayOp, noise::AbstractArray,
        p::Real, x::AbstractArray, α::Real, A::Real, B::Real)
    if LoopVectorization.check_args(noise, x, res)
        @tturbo for I in indices((noise, x, res))
            res[I] = ifelse(noise[I] > p, x[I], α) * A + B
        end
    else
        @batch for I in indices((noise, x, res))
            res[I] = ifelse(noise[I] > p, x[I], α) * A + B
        end
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        cfg::EnzymeRules.ConfigWidth, ::EnzymeCore.Const{typeof(_alpha_dropout_kernel!)},
        ::Type{RT}, res::EnzymeCore.Annotation{<:AbstractArray},
        opmode::EnzymeCore.Const{LoopedArrayOp}, noise::EnzymeCore.Const{<:AbstractArray},
        p::EnzymeCore.Annotation{<:Real}, x::EnzymeCore.Annotation{<:AbstractArray},
        α::EnzymeCore.Annotation{<:Real}, A::EnzymeCore.Annotation{<:Real},
        B::EnzymeCore.Annotation{<:Real}) where {RT}
    _cond = similar(noise.val, Bool)
    if LoopVectorization.check_args(noise.val, res.val, _cond)
        @tturbo for I in indices((noise.val, res.val, _cond))
            _cond[I] = noise.val[I] > p.val
            res.val[I] = ifelse(_cond[I], x.val[I], α.val) * A.val + B.val
        end
    else
        @batch for I in indices((noise.val, res.val, _cond))
            _cond[I] = noise.val[I] > p.val
            res.val[I] = ifelse(_cond[I], x.val[I], α.val) * A.val + B.val
        end
    end

    primal = EnzymeRules.needs_primal(cfg) ? res.val : nothing
    shadow = EnzymeRules.needs_shadow(cfg) ? res.dval : nothing

    return EnzymeRules.AugmentedReturn(primal, shadow, (_cond,))
end

function EnzymeRules.reverse(
        cfg::EnzymeRules.ConfigWidth, ::EnzymeCore.Const{typeof(_alpha_dropout_kernel!)},
        ::Type{RT}, (_cond,), res::EnzymeCore.Annotation{<:AbstractArray},
        opmode::EnzymeCore.Const{LoopedArrayOp}, noise::EnzymeCore.Const{<:AbstractArray},
        p::EnzymeCore.Annotation{<:Real}, x::EnzymeCore.Annotation{<:AbstractArray},
        α::EnzymeCore.Annotation{<:Real}, A::EnzymeCore.Annotation{<:Real},
        B::EnzymeCore.Annotation{<:Real}) where {RT}
    dress = res.dval
    dxs = (typeof(x) <: EnzymeCore.Const) ? dCs : x.dval

    if EnzymeRules.width(cfg) == 1
        dress = (dress,)
        dxs = (dxs,)
    end

    for (dres, dx) in zip(dress, dxs)
        if !(typeof(res) <: EnzymeCore.Const) && dres !== res.val
            if !(typeof(x) <: EnzymeCore.Const) && dx !== x.val
                if LoopVectorization.check_args(dx, dres, _cond)
                    @tturbo for I in indices((dx, dres, _cond))
                        dx[I] = _cond[I] * dres[I] * A.val
                    end
                else
                    @batch for I in indices((dx, dres, _cond))
                        dx[I] = _cond[I] * dres[I] * A.val
                    end
                end
            end

            dres .= 0
        end
    end

    # NOTE: we drop the gradients for the scalars p, A, B and alpha
    dp = typeof(p) <: EnzymeCore.Const ? nothing : zero(p.val)
    dα = typeof(α) <: EnzymeCore.Const ? nothing : zero(α.val)
    dA = typeof(A) <: EnzymeCore.Const ? nothing : zero(A.val)
    dB = typeof(B) <: EnzymeCore.Const ? nothing : zero(B.val)

    return (nothing, nothing, nothing, dp, nothing, dα, dA, dB)
end

# We intentionally drop the gradients for p, A, B and alpha
function CRC.rrule(::typeof(_alpha_dropout_kernel), ::LoopedArrayOp, noise::AbstractArray,
        p::Real, x::AbstractArray, α::Real, A::Real, B::Real)
    _cond = similar(noise, Bool)
    y = similar(x, promote_type(typeof(p), typeof(α), typeof(A), typeof(B), eltype(x)))
    if LoopVectorization.check_args(noise, x, y, _cond)
        @tturbo for I in indices((noise, x, y, _cond))
            _cond[I] = noise[I] > p
            y[I] = ifelse(_cond[I], x[I], α) * A + B
        end
    else
        @batch for I in indices((noise, x, y, _cond))
            _cond[I] = noise[I] > p
            y[I] = ifelse(_cond[I], x[I], α) * A + B
        end
    end

    proj_x = CRC.ProjectTo(x)
    _∇alpha_dropout_kernel = let _cond = _cond, proj_x = proj_x, x = x
        Δ -> begin
            ∂x = similar(x)
            if LoopVectorization.check_args(∂x, _cond, Δ)
                @tturbo for I in indices((∂x, _cond, Δ))
                    ∂x[I] = _cond[I] * Δ[I] * A
                end
            else
                @batch for I in indices((∂x, _cond, Δ))
                    ∂x[I] = _cond[I] * Δ[I] * A
                end
            end
            return (ntuple(Returns(∂∅), 4)..., proj_x(∂x), ntuple(Returns(∂∅), 3)...)
        end
    end

    return y, _∇alpha_dropout_kernel
end

function CRC.rrule(::typeof(_alpha_dropout_kernel), ::AbstractBroadcastOpMode,
        noise::AbstractArray, p::Real, x::AbstractArray, α::Real, A::Real, B::Real)
    _cond = broadcast(>, noise, p)
    y = @. ifelse(_cond, x, α) * A + B

    proj_x = CRC.ProjectTo(x)
    _∇alpha_dropout_kernel = @closure Δ -> begin
        ∂x = proj_x(@.(Δ*_cond*A))
        return (ntuple(Returns(∂∅), 4)..., ∂x, ntuple(Returns(∂∅), 3)...)
    end

    return y, _∇alpha_dropout_kernel
end

_dropout_fptype(x) = float(real(remove_tracking(eltype(x))))

CRC.@non_differentiable _dropout_fptype(::Any...)

@stable default_mode="disable" function _alpha_dropout_noise(rng, x)
    rng = LuxCore.replicate(rng)
    noise = similar(x, _dropout_fptype(x))
    rand!(rng, noise)
    return noise, rng
end

CRC.@non_differentiable _alpha_dropout_noise(::Any...)
EnzymeRules.inactive_noinl(::typeof(_alpha_dropout_noise), ::Any...) = nothing

@stable default_mode="disable" function _generate_dropout_mask(
        rng::AbstractRNG, x, p, invp; dims)
    rng = LuxCore.replicate(rng)
    y = similar(x, _dropout_fptype(x), _dropout_shape(x, dims))
    rand!(rng, y)
    opmode = internal_operation_mode(y)
    if opmode isa LoopedArrayOp
        if LoopVectorization.check_args(y)
            @tturbo for I in indices(y)
                y[I] = (y[I] > p) * invp
            end
        else
            @batch for I in indices(y)
                y[I] = (y[I] > p) * invp
            end
        end
    else
        @. y = (y > p) * invp
    end
    return y, rng
end

CRC.@non_differentiable _generate_dropout_mask(::Any...)
EnzymeRules.inactive(::typeof(_generate_dropout_mask), ::Any...) = nothing

# dropout -- force don't compute some gradients
@stable default_mode="disable" function __dropout_dot_mul(
        x::AbstractArray, mask::AbstractArray)
    return x .* mask
end

function CRC.rrule(::typeof(__dropout_dot_mul), x::AbstractArray, mask::AbstractArray)
    res = __dropout_dot_mul(x, mask)  # size(res) == size(x)
    proj_x = CRC.ProjectTo(x)
    ∇dropout_dot_mul = @closure Δ -> begin
        ∂x = proj_x(__dropout_dot_mul(Δ, mask))
        return ∂∅, ∂x, ∂∅
    end
    return res, ∇dropout_dot_mul
end
