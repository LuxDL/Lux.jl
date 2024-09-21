# Entry Points
## dropout
function dropout(rng::AbstractRNG, x::AbstractArray, p::T, ::True, invp::T, dims) where {T}
    mask, rngₙ = generate_dropout_mask(rng, x, p, invp, dims)
    return dropout_dot_mul(x, mask), mask, rngₙ
end

dropout(rng::AbstractRNG, x::AbstractArray, ::T, ::False, ::T, dims) where {T} = (x, x, rng)

function dropout(rng::AbstractRNG, x::AbstractArray, ::AbstractArray, p::T,
        training::StaticBool, ::True, invp::T, dims) where {T}
    return dropout(rng, x, p, training, invp, dims)
end

function dropout(rng::AbstractRNG, x::AbstractArray, mask::AbstractArray,
        ::T, ::True, ::False, invp::T, dims) where {T}
    check_dropout_mask_shape_mismatch(x, mask, dims)
    return dropout_dot_mul(x, mask), mask, rng
end

function dropout(rng::AbstractRNG, x::AbstractArray, mask::AbstractArray,
        ::T, ::False, ::False, invp::T, dims) where {T}
    return x, mask, rng
end

function check_dropout_mask_shape_mismatch(x::AbstractArray, mask::AbstractArray, dims)
    @assert dropout_shape(x, dims)==size(mask) "`mask` is not of the same size as `LuxLib.dropout_shape(x, dims)`."
    return nothing
end

CRC.@non_differentiable check_dropout_mask_shape_mismatch(::Any...)

## alpha_dropout
function alpha_dropout(rng::AbstractRNG, x::AbstractArray{T}, p, ::True) where {T}
    α = T(-1.7580993408473766)
    A = T(inv(sqrt((1 - p) * (1 + p * α^2))))
    B = T(-A * α * p)
    return alpha_dropout(rng, x, p, True(), α, A, B)
end

function alpha_dropout(rng::AbstractRNG, x::AbstractArray{T}, p, ::False) where {T}
    return alpha_dropout(rng, x, p, False(), T(0), T(0), T(0))
end

function alpha_dropout(rng::AbstractRNG, x::AbstractArray{T}, p, ::True, α, A, B) where {T}
    noise, rngₙ = generate_alpha_dropout_noise(rng, x)
    return alpha_dropout(noise, p, x, α, A, B), rngₙ
end

alpha_dropout(rng::AbstractRNG, x::AbstractArray{T}, p, ::False, α, A, B) where {T} = x, rng

# Core Implementation
dropout_shape(s, ::Colon) = size(s)
function dropout_shape(s, dims)
    return ntuple(@closure(i->ifelse(i ∈ dims, size(s, i), 1)), ndims(s))
end

CRC.@non_differentiable dropout_shape(::Any...)

function alpha_dropout(noise::AbstractArray, p, x::AbstractArray, α, A, B)
    return alpha_dropout(internal_operation_mode((noise, x)), noise, p, x, α, A, B)
end

@stable default_mode="disable" function alpha_dropout(
        ::AbstractInternalArrayOpMode, noise::AbstractArray, p::Real,
        x::AbstractArray{T}, α::Real, A::Real, B::Real) where {T}
    A′, B′, α = T(A), T(B), T(α)
    return @. muladd(ifelse(noise > p, x, α), A′, B′)
end

@stable default_mode="disable" function alpha_dropout(
        opmode::LoopedArrayOp, noise::AbstractArray, p::Real,
        x::AbstractArray, α::Real, A::Real, B::Real)
    res = similar(x, promote_type(typeof(p), typeof(α)))
    alpha_dropout!(res, opmode, noise, p, x, α, A, B)
    return res
end

function CRC.rrule(::typeof(alpha_dropout), ::LoopedArrayOp, noise::AbstractArray,
        p::Real, x::AbstractArray, α::Real, A::Real, B::Real)
    cond = similar(noise, Bool)
    y = similar(x, promote_type(typeof(p), typeof(α), typeof(A), typeof(B), eltype(x)))
    if LV.check_args(noise, x, y, cond)
        @tturbo for I in indices((noise, x, y, cond))
            cond[I] = noise[I] > p
            y[I] = ifelse(cond[I], x[I], α) * A + B
        end
    else
        @batch for I in indices((noise, x, y, cond))
            cond[I] = noise[I] > p
            y[I] = ifelse(cond[I], x[I], α) * A + B
        end
    end

    ∇alpha_dropout = let cond = cond, 𝒫x = CRC.ProjectTo(x), x = x
        Δ -> begin
            ∂x = similar(x)
            if LV.check_args(∂x, cond, Δ)
                @tturbo for I in indices((∂x, cond, Δ))
                    ∂x[I] = cond[I] * Δ[I] * A
                end
            else
                @batch for I in indices((∂x, cond, Δ))
                    ∂x[I] = cond[I] * Δ[I] * A
                end
            end
            return (ntuple(Returns(∂∅), 4)..., 𝒫x(∂x), ntuple(Returns(∂∅), 3)...)
        end
    end

    return y, ∇alpha_dropout
end

function CRC.rrule(::typeof(alpha_dropout), ::AbstractInternalArrayOpMode,
        noise::AbstractArray, p::Real, x::AbstractArray, α::Real, A::Real, B::Real)
    cond = noise .> p
    y = @. ifelse(cond, x, α) * A + B

    𝒫x = CRC.ProjectTo(x)
    ∇alpha_dropout = @closure Δ -> begin
        ∂x = 𝒫x(Δ .* cond .* A)
        return (ntuple(Returns(∂∅), 4)..., ∂x, ntuple(Returns(∂∅), 3)...)
    end

    return y, ∇alpha_dropout
end

function alpha_dropout!(res::AbstractArray, ::LoopedArrayOp, noise::AbstractArray,
        p::Real, x::AbstractArray, α::Real, A::Real, B::Real)
    if LV.check_args(noise, x, res)
        @tturbo for I in indices((noise, x, res))
            res[I] = ifelse(noise[I] > p, x[I], α) * A + B
        end
    else
        @batch for I in indices((noise, x, res))
            res[I] = ifelse(noise[I] > p, x[I], α) * A + B
        end
    end
end

function alpha_dropout_simd_loop!(
        res::AbstractArray{T}, ::LoopedArrayOp, noise::AbstractArray{T},
        p::Real, x::AbstractArray{T}, α::Real, A::Real, B::Real) where {T}
    @simd ivdep for I in indices((noise, x, res))
        res[I] = ifelse(noise[I] > p, x[I], α) * A + B
    end
end

@enzyme_alternative alpha_dropout! alpha_dropout_simd_loop!

dropout_fptype(x) = float(real(remove_tracking(eltype(x))))

CRC.@non_differentiable dropout_fptype(::Any...)

@stable default_mode="disable" function generate_alpha_dropout_noise(rng::AbstractRNG, x)
    rng = LuxCore.replicate(rng)
    noise = similar(x, dropout_fptype(x))
    rand!(rng, noise)
    return noise, rng
end

CRC.@non_differentiable generate_alpha_dropout_noise(::Any...)

@stable default_mode="disable" function generate_dropout_mask(
        rng::AbstractRNG, x, p, invp, dims)
    rng = LuxCore.replicate(rng)
    y = similar(remove_tracking(x), dropout_fptype(x), dropout_shape(x, dims))
    rand!(rng, y)
    generate_dropout_mask!(y, internal_operation_mode(y), p, invp)
    return y, rng
end

CRC.@non_differentiable generate_dropout_mask(::Any...)

function generate_dropout_mask!(y::AbstractArray, ::LoopedArrayOp, p, invp)
    generate_dropout_mask_loop!(y, p, invp)
    return
end

function generate_dropout_mask_loop!(y::AbstractArray, p, invp)
    if LV.check_args(y)
        @tturbo for I in indices(y)
            y[I] = (y[I] > p) * invp
        end
    else
        @batch for I in indices(y)
            y[I] = (y[I] > p) * invp
        end
    end
end

function generate_dropout_mask_simd_loop!(y::AbstractArray{T}, p, invp) where {T}
    p, invp = T(p), T(invp)
    @simd ivdep for I in indices(y)
        y[I] = (y[I] > p) * invp
    end
end

@enzyme_alternative generate_dropout_mask_loop! generate_dropout_mask_simd_loop!

function generate_dropout_mask!(
        y::AbstractArray{T}, ::AbstractInternalArrayOpMode, p, invp) where {T}
    p, invp = T(p), T(invp)
    @. y = (y > p) * invp
    return
end

dropout_dot_mul(x::AbstractArray, mask::AbstractArray) = x .* mask

function CRC.rrule(::typeof(dropout_dot_mul), x::AbstractArray, mask::AbstractArray)
    ∇dropout_dot_mul = @closure Δ -> begin
        return ∂∅, (CRC.ProjectTo(x))(dropout_dot_mul(Δ, mask)), ∂∅
    end
    return dropout_dot_mul(x, mask), ∇dropout_dot_mul
end
