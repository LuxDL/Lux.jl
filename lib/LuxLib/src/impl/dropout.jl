_dropout_shape(s, ::Colon) = size(s)
function _dropout_shape(s, dims)
    return ntuple(@closure(i->ifelse(i ∈ dims, size(s, i), 1)), ndims(s))
end

CRC.@non_differentiable _dropout_shape(::Any...)
EnzymeRules.inactive_noinl(::typeof(_dropout_shape), ::Any...) = nothing

function _alpha_dropout_kernel(noise::AbstractArray, p, x::AbstractArray, α, A, B)
    return _alpha_dropout_kernel(internal_operation_mode((noise, x)), noise, p, x, α, A, B)
end

@stable default_mode="disable" function _alpha_dropout_kernel(
        ::LoopedArrayOp, noise::AbstractArray, p::Real,
        x::AbstractArray, α::Real, A::Real, B::Real)
    res = similar(x, promote_type(typeof(p), typeof(α)))
    @simd ivdep for i in eachindex(noise)
        @inbounds res[i] = muladd(ifelse(noise[i] > p, x[i], α), A, B)
    end
    return res
end

@stable default_mode="disable" function _alpha_dropout_kernel(
        ::AbstractBroadcastOpMode, noise::AbstractArray,
        p::Real, x::AbstractArray, α::Real, A::Real, B::Real)
    A′, B′, α = eltype(x)(A), eltype(x)(B), eltype(x)(α)
    return @. muladd(ifelse(noise > p, x, α), A′, B′)
end

# We intentionally drop the gradients for p, A, B and alpha
function CRC.rrule(::typeof(_alpha_dropout_kernel), ::LoopedArrayOp, noise::AbstractArray,
        p::Real, x::AbstractArray, α::Real, A::Real, B::Real)
    _cond = similar(noise, Bool)
    y = similar(x, promote_type(typeof(p), typeof(α), typeof(A), typeof(B), eltype(x)))
    @simd ivdep for i in eachindex(noise)
        @inbounds _cond[i] = noise[i] > p
        @inbounds y[i] = muladd(ifelse(_cond[i], x[i], α), A, B)
    end

    proj_x = CRC.ProjectTo(x)
    _∇alpha_dropout_kernel = let _cond = _cond, proj_x = proj_x, x = x, noise = noise
        Δ -> begin
            ∂x = similar(x)
            @simd ivdep for i in eachindex(noise)
                @inbounds ∂x[i] = _cond[i] * Δ[i] * A
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

_dropout_fptype(x) = float(real(__value(eltype(x))))

CRC.@non_differentiable _dropout_fptype(::Any...)
EnzymeRules.inactive_noinl(::typeof(_dropout_fptype), ::Any...) = nothing

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
        @simd ivdep for i in eachindex(y)
            @inbounds y[i] = (y[i] > p) * invp
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
