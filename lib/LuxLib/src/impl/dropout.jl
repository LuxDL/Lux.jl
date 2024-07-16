_dropout_shape(s, ::Colon) = size(s)
function _dropout_shape(s, dims)
    return ntuple(@closure(i->ifelse(i ∈ dims, size(s, i), 1)), ndims(s))
end

CRC.@non_differentiable _dropout_shape(::Any...)
EnzymeRules.inactive_noinl(::typeof(_dropout_shape), ::Any...) = nothing

__alpha_dropout_kernel(x, noise, p, α) = ifelse(noise > p, x, α)
_alpha_dropout_kernel(noise, p, x, α) = broadcast(__alpha_dropout_kernel, x, noise, p, α)

__partial_alpha_dropout(Δ, c) = (1 - c) * Δ

function CRC.rrule(::typeof(_alpha_dropout_kernel), noise, p, x, α)
    _cond = broadcast(>, noise, p)
    y = broadcast(ifelse, _cond, x, α)
    _∇alpha_dropout_kernel = @closure Δ -> begin
        ∂x = broadcast(*, Δ, _cond)
        ∂α = sum(broadcast(__partial_alpha_dropout, Δ, _cond))
        return NoTangent(), NoTangent(), NoTangent(), ∂x, ∂α
    end
    return y, _∇alpha_dropout_kernel
end

_dropout_fptype(x) = float(real(__value(eltype(x))))

CRC.@non_differentiable _dropout_fptype(::Any...)
EnzymeRules.inactive_noinl(::typeof(_dropout_fptype), ::Any...) = nothing

function _alpha_dropout_noise(rng, x)
    rng = LuxCore.replicate(rng)
    noise = similar(x, _dropout_fptype(x))
    rand!(rng, noise)
    return noise, rng
end

CRC.@non_differentiable _alpha_dropout_noise(::Any...)
EnzymeRules.inactive_noinl(::typeof(_alpha_dropout_noise), ::Any...) = nothing

_dropout_kernel(y, p, invp) = ifelse(y > p, invp, oftype(y, 0))

function _generate_dropout_mask(rng::AbstractRNG, x, p, invp; dims)
    y = similar(x, _dropout_fptype(x), _dropout_shape(x, dims))
    rand!(rng, y)
    broadcast!(_dropout_kernel, y, y, p, invp)
    return y
end

CRC.@non_differentiable _generate_dropout_mask(::Any...)
EnzymeRules.inactive_noinl(::typeof(_generate_dropout_mask), ::Any...) = nothing
