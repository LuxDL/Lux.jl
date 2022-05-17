# Non Differentiable Functions
ChainRulesCore.@non_differentiable replicate(::Any)
ChainRulesCore.@non_differentiable update_statistics(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any)
ChainRulesCore.@non_differentiable generate_dropout_mask(::Any, ::Any, ::Any)
ChainRulesCore.@non_differentiable compute_adaptive_pooling_dims(::Any, ::Any)
ChainRulesCore.@non_differentiable glorot_normal(::Any...)
ChainRulesCore.@non_differentiable glorot_uniform(::Any...)
ChainRulesCore.@non_differentiable check_use_cuda()
ChainRulesCore.@non_differentiable istraining(::Any)

ChainRulesCore.Tangent{P}(; kwargs...) where {P<:AbstractExplicitLayer} = NoTangent()

ChainRulesCore.rrule(::typeof(istraining)) = true, _ -> (NoTangent(),)

ChainRulesCore.rrule(::typeof(Base.broadcasted), ::typeof(identity), x) = x, Δ -> (NoTangent(), NoTangent(), Δ)

# Base Functions
function ChainRulesCore.rrule(::typeof(merge), nt1::NamedTuple{f1}, nt2::NamedTuple{f2}) where {f1,f2}
    nt = merge(nt1, nt2)
    function merge_pullback(Δ)
        return (
            NoTangent(),
            NamedTuple{f1}(map(k -> k ∈ f2 ? NoTangent() : getproperty(Δ, k), keys(Δ))),
            NamedTuple{f2}(getproperty.((Δ,), f2)),
        )
    end
    return nt, merge_pullback
end

function ChainRulesCore.rrule(::typeof(lastindex), nt::NTuple{N,Int64}) where {N}
    res = lastindex(nt)
    function lastindex_pullback(Δ)
        return (NoTangent(), (ntuple(_ -> NoTangent(), N - 1)..., Δ))
    end
    return res, lastindex_pullback
end

# NNlib Functions
function ChainRulesCore.rrule(
    ::typeof(batchnorm),
    g::CuArray{T},
    b::CuArray{T},
    x::Union{CuArray{T,4},CuArray{T,5}},
    running_mean,
    running_var,
    momentum;
    kwargs...,
) where {T<:CUDNNFloat}
    y = batchnorm(g, b, x, running_mean, running_var, momentum; kwargs...)
    function batchnorm_pullback(dy)
        dg, db, dx = ∇batchnorm(g, b, x, dy, running_mean, running_var, momentum; kwargs...)
        return NoTangent(), dg, db, dx, NoTangent(), NoTangent(), NoTangent()
    end
    return y, batchnorm_pullback
end

# Activation Rrules
function ChainRulesCore.rrule(
    ::typeof(applyactivation), f::cudnnValidActivationTypes, x::CuArray{T}
) where {T<:CUDNNFloat}
    mode = getCUDNNActivationMode(f)
    y = CUDNN.cudnnActivationForward(x; mode)
    function applyactivation_pullback(Δ)
        return NoTangent(), NoTangent(), cudnnActivationBackward(y, Δ, x; mode), NoTangent()
    end
    return y, applyactivation_pullback
end

# Elementwise Functions
function ChainRulesCore.rrule(::typeof(elementwise_add), x, y) where {T}
    z = elementwise_add(x, y)
    _elementwise_add_pullback(Δ) = (NoTangent(), elementwise_add_pullback(x, y, Δ)...)
    return z, _elementwise_add_pullback
end

function ChainRulesCore.rrule(::typeof(elementwise_mul), x, y) where {T}
    z = elementwise_mul(x, y)
    _elementwise_mul_pullback(Δ) = (NoTangent(), elementwise_mul_pullback(x, y, Δ)...)
    return z, _elementwise_mul_pullback
end

# Zygote Fixes
function Zygote.accum(x::ComponentArray, ys::ComponentArray...)
    return ComponentArray(Zygote.accum(getdata(x), getdata.(ys)...), getaxes(x))
end

# Adapt Interface
function ChainRulesCore.rrule(::typeof(Array), x::CUDA.CuArray)
    return Array(x), d -> (NoTangent(), CUDA.cu(d))
end

function ChainRulesCore.rrule(::typeof(adapt_storage), to::LuxCPUAdaptor, x::CUDA.AbstractGPUArray)
    return adapt_storage(to, x), d -> (NoTangent(), NoTangent(), adapt_storage(LuxCUDAAdaptor(), d))
end

function ChainRulesCore.rrule(::typeof(adapt_storage), to::LuxCUDAAdaptor, x::Array)
    return adapt_storage(to, x), d -> (NoTangent(), NoTangent(), adapt_storage(LuxCPUAdaptor(), d))
end

# RNN Helpers
## Taken from https://github.com/FluxML/Flux.jl/blob/1f82da4bfa051c809f7f3ce7dd7aeb43be515b14/src/layers/recurrent.jl#L9
function ChainRulesCore.rrule(::typeof(multigate), x::AbstractArray, c::Val{N}) where {N}
    function multigate_pullback(dy)
        dx = map!(zero, similar(x, float(eltype(x)), axes(x)), x)
        foreach(multigate(dx, c), dy) do dxᵢ, dyᵢ
            dyᵢ isa AbstractZero && return
            @. dxᵢ += dyᵢ
        end
        return (NoTangent(), dx, NoTangent(), NoTangent())
    end
    return multigate(x, c), multigate_pullback
end
