# Non Differentiable Functions
ChainRulesCore.@non_differentiable replicate(::Any)
ChainRulesCore.@non_differentiable update_statistics(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any)
ChainRulesCore.@non_differentiable generate_dropout_mask(::Any, ::Any, ::Any)
ChainRulesCore.@non_differentiable compute_adaptive_pooling_dims(::Any, ::Any)
ChainRulesCore.@non_differentiable glorot_normal(::Any...)
ChainRulesCore.@non_differentiable glorot_uniform(::Any...)
ChainRulesCore.@non_differentiable check_use_cuda()

ChainRulesCore.Tangent{P}(; kwargs...) where {P<:AbstractExplicitLayer} = NoTangent()

ChainRulesCore.rrule(::typeof(istraining)) = true, _ -> (NoTangent(),)
function ChainRulesCore.rrule(::typeof(istraining), st::NamedTuple)
    return st.training, _ -> (NoTangent(), NoTangent())
end

no_grad(x) = zero(x)
no_grad(nt::NamedTuple) = fmap(no_grad, nt)
no_grad(::Union{AbstractRNG,Bool,AbstractExplicitLayer}) = NoTangent()

ChainRulesCore.rrule(::typeof(Base.broadcasted), ::typeof(identity), x) = x, Δ -> (NoTangent(), NoTangent(), Δ)

# Base Functions
function ChainRulesCore.rrule(::typeof(merge), nt1::NamedTuple{f1}, nt2::NamedTuple{f2}) where {f1,f2}
    nt = merge(nt1, nt2)
    function merge_pullback(Δ)
        return (
            NoTangent(),
            NamedTuple{f1}(map(k -> k ∈ f2 ? NoTangent() : Δ[k], keys(Δ))),
            NamedTuple{f2}(getfield.((Δ,), f2)),
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
function ChainRulesCore.rrule(::typeof(applydropout), x, mask)
    y, broadcast_pullback = rrule_via_ad(Zygote.ZygoteRuleConfig(), broadcast, *, x, mask)
    function applydropout_pullback(Δ)
        _, _, Δx, _ = broadcast_pullback(Δ)
        return (NoTangent(), Δx, NoTangent())
    end
    return y, applydropout_pullback
end

function ChainRulesCore.rrule(::typeof(dropout), rng, x, prob, dims, training)
    @assert training AssertionError("Trying to conpute dropout gradients when `training`=false")
    rng = replicate(rng)
    mask = generate_dropout_mask(rng, x, prob; dims)
    y, broadcast_pullback = rrule_via_ad(Zygote.ZygoteRuleConfig(), broadcast, *, x, mask)
    function dropout_pullback(Δ)
        _, _, Δx, _ = broadcast_pullback(Δ[1])
        return (NoTangent(), NoTangent(), Δx, NoTangent(), NoTangent(), NoTangent())
    end
    return (y, mask, rng), dropout_pullback
end

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
    ::typeof(applyactivation),
    f::cudnnValidActivationTypes,
    x::CuArray{T},
    inplace, # Just ignore this argument
) where {T}
    mode = getCUDNNActivationMode(f)
    y = CUDA.CUDNN.cudnnActivationForward(x; mode)
    function applyactivation_pullback(Δ)
        # NOTE: Since this is an internal function, we are violating our pure function standards
        #       and mutating the input
        Δx = Δ
        desc = CUDA.CUDNN.cudnnActivationDescriptor(mode, CUDA.CUDNN.CUDNN_NOT_PROPAGATE_NAN, Cdouble(1))
        CUDA.CUDNN.cudnnActivationBackward(
            CUDA.CUDNN.handle(),
            desc,
            CUDA.CUDNN.scalingParameter(T, 1),
            CUDA.CUDNN.cudnnTensorDescriptor(y),
            y,
            CUDA.CUDNN.cudnnTensorDescriptor(Δ),
            Δ,
            CUDA.CUDNN.cudnnTensorDescriptor(x),
            x,
            CUDA.CUDNN.scalingParameter(T, 0),
            CUDA.CUDNN.cudnnTensorDescriptor(Δx),
            Δx,
        )
        return NoTangent(), NoTangent(), Δx, NoTangent()
    end
    return y, applyactivation_pullback
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
