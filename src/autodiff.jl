no_grad(x) = zero(x)
no_grad(nt::NamedTuple) = fmap(no_grad, nt)
no_grad(::Union{AbstractRNG,Bool,AbstractExplicitLayer}) = NoTangent()

ChainRulesCore.Tangent{P}(; kwargs...) where {P<:AbstractExplicitLayer} = NoTangent()

ChainRulesCore.rrule(::typeof(istraining)) = true, _ -> (NoTangent(),)
function ChainRulesCore.rrule(::typeof(istraining), st::NamedTuple)
    return st.training, _ -> (NoTangent(), NoTangent())
end

ChainRulesCore.@non_differentiable replicate(::Any)
ChainRulesCore.@non_differentiable update_statistics(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any)
ChainRulesCore.@non_differentiable generate_dropout_mask(::Any, ::Any, ::Any)
ChainRulesCore.@non_differentiable compute_adaptive_pooling_dims(::Any, ::Any)
ChainRulesCore.@non_differentiable update_state(::Any, ::Any, ::Any)

ChainRulesCore.rrule(::typeof(Base.broadcasted), ::typeof(identity), x) = x, Δ -> (NoTangent(), NoTangent(), Δ)

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

# Sparse Arrays
_project(x, y) = x .* one.(y)

function ChainRulesCore.rrule(
    ::typeof(*),
    X::EFLSparseMatrixCSC{<:Union{AbstractSparseMatrixCSC,AbstractCuSparseMatrix}},
    Y::Union{Matrix,CuMatrix},
)
    Z = X * Y
    function sparse_matmul_pullback(Δ)
        Δ = unthunk(Δ)
        return NoTangent(), _project(Δ * Y', X), X.mat' * Δ
    end
    return Z, sparse_matmul_pullback
end

function ChainRulesCore.rrule(::typeof(lastindex), nt::NTuple{N,Int64}) where {N}
    res = lastindex(nt)
    function lastindex_pullback(Δ)
        return (NoTangent(), (ntuple(_ -> NoTangent(), N - 1)..., Δ))
    end
    return res, lastindex_pullback
end

# Activation Rrules
function ChainRulesCore.rrule(
    ::typeof(applyactivation), f::cudnnValidActivationTypes, x::CuArray{T}, inplace # Just ignore this argument
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

## Zygote Fixes

# FIXME: Before merging PR add conditional Dependency
# For supporting Yota
## `zero` should be called something else in Yota.jl ideally with a fallback to
## `Base.zero`
Base.zero(s::NamedTuple{(),Tuple{}}) = s
Base.zero(::Symbol) = NoTangent()
Base.zero(nt::NamedTuple{fields}) where {fields} = NamedTuple{fields}(zero.(values(nt)))
Base.zero(l::AbstractExplicitLayer) = NoTangent()
Base.zero(::AbstractRNG) = NoTangent()

function Yota.ungetindex(nt::NamedTuple{fields}, dy, I...) where {fields}
    dx = map(1:length(fields)) do i
        i in I ? dy : zero(nt[i])
    end
    return NamedTuple{fields}(dx)
end

recursive_sum(x, y) = x + y
recursive_sum(x::AbstractArray{T}, y::AbstractArray{T}) where {T} = x .+ y
recursive_sum(x::T, y::T) where {T<:AbstractExplicitLayer} = x
recursive_sum(nt1::NamedTuple{fields}, nt2::NamedTuple{fields}) where {fields} = map(recursive_sum, nt1, nt2)

# Definitely should not be doing this. Upsteam in some reasonable way to Yota
function Base.broadcast(::typeof(+), nt1::NamedTuple{fields}, nt2::NamedTuple{fields}) where {fields}
    # This should always be a tuple of NoTangents
    return nt1
end

function Base.broadcast(::typeof(+), nt::NoTangent, nt2::NamedTuple)
    return nt
end

function Base.broadcast(::typeof(+), nt1::NamedTuple, nt::NoTangent)
    return nt
end
