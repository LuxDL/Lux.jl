Base.zero(s::NamedTuple{(),Tuple{}}) = s

Base.zero(::Symbol) = Symbol()

Base.zero(nt::NamedTuple{fields}) where {fields} = NamedTuple{fields}(zero.(values(nt)))

# Layers are stateless so we can simply return that
Base.zero(l::AbstractExplicitLayer) = l

ChainRulesCore.rrule(::typeof(istraining)) = true, _ -> (NoTangent(),)
ChainRulesCore.rrule(::typeof(istraining), st::NamedTuple) = true, _ -> (NoTangent(), NoTangent())

ChainRulesCore.@non_differentiable _update_stats!(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any)
ChainRulesCore.@non_differentiable update_statistics(::Any, ::Any, ::Any, ::Any, ::Any, ::Any)
ChainRulesCore.@non_differentiable _dropout_mask(::Any, ::Any, ::Any)

ChainRulesCore.rrule(::typeof(Base.broadcasted), ::typeof(identity), x) = x, Δ -> (NoTangent(), NoTangent(), Δ)

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

# Fast Matmul
function ChainRulesCore.rrule(
    ::typeof(fast_matmul!), C::AbstractVecOrMat{T}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}
) where {T}
    fast_matmul!(C, A, B)
    function fast_matmul!_pullback(Δ)
        Δ = unfill_array(unthunk(Δ))
        return NoTangent(), Δ, fast_matmul(Δ, B'), fast_matmul(A', Δ)
    end
    function fast_matmul!_pullback(Δ, cache)
        Δ = unfill_array(unthunk(Δ))
        return NoTangent(), Δ, fast_matmul!(cache.∇A, Δ, B'), fast_matmul!(cache.∇B, A', Δ)
    end
    return C, fast_matmul!_pullback
end

# Dense Layer
# function ChainRulesCore.rrule(
#     d::Dense{bias}, x::AbstractArray{T,N}, ps::NamedTuple, st::NamedTuple, cache::NamedTuple
# ) where {T,N,bias}
#     # FIXME: Activation
#     if bias
#         b = N == 1 ? view(ps.bias, :, 1) : ps.bias
#         Wx, fast_matmul_pullback = rrule(fast_matmul!, cache.Wx, ps.weight, x)
#         cache.λWxb .= Wx .+ b
#         λWxb, activation_pullback = rrule(Base.broadcasted, d.λ, cache.λWxb)
#         function Dense_pullback(Δ)
#             Δ = unfill_array(unthunk(activation_pullback(Δ)[3]))
#         end
#         return (λWxb, st), Dense_pullback
#     else
#     end
# end

# Convolution
function ChainRulesCore.rrule(
    config,
    ::typeof(fast_conv_bias_act),
    x::AbstractArray{xT,N},
    w::AbstractArray{wT,N},
    cdims::ConvDims,
    b::AbstractArray{bT,N},
    λ::T;
    kwargs...,
) where {xT,wT,bT,N,T}
    y = conv(x, w, cdims)
    @. y += b
    if T != typeof(identity)
        act = rrule(config, Base.broadcasted, λ, y)
        y, act_pullback = if act === nothing
            rrule_via_ad(config, Base.broadcasted, λ, y)
        else
            act
        end
        function fast_conv_bias_act_pullback(Δ)
            Δ = unfill_array(unthunk(act_pullback(Δ)[3]))
            ∇b = sum(Δ; dims=[1:(N - 2); N])
            return (
                NoTangent(),
                NNlib.∇conv_data(Δ, w, cdims),
                NNlib.∇conv_filter(x, Δ, cdims),
                NoTangent(),
                ∇b,
                NoTangent(),
            )
        end
        return y, fast_conv_bias_act_pullback
    else
        function fast_conv_bias_pullback(Δ)
            Δ = unfill_array(unthunk(Δ))
            ∇b = sum(Δ; dims=[1:(N - 2); N])
            return (
                NoTangent(),
                NNlib.∇conv_data(Δ, w, cdims),
                NNlib.∇conv_filter(x, Δ, cdims),
                NoTangent(),
                ∇b,
                NoTangent(),
            )
        end
        return y, fast_conv_bias_pullback
    end
end
