function Impl.batchnorm(
    x::AnyTracedRArray{T},
    γ::Optional{<:AnyTracedRVector},
    β::Optional{<:AnyTracedRVector},
    rμ::Optional{<:AnyTracedRVector},
    rσ²::Optional{<:AnyTracedRVector},
    ::False,
    act::F,
    momentum,
    ϵ,
) where {T,F}
    x = Reactant.materialize_traced_array(x)
    γ !== nothing && (γ = Reactant.materialize_traced_array(γ))
    β !== nothing && (β = Reactant.materialize_traced_array(β))

    if rμ === nothing && rσ² === nothing
        μ, σ² = Impl.mean_var(
            x; dims=Utils.unsafe_known(Impl.batchnorm_reduce_dims(x)), corrected=false
        )
        μ = TracedUtils.materialize_traced_array(vec(μ))
        σ² = TracedUtils.materialize_traced_array(vec(σ²))
    else
        @assert rμ !== nothing && rσ² !== nothing
        μ = Reactant.materialize_traced_array(rμ)
        σ² = Reactant.materialize_traced_array(rσ²)
    end

    return (
        act.(
            Ops.batch_norm_inference(
                x, γ, β, μ, σ²; epsilon=ϵ, feature_index=Int64(ndims(x) - 1)
            )
        ),
        rμ,
        rσ²,
    )
end

function Impl.batchnorm(
    x::AnyTracedRArray{T},
    γ::Optional{<:AnyTracedRVector},
    β::Optional{<:AnyTracedRVector},
    rμ::Optional{<:AnyTracedRVector},
    rσ²::Optional{<:AnyTracedRVector},
    ::True,
    act::F,
    momentum,
    ϵ,
) where {T,F}
    x = Reactant.materialize_traced_array(x)
    γ !== nothing && (γ = Reactant.materialize_traced_array(γ))
    β !== nothing && (β = Reactant.materialize_traced_array(β))

    y, μ, σ² = Ops.batch_norm_training(
        x, γ, β; epsilon=ϵ, feature_index=Int64(ndims(x) - 1)
    )
    res = act.(y)

    (rμ === nothing && rσ² === nothing) && return (res, nothing, nothing)

    @assert rμ !== nothing && rσ² !== nothing
    m = T(Impl.accum_size(x, Impl.batchnorm_reduce_dims(x)))
    rμ, rσ² = Impl.update_running_statistics(
        rμ, rσ², μ, σ², momentum, momentum * m / (m - one(m))
    )
    return res, rμ, rσ²
end
