module LuxLibReactantExt

using Reactant: Reactant, MLIR, Ops, TracedUtils, TracedRArray, AnyTracedRArray,
                AnyTracedRVector
using Static: StaticBool, True, False

using LuxLib: LuxLib, Impl, Optional, Utils

# Most of the NN code gen happens in Reactant.jl via an extension on NNlib, however,
# NNlib doesn't have certain ops implemented. In those cases we can emit more optimized
# StableHLO

function Impl.batchnorm(
        x::AnyTracedRArray{T},
        γ::Optional{<:AnyTracedRVector}, β::Optional{<:AnyTracedRVector},
        rμ::Optional{<:AnyTracedRVector}, rσ²::Optional{<:AnyTracedRVector},
        training::StaticBool, act::F, momentum::Real, ϵ::Real
) where {T, F}
    x = TracedUtils.materialize_traced_array(x)

    γ = if γ === nothing
        Ops.constant(fill(T(1), size(x, ndims(x) - 1)))
    else
        TracedUtils.materialize_traced_array(γ)
    end
    β = if β === nothing
        Ops.constant(fill(T(0), size(x, ndims(x) - 1)))
    else
        TracedUtils.materialize_traced_array(β)
    end

    if training isa True
        op = MLIR.Dialects.stablehlo.batch_norm_training(
            TracedUtils.get_mlir_data(x),
            TracedUtils.get_mlir_data(γ),
            TracedUtils.get_mlir_data(β);
            epsilon=Float32(ϵ),
            feature_index=Int64(ndims(x) - 2)
        )

        res = act.(TracedRArray{T, ndims(x)}((), MLIR.IR.result(op, 1), size(x)))
        μ = TracedRArray{T, 1}((), MLIR.IR.result(op, 2), size(x, ndims(x) - 1))
        σ² = TracedRArray{T, 1}((), MLIR.IR.result(op, 3), size(x, ndims(x) - 1))

        if rμ === nothing && rσ² === nothing
            return res, nothing, nothing
        else
            @assert rμ !== nothing && rσ² !== nothing
            m = T(Impl.accum_size(x, Impl.batchnorm_reduce_dims(x)))
            rμ, rσ² = Impl.update_running_statistics(
                rμ, rσ², μ, σ², momentum, momentum * m / (m - one(m))
            )
            return res, rμ, rσ²
        end
    else
        if rμ === nothing && rσ² === nothing
            μ, σ² = Impl.mean_var(
                x; dims=Utils.unsafe_known(Impl.batchnorm_reduce_dims(x)), corrected=false
            )
            μ = TracedUtils.materialize_traced_array(vec(μ))
            σ² = TracedUtils.materialize_traced_array(vec(σ²))
        else
            @assert rμ !== nothing && rσ² !== nothing
            μ = TracedUtils.materialize_traced_array(rμ)
            σ² = TracedUtils.materialize_traced_array(rσ²)
        end

        res = MLIR.IR.result(
            MLIR.Dialects.stablehlo.batch_norm_inference(
                TracedUtils.get_mlir_data(x),
                TracedUtils.get_mlir_data(γ),
                TracedUtils.get_mlir_data(β),
                TracedUtils.get_mlir_data(μ),
                TracedUtils.get_mlir_data(σ²);
                epsilon=Float32(ϵ),
                feature_index=Int64(ndims(x) - 2)
            ),
            1
        )

        return act.(TracedRArray{T, ndims(x)}((), res, size(x))), rμ, rσ²
    end
end

end
