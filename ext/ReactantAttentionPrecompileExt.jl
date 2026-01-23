module ReactantAttentionPrecompileExt

using ADTypes: AutoEnzyme
using Reactant: Reactant, @compile
using Lux: Lux
using LuxCore: LuxCore
using MLDataDevices: reactant_device
using Optimisers: Optimisers
using Random: Random
using PrecompileTools: @setup_workload, @compile_workload

function sumabs2attnloss(model, ps, st, data)
    (y, _), stₙ = model(data, ps, st)
    return sum(abs2, y), stₙ, NamedTuple()
end

if Reactant.Reactant_jll.is_available()
    @setup_workload begin
        orig_backend = Reactant.XLA.default_backend()
        Reactant.set_default_backend("cpu")

        @compile_workload begin
            @static if Reactant.precompilation_supported()
                dev = reactant_device(; force=true)

                mha = Lux.MultiHeadAttention(4; nheads=2)
                ps_mha, st_mha = Lux.setup(Random.default_rng(), mha) |> dev

                q = ones(Float32, (4, 3, 2)) |> dev
                k = ones(Float32, (4, 3, 2)) |> dev
                v = ones(Float32, (4, 3, 2)) |> dev

                try
                    @compile mha((q, k, v), ps_mha, LuxCore.testmode(st_mha))

                    Lux.Training.single_train_step(
                        AutoEnzyme(),
                        sumabs2attnloss,
                        (q, k, v),
                        Lux.Training.TrainState(
                            mha, ps_mha, st_mha, Optimisers.Adam(0.001f0)
                        ),
                    )
                catch err
                    if !(err isa Reactant.ReactantPrecompilationException)
                        rethrow(err)
                    end
                end
            end
        end

        Reactant.set_default_backend(orig_backend)
    end
end

end
