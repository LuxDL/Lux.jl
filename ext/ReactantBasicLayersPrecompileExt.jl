module ReactantBasicLayersPrecompileExt

using ADTypes: AutoEnzyme
using Reactant: Reactant, @compile
using Lux: Lux
using LuxCore: LuxCore
using MLDataDevices: reactant_device
using Optimisers: Optimisers
using Random: Random
using PrecompileTools: @setup_workload, @compile_workload

function sumabs2loss(model, ps, st, data)
    y, stₙ = model(data, ps, st)
    return sum(abs2, y), stₙ, NamedTuple()
end

if Reactant.Reactant_jll.is_available()
    @setup_workload begin
        orig_backend = Reactant.XLA.default_backend()
        Reactant.set_default_backend("cpu")

        @compile_workload begin
            @static if Reactant.precompilation_supported()
                dev = reactant_device(; force=true)

                conv_model = Lux.Chain(
                    Lux.Conv((3, 3), 3 => 32),
                    Lux.Conv((3, 3), 32 => 64),
                    Lux.GlobalMaxPool(),
                    Lux.FlattenLayer(),
                    Lux.Dense(64 => 10),
                )
                ps_conv_model, st_conv_model =
                    Lux.setup(Random.default_rng(), conv_model) |> dev

                x = ones(Float32, (28, 28, 3, 2)) |> dev

                try
                    @compile conv_model(x, ps_conv_model, LuxCore.testmode(st_conv_model))

                    Lux.Training.single_train_step(
                        AutoEnzyme(),
                        sumabs2loss,
                        x,
                        Lux.Training.TrainState(
                            conv_model,
                            ps_conv_model,
                            st_conv_model,
                            Optimisers.Adam(0.001f0),
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
