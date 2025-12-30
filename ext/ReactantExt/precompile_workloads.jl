using PrecompileTools: @setup_workload, @compile_workload

module PrecompileWorkloads

function sumabs2attnloss(model, ps, st, data)
    (y, _), stₙ = model(data, ps, st)
    return sum(abs2, y), stₙ, NamedTuple()
end

function sumabs2loss(model, ps, st, data)
    y, stₙ = model(data, ps, st)
    return sum(abs2, y), stₙ, NamedTuple()
end

end

if Reactant.Reactant_jll.is_available()
    @setup_workload begin
        orig_backend = Reactant.XLA.default_backend()
        Reactant.set_default_backend("cpu") # always precompile on CPU

        dev = reactant_device(; force=true)

        # attention model
        mha = Lux.MultiHeadAttention(4; nheads=2)
        ps_mha, st_mha = Lux.setup(Random.default_rng(), mha) |> dev

        q = rand(Float32, (4, 3, 2)) |> dev
        k = rand(Float32, (4, 3, 2)) |> dev
        v = rand(Float32, (4, 3, 2)) |> dev

        # convolution + dense model
        conv_model = Lux.Chain(
            Lux.Conv((3, 3), 3 => 32),
            Lux.Conv((3, 3), 32 => 64),
            Lux.GlobalMaxPool(),
            Lux.FlattenLayer(),
            Lux.Dense(64 => 10),
        )
        ps_conv_model, st_conv_model = Lux.setup(Random.default_rng(), conv_model) |> dev

        x = rand(Float32, (28, 28, 3, 2)) |> dev

        @compile_workload begin
            @compile mha((q, k, v), ps_mha, LuxCore.testmode(st_mha))

            Lux.Training.single_train_step(
                AutoEnzyme(),
                PrecompileWorkloads.sumabs2attnloss,
                (q, k, v),
                Lux.Training.TrainState(mha, ps_mha, st_mha, Optimisers.Adam(0.001f0)),
            )

            @compile conv_model(x, ps_conv_model, LuxCore.testmode(st_conv_model))

            Lux.Training.single_train_step(
                AutoEnzyme(),
                PrecompileWorkloads.sumabs2loss,
                x,
                Lux.Training.TrainState(
                    conv_model, ps_conv_model, st_conv_model, Optimisers.Adam(0.001f0)
                ),
            )
        end

        Reactant.clear_oc_cache()
        Reactant.set_default_backend(orig_backend)
    end
end
