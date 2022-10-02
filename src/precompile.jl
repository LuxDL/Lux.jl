if Preferences.@load_preference("LuxSnoopPrecompile", true)
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    # SnoopPrecompile can't do `gpu` for now :(
    DEVICES = [cpu]

    LAYERS = [
        Chain(Chain(Dense(10 => 10, sigmoid), GroupNorm(10, 5; track_stats=false)),
              Chain(Dense(10 => 5), BatchNorm(5, relu)), Dense(5 => 2, tanh), Dense(2 => 1);
              disable_optimizations=true),
        Chain(Chain(Conv((3, 3), 3 => 16, sigmoid), GroupNorm(16, 4; track_stats=false)),
              Chain(Conv((3, 3), 16 => 8, relu), BatchNorm(8, relu)),
              Conv((3, 3), 8 => 1, tanh); disable_optimizations=true),
        Chain(Chain(Conv((3, 3), 3 => 16, sigmoid; pad=SamePad()),
                    GroupNorm(16, 4; track_stats=false)),
              Chain(Conv((3, 3), 16 => 8, relu; pad=SamePad()), BatchNorm(8, relu)),
              Conv((3, 3), 8 => 1, tanh; pad=SamePad()); disable_optimizations=true),
    ]
    X_SIZE = [(10, 2), (16, 16, 3, 2), (16, 16, 3, 2)]

    for dev in DEVICES, (layer, x_size) in zip(LAYERS, X_SIZE)
        ps, st = setup(rng, layer) .|> dev
        x = rand(rng, Float32, x_size...) |> dev

        layer(x, ps, st)
        Zygote.gradient(p -> sum(layer(x, ps, st)[1]), ps)

        # ComponentArrays
        if Preferences.@load_preference("LuxPrecompileComponentArrays", true)
            ps, st = setup(rng, layer)
            ps = ps |> ComponentArray |> dev
            st = st |> dev
            x = rand(rng, Float32, x_size...) |> dev

            layer(x, ps, st)
            Zygote.gradient(p -> sum(layer(x, ps, st)[1]), ps)
        end
    end
end
