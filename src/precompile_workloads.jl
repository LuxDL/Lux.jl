using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # attention model
    mha = MultiHeadAttention(4; nheads=2)
    ps_mha, st_mha = setup(Random.default_rng(), mha)

    q = rand(Float32, (4, 3, 2))
    k = rand(Float32, (4, 3, 2))
    v = rand(Float32, (4, 3, 2))

    # convolution + dense model
    conv_model = Chain(
        Conv((3, 3), 3 => 32; use_bias=false),
        Conv((3, 3), 32 => 64; use_bias=false),
        GlobalMaxPool(),
        FlattenLayer(),
        Dense(64 => 10),
    )
    ps_conv_model, st_conv_model = setup(Random.default_rng(), conv_model)

    x = rand(Float32, (28, 28, 3, 2))

    @compile_workload begin
        mha((q, k, v), ps_mha, st_mha)
        conv_model(x, ps_conv_model, st_conv_model)
    end
end
