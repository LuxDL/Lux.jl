# # ConvMixer on CIFAR-10

# ## Package Imports

using ArgParse, Interpolations, Lux, Optimisers, Printf, Random, Statistics

include("common.jl")

# ## Model Definition

function ConvMixer(; dim, depth, kernel_size=5, patch_size=2)
    return Chain(
        Conv((patch_size, patch_size), 3 => dim, relu; stride=patch_size),
        BatchNorm(dim),
        [
            Chain(
                SkipConnection(
                    Chain(
                        Conv(
                            (kernel_size, kernel_size),
                            dim => dim,
                            relu;
                            groups=dim,
                            pad=SamePad(),
                        ),
                        BatchNorm(dim),
                    ),
                    +,
                ),
                Conv((1, 1), dim => dim, relu),
                BatchNorm(dim),
            ) for _ in 1:depth
        ]...,
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(dim => 10),
    )
end

# ## Entry Point

function main(;
    batchsize::Int=512,
    hidden_dim::Int=256,
    depth::Int=8,
    patch_size::Int=2,
    kernel_size::Int=5,
    weight_decay::Float64=0.0001,
    clip_norm::Bool=false,
    seed::Int=1234,
    epochs::Int=25,
    lr_max::Float64=0.05,
    bfloat16::Bool=false,
)
    model = ConvMixer(; dim=hidden_dim, depth, kernel_size, patch_size)

    opt = AdamW(; eta=lr_max, lambda=weight_decay)
    clip_norm && (opt = OptimiserChain(ClipNorm(), opt))

    lr_schedule = linear_interpolation(
        [0, epochs * 2 ÷ 5, epochs * 4 ÷ 5, epochs + 1], [0, lr_max, lr_max / 20, 0]
    )

    return train_model(model, opt, lr_schedule; batchsize, seed, epochs, bfloat16)
end

function get_argparse_settings()
    s = ArgParseSettings(; autofix_names=true)
    @add_arg_table! s begin
        "--batchsize"
            arg_type = Int
            default = 512
        "--hidden-dim"
            arg_type = Int
            default = 256
        "--depth"
            arg_type = Int
            default = 8
        "--patch-size"
            arg_type = Int
            default = 2
        "--kernel-size"
            arg_type = Int
            default = 5
        "--weight-decay"
            arg_type = Float64
            default = 0.0001
        "--clip-norm"
            action = :store_true
        "--seed"
            arg_type = Int
            default = 1234
        "--epochs"
            arg_type = Int
            default = 25
        "--lr-max"
            arg_type = Float64
            default = 0.05
        "--bfloat16"
            action = :store_true
    end
    return s
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args(ARGS, get_argparse_settings(); as_symbols=true)
    main(;
        batchsize=args[:batchsize],
        hidden_dim=args[:hidden_dim],
        depth=args[:depth],
        patch_size=args[:patch_size],
        kernel_size=args[:kernel_size],
        weight_decay=args[:weight_decay],
        clip_norm=args[:clip_norm],
        seed=args[:seed],
        epochs=args[:epochs],
        lr_max=args[:lr_max],
        bfloat16=args[:bfloat16],
    )
end
