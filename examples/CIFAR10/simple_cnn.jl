# # Convolutional Neural Network on CIFAR-10

# ## Package Imports

using ArgParse, Lux, Optimisers, Printf, Random, Statistics, Enzyme

include("common.jl")

# ## Model Definition

function SimpleCNN()
    return Chain(
        Chain(
            Conv((3, 3), 3 => 16, relu; stride=2, pad=1),
            BatchNorm(16),
            Conv((3, 3), 16 => 32, relu; stride=2, pad=1),
            BatchNorm(32),
            Conv((3, 3), 32 => 64, relu; stride=2, pad=1),
            BatchNorm(64),
            Conv((3, 3), 64 => 128, relu; stride=2, pad=1),
            BatchNorm(128),
        ),
        GlobalMeanPool(),
        FlattenLayer(),
        Chain(Dense(128 => 64, relu), BatchNorm(64), Dense(64 => 10)),
    )
end

# ## Entry Point

function main(;
    batchsize::Int=512,
    weight_decay::Float64=0.0001,
    clip_norm::Bool=false,
    seed::Int=1234,
    epochs::Int=50,
    lr::Float64=0.003,
    bfloat16::Bool=false,
    minimal::Bool=false,
)
    if minimal
        epochs = 1
        subset_size = 2 * batchsize
    else
        subset_size = nothing
    end

    model = SimpleCNN()

    opt = AdamW(; eta=lr, lambda=weight_decay)
    clip_norm && (opt = OptimiserChain(ClipNorm(), opt))

    return train_model(model, opt, nothing; batchsize, seed, epochs, bfloat16, subset_size)
end

function get_argparse_settings()
    s = ArgParseSettings(; autofix_names=true)
    #! format: off
    @add_arg_table! s begin
        "--batchsize"
            arg_type = Int
            default = 512
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
            default = 50
        "--lr"
            arg_type = Float64
            default = 0.003
        "--bfloat16"
            action = :store_true
        "--minimal"
            action = :store_true
    end
    #! format: on
    return s
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args(ARGS, get_argparse_settings(); as_symbols=true)
    main(; args...)
end
