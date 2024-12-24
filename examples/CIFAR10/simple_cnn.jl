using Comonicon, Lux, Optimisers, Printf, Random, Statistics, Zygote, Enzyme

@isdefined(includet) ? includet("common.jl") : include("common.jl")

CUDA.allowscalar(false)

function SimpleCNN()
    return Chain(
        Conv((3, 3), 3 => 16, gelu; stride=2, pad=1),
        BatchNorm(16),
        Conv((3, 3), 16 => 32, gelu; stride=2, pad=1),
        BatchNorm(32),
        Conv((3, 3), 32 => 64, gelu; stride=2, pad=1),
        BatchNorm(64),
        Conv((3, 3), 64 => 128, gelu; stride=2, pad=1),
        BatchNorm(128),
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(128 => 64, gelu),
        BatchNorm(64),
        Dense(64 => 10)
    )
end

Comonicon.@main function main(;
        batchsize::Int=512, weight_decay::Float64=0.0001,
        clip_norm::Bool=false, seed::Int=1234, epochs::Int=50, lr::Float64=0.003,
        backend::String="reactant"
)
    model = SimpleCNN()

    opt = AdamW(; eta=lr, lambda=weight_decay)
    clip_norm && (opt = OptimiserChain(ClipNorm(), opt))

    return train_model(model, opt, nothing; backend, batchsize, seed, epochs)
end
