using Comonicon, Interpolations, Lux, Optimisers, Printf, Random, Statistics, Zygote, Enzyme

@isdefined(includet) ? includet("common.jl") : include("common.jl")

CUDA.allowscalar(false)

function ConvMixer(; dim, depth, kernel_size=5, patch_size=2)
    #! format: off
    return Chain(
        Conv((patch_size, patch_size), 3 => dim, gelu; stride=patch_size),
        BatchNorm(dim),
        [
            Chain(
                SkipConnection(
                    Chain(
                        Conv(
                            (kernel_size, kernel_size), dim => dim, gelu;
                            groups=dim, pad=SamePad()
                        ),
                        BatchNorm(dim)
                    ),
                    +
                ),
                Conv((1, 1), dim => dim, gelu),
                BatchNorm(dim)
            )
            for _ in 1:depth
        ]...,
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(dim => 10)
    )
    #! format: on
end

Comonicon.@main function main(;
        batchsize::Int=512, hidden_dim::Int=256, depth::Int=8,
        patch_size::Int=2, kernel_size::Int=5, weight_decay::Float64=0.0001,
        clip_norm::Bool=false, seed::Int=1234, epochs::Int=25, lr_max::Float64=0.05,
        backend::String="reactant"
)
    model = ConvMixer(; dim=hidden_dim, depth, kernel_size, patch_size)

    opt = AdamW(; eta=lr_max, lambda=weight_decay)
    clip_norm && (opt = OptimiserChain(ClipNorm(), opt))

    lr_schedule = linear_interpolation(
        [0, epochs * 2 ÷ 5, epochs * 4 ÷ 5, epochs + 1], [0, lr_max, lr_max / 20, 0]
    )

    return train_model(model, opt, lr_schedule; backend, batchsize, seed, epochs)
end
