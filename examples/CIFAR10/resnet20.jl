using Comonicon, Lux, Optimisers, Printf, Random, Statistics, Zygote

include("common.jl")

function ConvBN(kernel_size, (in_chs, out_chs), act; kwargs...)
    return Chain(
        Conv(kernel_size, in_chs => out_chs, act; kwargs...),
        BatchNorm(out_chs)
    )
end

function BasicBlock(in_channels, out_channels; stride=1)
    connection = if (stride == 1 && in_channels == out_channels)
        NoOpLayer()
    else
        Conv((3, 3), in_channels => out_channels, identity; stride=stride, pad=SamePad())
    end
    return Chain(
        Parallel(
            +,
            connection,
            Chain(
                ConvBN((3, 3), in_channels => out_channels, relu; stride, pad=SamePad()),
                ConvBN((3, 3), out_channels => out_channels, identity; pad=SamePad())
            )
        ),
        Base.BroadcastFunction(relu)
    )
end

function ResNet20(; num_classes=10)
    layers = []

    # Initial Conv Layer
    push!(layers, Chain(
        Conv((3, 3), 3 => 16, relu; pad=SamePad()),
        BatchNorm(16)
    ))

    # Residual Blocks
    block_configs = [
        (16, 16, 3, 1),  # (in_channels, out_channels, num_blocks, stride)
        (16, 32, 3, 2),
        (32, 64, 3, 2)
    ]

    for (in_channels, out_channels, num_blocks, stride) in block_configs
        for i in 1:num_blocks
            push!(layers,
                BasicBlock(
                    i == 1 ? in_channels : out_channels, out_channels;
                    stride=(i == 1 ? stride : 1)
                ))
        end
    end

    # Global Pooling and Final Dense Layer
    push!(layers, GlobalMeanPool())
    push!(layers, FlattenLayer())
    push!(layers, Dense(64 => num_classes))

    return Chain(layers...)
end

Comonicon.@main function main(;
        batchsize::Int=512, weight_decay::Float64=0.0001,
        clip_norm::Bool=false, seed::Int=1234, epochs::Int=100, lr::Float64=0.001,
        backend::String="reactant", bfloat16::Bool=false
)
    model = ResNet20()

    opt = AdamW(; eta=lr, lambda=weight_decay)
    clip_norm && (opt = OptimiserChain(ClipNorm(), opt))

    lr_schedule = nothing

    return train_model(model, opt, lr_schedule; backend, batchsize, seed, epochs, bfloat16)
end
