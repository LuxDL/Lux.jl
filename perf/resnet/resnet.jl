using Lux, Random

conv_layer(args...; kwargs...) = Conv(args...; use_bias=false, kwargs...)
norm_layer(args...; kwargs...) = BatchNorm(args...; epsilon=1.0e-5, momentum=0.1, kwargs...)

addact(act) = (x, y) -> act.(x .+ y)
addact(act, x, y) = @. act(x + y)

function ResNetBlock(;
    in_chs::Int,
    out_chs::Int,
    conv_layer=conv_layer,
    norm_layer=norm_layer,
    act,
    stride::Tuple{Int,Int}=(1, 1),
)
    main_block = Chain(
        conv_layer((3, 3), in_chs => out_chs; stride, pad=1),
        norm_layer(out_chs, act),
        conv_layer((3, 3), out_chs => out_chs; pad=1),
        norm_layer(out_chs, act; init_scale=zeros32),
    )
    residual_block = if out_chs != in_chs || !all(==(1), stride)
        Chain(conv_layer((1, 1), in_chs => out_chs; stride), norm_layer(out_chs))
    else
        NoOpLayer()
    end
    return Parallel(addact(act), main_block, residual_block)
end

function BottleneckResNetBlock(;
    in_chs::Int,
    out_chs::Int,
    conv_layer=conv_layer,
    norm_layer=norm_layer,
    act,
    stride::Tuple{Int,Int}=(1, 1),
)
    main_block = Chain(
        conv_layer((1, 1), in_chs => out_chs),
        norm_layer(out_chs, act),
        conv_layer((3, 3), out_chs => out_chs; stride, pad=1),
        norm_layer(out_chs, act),
        conv_layer((1, 1), out_chs => out_chs * 4),
        norm_layer(out_chs * 4; init_scale=zeros32),
    )
    residual_block = if in_chs != out_chs * 4 || !all(==(1), stride)
        Chain(conv_layer((1, 1), in_chs => out_chs * 4; stride), norm_layer(out_chs * 4))
    else
        NoOpLayer()
    end
    return Parallel(addact(act), main_block, residual_block)
end

function ResNet(;
    stage_sizes::Vector{Int},
    num_classes::Int,
    num_filters::Int=64,
    block,
    in_chs::Int,
    act=relu,
)
    initial_block = Chain(
        conv_layer((7, 7), in_chs => num_filters; stride=(2, 2), pad=(3, 3)),
        norm_layer(num_filters, relu),
        MaxPool((3, 3); stride=(2, 2), pad=SamePad()),
    )

    blocks = []
    out_chs, in_chs = num_filters, num_filters
    for (i, block_size) in enumerate(stage_sizes)
        semi_blocks = []
        for j in 1:block_size
            stride = i > 1 && j == 1 ? (2, 2) : (1, 1)
            out_chs = num_filters * (2^(i - 1))
            push!(
                semi_blocks, block(; in_chs, out_chs, stride, conv_layer, norm_layer, act)
            )
            in_chs = block === BottleneckResNetBlock ? out_chs * 4 : out_chs
        end
        push!(blocks, Chain(semi_blocks...))
    end
    mid_block = Chain(blocks...)

    classifier = Chain(GlobalMeanPool(), FlattenLayer(), Dense(in_chs, num_classes))

    return Chain(initial_block, mid_block, classifier)
end

function ResNet(sz::Int)
    kwargs = (; num_classes=1000, num_filters=64, act=relu, in_chs=3)
    if sz == 18
        return ResNet(; stage_sizes=[2, 2, 2, 2], block=ResNetBlock, kwargs...)
    elseif sz == 34
        return ResNet(; stage_sizes=[3, 4, 6, 3], block=ResNetBlock, kwargs...)
    elseif sz == 50
        return ResNet(; stage_sizes=[3, 4, 6, 3], block=BottleneckResNetBlock, kwargs...)
    elseif sz == 101
        return ResNet(; stage_sizes=[3, 4, 23, 3], block=BottleneckResNetBlock, kwargs...)
    elseif sz == 152
        return ResNet(; stage_sizes=[3, 8, 36, 3], block=BottleneckResNetBlock, kwargs...)
    elseif sz == 200
        return ResNet(; stage_sizes=[3, 24, 36, 3], block=BottleneckResNetBlock, kwargs...)
    else
        error("Invalid model size: $(sz)")
    end
end
