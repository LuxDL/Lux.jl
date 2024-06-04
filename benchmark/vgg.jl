function add_vgg_benchmarks!()
    # Julia Fails to Specialize on Large NTs, so split it up
    function conv_bn(ksize, (in_chs, out_chs)::Pair, args...; kwargs...)
        return Chain(
            Conv(ksize, in_chs => out_chs, args...; kwargs...), BatchNorm(out_chs);
            name="ConvBN")
    end

    vgg16 = Chain(
        Chain(conv_bn((3, 3), 3 => 64, relu; pad=(1, 1), stride=(1, 1)),
            conv_bn((3, 3), 64 => 64, relu; pad=(1, 1), stride=(1, 1)),
            MaxPool((2, 2)),
            conv_bn((3, 3), 64 => 128, relu; pad=(1, 1), stride=(1, 1)),
            conv_bn((3, 3), 128 => 128, relu; pad=(1, 1), stride=(1, 1)),
            MaxPool((2, 2)),
            conv_bn((3, 3), 128 => 256, relu; pad=(1, 1), stride=(1, 1)),
            conv_bn((3, 3), 256 => 256, relu; pad=(1, 1), stride=(1, 1)),
            conv_bn((3, 3), 256 => 256, relu; pad=(1, 1), stride=(1, 1)),
            MaxPool((2, 2)),
            conv_bn((3, 3), 256 => 512, relu; pad=(1, 1), stride=(1, 1)),
            conv_bn((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)),
            conv_bn((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)),
            MaxPool((2, 2)),
            conv_bn((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)),
            conv_bn((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)),
            conv_bn((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)),
            MaxPool((2, 2)); disable_optimizations=true),
        FlattenLayer(),
        Chain(Dense(512, 4096, relu), Dropout(0.5f0), Dense(4096, 4096, relu),
            Dropout(0.5f0), Dense(4096, 10); name="Classifier"); disable_optimizations=true)

    flux_model = () -> Flux.Chain(
        Flux.Conv((3, 3), 3 => 64, relu; pad=(1, 1), stride=(1, 1)),
        Flux.BatchNorm(64), Flux.Conv((3, 3), 64 => 64, relu; pad=(1, 1), stride=(1, 1)),
        Flux.BatchNorm(64), Flux.MaxPool((2, 2)),
        Flux.Conv((3, 3), 64 => 128, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(128),
        Flux.Conv((3, 3), 128 => 128, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(128),
        Flux.MaxPool((2, 2)),
        Flux.Conv((3, 3), 128 => 256, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(256),
        Flux.Conv((3, 3), 256 => 256, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(256),
        Flux.Conv((3, 3), 256 => 256, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(256),
        Flux.MaxPool((2, 2)),
        Flux.Conv((3, 3), 256 => 512, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(512),
        Flux.Conv((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(512),
        Flux.Conv((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(512),
        Flux.MaxPool((2, 2)),
        Flux.Conv((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(512),
        Flux.Conv((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(512),
        Flux.Conv((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)), Flux.BatchNorm(512),
        Flux.MaxPool((2, 2)), Flux.flatten, Flux.Dense(512, 4096, relu), Flux.Dropout(0.5),
        Flux.Dense(4096, 4096, relu), Flux.Dropout(0.5), Flux.Dense(4096, 10))

    for bsize in (2, 16, 64)
        benchmark_forward_pass(
            "VGG16", "(32, 32, 3, $bsize)", vgg16, (32, 32, 3, bsize); flux_model)
        benchmark_reverse_pass(
            "VGG16", "(32, 32, 3, $bsize)", (AutoTracker(), AutoZygote()),
            vgg16, (32, 32, 3, bsize); flux_model)
    end

    return
end

add_vgg_benchmarks!()
