function add_vgg_benchmarks!()
    vgg16 = Chain(Conv((3, 3), 3 => 64, relu; pad=(1, 1), stride=(1, 1)), BatchNorm(64),
        Conv((3, 3), 64 => 64, relu; pad=(1, 1), stride=(1, 1)), BatchNorm(64),
        MaxPool((2, 2)), Conv((3, 3), 64 => 128, relu; pad=(1, 1), stride=(1, 1)),
        BatchNorm(128), Conv((3, 3), 128 => 128, relu; pad=(1, 1), stride=(1, 1)),
        BatchNorm(128), MaxPool((2, 2)),
        Conv((3, 3), 128 => 256, relu; pad=(1, 1), stride=(1, 1)),
        BatchNorm(256), Conv((3, 3), 256 => 256, relu; pad=(1, 1), stride=(1, 1)),
        BatchNorm(256), Conv((3, 3), 256 => 256, relu; pad=(1, 1), stride=(1, 1)),
        BatchNorm(256), MaxPool((2, 2)),
        Conv((3, 3), 256 => 512, relu; pad=(1, 1), stride=(1, 1)), BatchNorm(512),
        Conv((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)), BatchNorm(512),
        Conv((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)), BatchNorm(512),
        MaxPool((2, 2)), Conv((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)),
        BatchNorm(512), Conv((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)),
        BatchNorm(512), Conv((3, 3), 512 => 512, relu; pad=(1, 1), stride=(1, 1)),
        BatchNorm(512), MaxPool((2, 2)), FlattenLayer(), Dense(512, 4096, relu),
        Dropout(0.5), Dense(4096, 4096, relu), Dropout(0.5), Dense(4096, 10))

    for bsize in (1, 16, 64)
        x, ps, st = general_setup(vgg16, (32, 32, 3, bsize))
        benchmark_forward_pass("vgg16", "(32, 32, 3, $bsize)", vgg16, x, ps, st)
    end

    return
end

add_vgg_benchmarks!()
