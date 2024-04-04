function add_vgg_benchmarks()
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

    x, ps, st = general_setup(vgg16, (32, 32, 3, 1))
    benchmark_forward_pass("vgg16 -- batchsize = 1", vgg16, x, ps, st)

    x, ps, st = general_setup(vgg16, (32, 32, 3, 16))
    benchmark_forward_pass("vgg16 -- batchsize = 16", vgg16, x, ps, st)

    x, ps, st = general_setup(vgg16, (32, 32, 3, 64))
    benchmark_forward_pass("vgg16 -- batchsize = 64", vgg16, x, ps, st)

    return
end

add_vgg_benchmarks()
