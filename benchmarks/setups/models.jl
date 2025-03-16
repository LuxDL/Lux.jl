function setup_vgg16_benchmarks!(
        suite::BenchmarkGroup, cpu_or_gpu::String,
        final_backend::String, dev::AbstractDevice
)
    # Julia Fails to Specialize on Large NTs, so split it up
    function conv_bn(ksize, (in_chs, out_chs)::Pair, args...; kwargs...)
        return Chain(
            Conv(ksize, in_chs => out_chs, args...; kwargs...), BatchNorm(out_chs);
            name="ConvBN"
        )
    end

    vgg16 = Chain(
        Chain(
            conv_bn((3, 3), 3 => 64, relu; pad=(1, 1), stride=(1, 1)),
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
            MaxPool((2, 2))
        ),
        FlattenLayer(),
        Chain(
            Dense(512, 4096, relu), Dropout(0.5f0), Dense(4096, 4096, relu),
            Dropout(0.5f0), Dense(4096, 10); name="Classifier"
        )
    )

    for bsize in (32, 64, 128)
        setup_forward_pass_benchmark!(
            suite, "vgg16(32, 32, 3, $bsize)",
            cpu_or_gpu, final_backend, vgg16, (32, 32, 3, bsize), dev
        )

        setup_reverse_pass_benchmark!(
            suite, "vgg16(32, 32, 3, $bsize)",
            cpu_or_gpu, final_backend, [AutoZygote()], vgg16, (32, 32, 3, bsize), dev
        )
    end
    return
end

function setup_mlp_benchmarks!(
        suite::BenchmarkGroup, cpu_or_gpu::String,
        final_backend::String, dev::AbstractDevice
)
    for act in (tanh, relu, gelu)
        mlp = Chain(
            Dense(32 => 256), BatchNorm(256, act),
            [Chain(Dense(256 => 256), BatchNorm(256, act))
             for _ in 1:5]...,
            Dense(256 => 10)
        )

        setup_forward_pass_benchmark!(
            suite, "mlp7layer_bn($act)(32 x 256)",
            cpu_or_gpu, final_backend, mlp, (32, 256), dev
        )

        setup_reverse_pass_benchmark!(
            suite, "mlp7layer_bn($act)(32 x 256)",
            cpu_or_gpu, final_backend, [AutoZygote(), AutoEnzyme()], mlp, (32, 256), dev
        )
    end
    return
end

function setup_lenet_benchmarks!(
        suite::BenchmarkGroup, cpu_or_gpu::String,
        final_backend::String, dev::AbstractDevice
)
    lenet = Chain(
        Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
        Dense(256, 120, relu), Dense(120, 84, relu), Dense(84, 10)
    )

    for bsize in (32, 64, 128)
        setup_forward_pass_benchmark!(
            suite, "lenet(28, 28, 1, $bsize)",
            cpu_or_gpu, final_backend, lenet, (28, 28, 1, bsize), dev
        )

        setup_reverse_pass_benchmark!(
            suite, "lenet(28, 28, 1, $bsize)",
            cpu_or_gpu, final_backend, [AutoZygote()], lenet, (28, 28, 1, bsize), dev
        )
    end
    return
end
