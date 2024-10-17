function setup_dense_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
        final_backend::String, dev::AbstractDevice)
    for n in (16, 128, 512), act in (identity, relu, gelu)

        layer = Dense(n => n, act)

        setup_forward_pass_benchmark!(suite, "Dense($n => $n, $act)($n x 128)",
            cpu_or_gpu, final_backend, layer, (n, 128), dev)

        setup_reverse_pass_benchmark!(suite, "Dense($n => $n, $act)($n x 128)",
            cpu_or_gpu, final_backend, [AutoZygote(), AutoEnzyme()], layer, (n, 128), dev)
    end
end

function setup_conv_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
        final_backend::String, dev::AbstractDevice)
    for ch in (2, 4, 32, 64), act in (identity, relu, gelu)

        layer = Conv((3, 3), ch => ch, act)

        setup_forward_pass_benchmark!(
            suite, "Conv((3, 3), $ch => $ch, $act)(64 x 64 x $ch x 128)",
            cpu_or_gpu, final_backend, layer, (64, 64, ch, 128), dev)

        setup_reverse_pass_benchmark!(
            suite, "Conv((3, 3), $ch => $ch, $act)(64 x 64 x $ch x 128)",
            cpu_or_gpu, final_backend, [AutoZygote(), AutoEnzyme()], layer, (
                64, 64, ch, 128), dev)
    end
end
