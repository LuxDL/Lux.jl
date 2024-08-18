function add_dense_benchmarks!(suite::BenchmarkGroup, group::String)
    for n in (16, 128, 512), act in (identity, relu, gelu)
        layer = Dense(n => n, act)

        benchmark_forward_pass!(suite, group, "Dense($n => $n, $act)($n x 128)",
            layer, (n, 128))

        benchmark_reverse_pass!(
            suite, group, [AutoZygote(), AutoEnzyme()], "Dense($n => $n, $act)($n x 128)",
            layer, (n, 128))
    end
    return
end

function add_conv_benchmarks!(suite::BenchmarkGroup, group::String)
    for ch in (2, 4, 32, 64), act in (identity, relu, gelu)
        layer = Conv((3, 3), ch => ch, act)

        benchmark_forward_pass!(
            suite, group, "Conv((3, 3), $ch => $ch, $act)(64 x 64 x $ch x 128)",
            layer, (64, 64, ch, 128))

        benchmark_reverse_pass!(suite, group, [AutoZygote(), AutoEnzyme()],
            "Conv((3, 3), $ch => $ch, $act)(64 x 64 x $ch x 128)",
            layer, (64, 64, ch, 128))
    end
    return
end
