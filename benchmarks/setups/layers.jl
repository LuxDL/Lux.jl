function add_dense_benchmarks!(suite::BenchmarkGroup, group::String)
    for n in (16, 128, 512,), act in (identity, relu, gelu)
        layer = Dense(n => n, act)
        simple_chains = n ≤ 200 ? Lux.ToSimpleChainsAdaptor((static(n),)) : nothing
        flux_model = () -> Flux.Dense(n => n, act)

        benchmark_forward_pass!(suite, group, "Dense($n => $n, $act)($n x 128)",
            layer, (n, 128); simple_chains, flux_model)

        benchmark_reverse_pass!(
            suite, group, [AutoZygote(), AutoEnzyme()], "Dense($n => $n, $act)($n x 128)",
            layer, (n, 128); simple_chains, flux_model)
    end
    return
end

function add_conv_benchmarks!(suite::BenchmarkGroup, group::String)
    for ch in (2, 4, 32, 64), act in (identity, relu, gelu)
        layer = Conv((3, 3), ch => ch, act)
        flux_model = () -> Flux.Conv((3, 3), ch => ch, act)

        simple_chains = ch ≤ 16 ?
                        Lux.ToSimpleChainsAdaptor((static(64), static(64), static(ch))) :
                        nothing

        benchmark_forward_pass!(
            suite, group, "Conv((3, 3), $ch => $ch, $act)(64 x 64 x $ch x 128)",
            layer, (64, 64, ch, 128); simple_chains, flux_model)

        benchmark_reverse_pass!(suite, group, [AutoZygote(), AutoEnzyme()],
            "Conv((3, 3), $ch => $ch, $act)(64 x 64 x $ch x 128)",
            layer, (64, 64, ch, 128); simple_chains, flux_model)
    end
    return
end
