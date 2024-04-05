function add_dense_benchmarks!()
    for n in (2, 20, 200, 2000)
        layer = Dense(n => n)
        simple_chains = n ≤ 200 ? Lux.ToSimpleChainsAdaptor((static(n),)) : nothing
        flux_model = () -> Flux.Dense(n => n)
        benchmark_forward_pass(
            "Dense($n => $n)", "($n, 128)", layer, (n, 128); simple_chains, flux_model)
        benchmark_reverse_pass(
            "Dense($n => $n)", "($n, 128)",
            (AutoTracker(), AutoReverseDiff(), AutoReverseDiff(true), AutoZygote()),
            layer, (n, 128); simple_chains, flux_model)
    end

    return
end

function add_conv_benchmarks!()
    for ch in (1, 3, 16, 64)
        layer = Conv((3, 3), ch => ch)
        simple_chains = ch ≤ 16 ?
                        Lux.ToSimpleChainsAdaptor((static(64), static(64), static(ch))) :
                        nothing
        flux_model = () -> Flux.Conv((3, 3), ch => ch)
        benchmark_forward_pass("Conv((3, 3), $ch => $ch)", "(64, 64, $ch, 128)",
            layer, (64, 64, ch, 128); simple_chains, flux_model)
        benchmark_reverse_pass("Conv((3, 3), $ch => $ch)", "(64, 64, $ch, 128)",
            (AutoTracker(), AutoReverseDiff(), AutoReverseDiff(true), AutoZygote()),
            layer, (64, 64, ch, 128); simple_chains, flux_model)
    end
end

add_dense_benchmarks!()
add_conv_benchmarks!()
