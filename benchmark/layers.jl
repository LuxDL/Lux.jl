function add_dense_benchmarks!()
    for n in (2, 20, 200, 2000), b in (128,), act in (identity, relu)
        layer = Dense(n => n, act)
        simple_chains = n ≤ 200 ? Lux.ToSimpleChainsAdaptor((static(n),)) : nothing
        flux_model = () -> Flux.Dense(n => n, act)
        benchmark_forward_pass(
            "Dense($n => $n, $(act))", "($n, $b)", layer, (n, 128); simple_chains, flux_model)
        benchmark_reverse_pass("Dense($n => $n, $(act))",
            "($n, $b)",
            (AutoTracker(), AutoReverseDiff(),
                AutoReverseDiff(true), AutoZygote(), AutoEnzyme()),
            layer,
            (n, b);
            simple_chains,
            flux_model)
    end

    return
end

function add_conv_benchmarks!()
    for ch in (1, 3, 16, 64), b in (128,), act in (identity, relu)
        layer = Conv((3, 3), ch => ch, act)
        simple_chains = ch ≤ 16 ?
                        Lux.ToSimpleChainsAdaptor((static(64), static(64), static(ch))) :
                        nothing
        flux_model = () -> Flux.Conv((3, 3), ch => ch, act)
        benchmark_forward_pass("Conv((3, 3), $ch => $ch, $act)", "(64, 64, $ch, $b)",
            layer, (64, 64, ch, 128); simple_chains, flux_model)
        benchmark_reverse_pass("Conv((3, 3), $ch => $ch, $act)",
            "(64, 64, $ch, $b)",
            (AutoTracker(), AutoReverseDiff(),
                AutoReverseDiff(true), AutoZygote(), AutoEnzyme()),
            layer,
            (64, 64, ch, b);
            simple_chains,
            flux_model)
    end
end

add_dense_benchmarks!()
add_conv_benchmarks!()
