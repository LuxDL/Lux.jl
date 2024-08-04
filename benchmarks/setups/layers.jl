function add_dense_benchmarks!(suite::BenchmarkGroup, group::String)
    for n in (16,)
        layer = Dense(n => n)
        simple_chains = n ≤ 200 ? Lux.ToSimpleChainsAdaptor((static(n),)) : nothing
        flux_model = () -> Flux.Dense(n => n)

        benchmark_forward_pass!(suite, group, "Dense($n => $n)($n x 128)",
            layer, (n, 128); simple_chains, flux_model)
        benchmark_reverse_pass!(
            suite, group, [AutoZygote(), AutoEnzyme()], "Dense($n => $n)($n x 128)",
            layer, (n, 128); simple_chains, flux_model)
    end
    return
end
