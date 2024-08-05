function add_dense_benchmarks!(suite::BenchmarkGroup, group::String)
    for n in (16,) #  128, 512, 2048), act in (identity, relu, gelu)
        act = identity
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
