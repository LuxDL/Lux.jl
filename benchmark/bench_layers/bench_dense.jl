include("../utils.jl")

const suite = BenchmarkGroup()

for n in (2, 20, 200, 2000), b in (128,), act in (identity, relu)
    layer = Dense(n => n, act)
    simple_chains = n ≤ 200 ? ToSimpleChainsAdaptor((static(n),)) : nothing
    flux_model = () -> Flux.Dense(n => n, act)

    benchmark_forward_pass!(suite, "Dense($n => $n, $(act))", "($n, $b)",
        layer, (n, 128); simple_chains, flux_model)

    benchmark_reverse_pass!(suite, "Dense($n => $n, $(act))", "($n, $b)",
        (AutoTracker(), AutoReverseDiff(),
            AutoReverseDiff(; compile=true), AutoZygote(), AutoEnzyme()),
        layer, (n, b); simple_chains, flux_model)
end
