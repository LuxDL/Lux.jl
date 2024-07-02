include("../utils.jl")

const suite = BenchmarkGroup()

for ch in (1, 3, 16, 64), b in (128,), act in (identity, relu)
    layer = Conv((3, 3), ch => ch, act)
    simple_chains = ch ≤ 16 ?
                    Lux.ToSimpleChainsAdaptor((static(64), static(64), static(ch))) :
                    nothing
    flux_model = () -> Flux.Conv((3, 3), ch => ch, act)
    benchmark_forward_pass!(suite, "Conv((3, 3), $ch => $ch, $act)", "(64, 64, $ch, $b)",
        layer, (64, 64, ch, 128); simple_chains, flux_model)
    benchmark_reverse_pass!(suite, "Conv((3, 3), $ch => $ch, $act)", "(64, 64, $ch, $b)",
        (AutoTracker(), AutoReverseDiff(), AutoReverseDiff(; compile=true), AutoZygote(),
            AutoEnzyme()),
        layer, (64, 64, ch, b); simple_chains, flux_model)
end
