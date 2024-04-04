# TODO: Special Handling for GPU Arrays with @sync
function benchmark_forward_pass(tag::String, model, x, ps, st)
    SUITE[tag]["forward"]["default"] = @benchmarkable Lux.apply($model, $x, $ps, $st)

    ps_ca = ComponentArray(ps)
    SUITE[tag]["forward"]["ComponentArray"] = @benchmarkable Lux.apply(
        $model, $x, $ps_ca, $st)

    return
end

function general_setup(model, x_dims)
    rng = StableRNG(0)
    ps, st = Lux.setup(rng, model)
    x = randn(rng, Float32, x_dims)
    return x, ps, st
end
