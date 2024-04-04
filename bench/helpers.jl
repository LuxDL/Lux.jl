# TODO: Special Handling for GPU Arrays with @sync
function benchmark_forward_pass(tag::String, end_tag::String, model, x, ps_nt::NamedTuple,
        st)
    SUITE[tag]["cpu"]["forward"]["NamedTuple"][end_tag] = @benchmarkable Lux.apply(
        $model, $x, $ps_nt, $st)

    ps_ca = ComponentArray(ps_nt)
    SUITE[tag]["cpu"]["forward"]["ComponentArray"][end_tag] = @benchmarkable Lux.apply(
        $model, $x, $ps_ca, $st)

    return
end

function general_setup(model, x_dims)
    rng = StableRNG(0)
    ps, st = Lux.setup(rng, model)
    x = randn(rng, Float32, x_dims)
    return x, ps, st
end
