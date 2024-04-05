# TODO: Special Handling for GPU Arrays with @sync
function benchmark_forward_pass(
        tag::String, end_tag::String, model, x_dims; simple_chains=nothing,
        flux_model=nothing)
    SUITE[tag]["cpu"]["forward"]["NamedTuple"][end_tag] = @benchmarkable Lux.apply(
        $model, x, ps_nt, st) setup=((x, ps_nt, st) = general_setup($model, $x_dims))

    SUITE[tag]["cpu"]["forward"]["ComponentArray"][end_tag] = @benchmarkable Lux.apply(
        $model, x, ps_ca, st) setup=((x, ps_nt, st) = general_setup($model, $x_dims); ps_ca = ComponentArray(ps_nt))

    if simple_chains !== nothing
        simple_chains_model = simple_chains(model)
        SUITE[tag]["cpu"]["forward"]["SimpleChains"][end_tag] = @benchmarkable Lux.apply(
            $simple_chains_model, x, ps_simple_chains, st_simple_chains) setup=((x, ps_simple_chains, st_simple_chains) = general_setup(
            $simple_chains_model, $x_dims))
    end

    if flux_model !== nothing
        SUITE[tag]["cpu"]["forward"]["Flux"][end_tag] = @benchmarkable fmodel(x) setup=(x = randn(
            StableRNG(0), Float32, $x_dims);
        fmodel = $(flux_model()))
    end

    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, backends, model, x_dims;
        simple_chains=nothing, flux_model=nothing)
    for backend in backends
        __benchmark_reverse_pass(tag, end_tag, backend, model, x_dims)
    end

    if simple_chains !== nothing
        simple_chains_model = simple_chains(model)
        __benchmark_reverse_pass_simple_chains(
            tag, end_tag, AutoZygote(), simple_chains_model, x_dims)
    end

    if flux_model !== nothing
        __benchmark_reverse_pass_flux(tag, end_tag, AutoZygote(), flux_model, x_dims)
    end

    return
end

function general_setup(model, x_dims)
    rng = StableRNG(0)
    ps, st = Lux.setup(rng, model)
    x_dims === nothing && return ps, st
    x = randn(rng, Float32, x_dims)
    return x, ps, st
end

# TODO: Remove these once DifferentiationInterface has been released
function __benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoEnzyme, model, x_dims)
    # TODO: Enable this. But enzyme doesn't handle closures well it seems...
    # SUITE[tag]["cpu"]["reverse"]["Enzyme"][end_tag] = @benchmarkable Enzyme.gradient(
    #     $Enzyme.Reverse, $f, $x)
    return error("Enzyme backend hasn't been implemented yet.")
end
function __benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoTapir, model, x_dims)
    SUITE[tag]["cpu"]["reverse"]["Tapir"][end_tag] = @benchmarkable Tapir.value_and_pullback!!(
        trrule, 1.0f0, f, ps_ca) setup=begin
        (x, ps, st) = general_setup($model, $x_dims)
        ps_ca = ComponentArray(ps)
        f = @closure(p->sum(abs2, first(Lux.apply($model, x, p, st))))
        trrule = Tapir.build_rrule(f, ps_ca)
    end
    return
end
function __benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoTracker, model, x_dims)
    SUITE[tag]["cpu"]["reverse"]["Tracker"][end_tag] = @benchmarkable Tracker.gradient(
        f, ps_ca) setup=begin
        (x, ps, st) = general_setup($model, $x_dims)
        ps_ca = ComponentArray(ps)
        f = @closure(p->sum(abs2, first(Lux.apply($model, x, p, st))))
    end
    return
end
function __benchmark_reverse_pass(
        tag::String, end_tag::String, ad::AutoReverseDiff, model, x_dims)
    if ad.compile
        SUITE[tag]["cpu"]["reverse"]["ReverseDiff (compiled)"][end_tag] = @benchmarkable ReverseDiff.gradient!(
            ∂ps, tape, ps_ca) setup=begin
            (x, ps, st) = general_setup($model, $x_dims)
            ps_ca = ComponentArray(ps)
            ∂ps = similar(ps_ca)
            f = @closure(p->sum(abs2, first(Lux.apply($model, x, p, st))))
            tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, ps_ca))
        end
    else
        SUITE[tag]["cpu"]["reverse"]["ReverseDiff"][end_tag] = @benchmarkable ReverseDiff.gradient(
            f, ps_ca) setup=begin
            (x, ps, st) = general_setup($model, $x_dims)
            ps_ca = ComponentArray(ps)
            f = @closure(p->sum(abs2, first(Lux.apply($model, x, p, st))))
        end
    end
end
function __benchmark_reverse_pass(tag::String, end_tag::String, ::AutoZygote, model, x_dims)
    SUITE[tag]["cpu"]["reverse"]["Zygote"][end_tag] = @benchmarkable Zygote.gradient(
        f, ps_ca) setup=begin
        (x, ps, st) = general_setup($model, $x_dims)
        ps_ca = ComponentArray(ps)
        f = @closure(p->sum(abs2, first(Lux.apply($model, x, p, st))))
    end
    return
end
function __benchmark_reverse_pass_simple_chains(
        tag::String, end_tag::String, ::AutoZygote, model, x_dims)
    SUITE[tag]["cpu"]["reverse"]["SimpleChains"][end_tag] = @benchmarkable Zygote.gradient(
        f, ps) setup=begin
        (x, ps, st) = general_setup($model, $x_dims)
        f = @closure(p->sum(abs2, first(Lux.apply($model, x, p, st))))
    end
    return
end
function __benchmark_reverse_pass_flux(
        tag::String, end_tag::String, ::AutoZygote, model, x_dims)
    SUITE[tag]["cpu"]["reverse"]["Flux"][end_tag] = @benchmarkable Zygote.gradient(
        f, m) setup=begin
        x = randn(StableRNG(0), Float32, $x_dims)
        m = $(model)()
        f = @closure(m->sum(abs2, m(x)))
    end
    return
end
