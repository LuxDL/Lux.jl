# TODO: Special Handling for GPU Arrays with @sync
function benchmark_forward_pass(tag::String, end_tag::String, model, x, ps_nt::NamedTuple,
        st; simple_chains=nothing)
    SUITE[tag]["cpu"]["forward"]["NamedTuple"][end_tag] = @benchmarkable Lux.apply(
        $model, $x, $ps_nt, $st)

    ps_ca = ComponentArray(ps_nt)
    SUITE[tag]["cpu"]["forward"]["ComponentArray"][end_tag] = @benchmarkable Lux.apply(
        $model, $x, $ps_ca, $st)

    if simple_chains !== nothing
        simple_chains_model = simple_chains(model)
        ps_simple_chains, st_simple_chains = general_setup(simple_chains_model, nothing)
        SUITE[tag]["cpu"]["forward"]["SimpleChains"][end_tag] = @benchmarkable Lux.apply(
            $simple_chains_model, $x, $ps_simple_chains, $st_simple_chains)
    end

    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, backends, model, x, ps_nt::NamedTuple, st;
        simple_chains=nothing)
    # Not everyone can handle NamedTuples so convert to ComponentArray
    __f = @closure ps -> sum(abs2, first(Lux.apply(model, x, ps, st)))
    ps_ca = ComponentArray(ps_nt)

    for backend in backends
        __benchmark_reverse_pass(tag, end_tag, backend, __f, ps_ca)
    end

    if simple_chains !== nothing
        simple_chains_model = simple_chains(model)
        ps_simple_chains, st_simple_chains = general_setup(simple_chains_model, nothing)
        __f = @closure ps -> sum(
            abs2, first(Lux.apply(simple_chains_model, x, ps, st_simple_chains)))
        __benchmark_reverse_pass_simple_chains(
            tag, end_tag, AutoZygote(), __f, ps_simple_chains)
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
        tag::String, end_tag::String, ::AutoEnzyme, f::F, x; kwargs...) where {F}
    # TODO: Enable this. But enzyme doesn't handle closures well it seems...
    # SUITE[tag]["cpu"]["reverse"]["Enzyme"][end_tag] = @benchmarkable Enzyme.gradient(
    #     $Enzyme.Reverse, $f, $x)
    return error("Enzyme backend hasn't been implemented yet.")
end
function __benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoTapir, f::F, x; kwargs...) where {F}
end
function __benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoTracker, f::F, x; kwargs...) where {F}
    SUITE[tag]["cpu"]["reverse"]["Tracker"][end_tag] = @benchmarkable Tracker.gradient(
        $f, $x)
    return
end
function __benchmark_reverse_pass(
        tag::String, end_tag::String, ad::AutoReverseDiff, f::F, x; kwargs...) where {F}
    if ad.compile
        SUITE[tag]["cpu"]["reverse"]["ReverseDiff (compiled)"][end_tag] = @benchmarkable ReverseDiff.gradient!(
            ∂x, tape, $x) setup=(∂x = similar($x);
        tape = ReverseDiff.compile(ReverseDiff.GradientTape($f, $x)))
    else
        SUITE[tag]["cpu"]["reverse"]["ReverseDiff"][end_tag] = @benchmarkable ReverseDiff.gradient(
            $f, $x)
    end
end
function __benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoZygote, f::F, x; kwargs...) where {F}
    SUITE[tag]["cpu"]["reverse"]["Zygote"][end_tag] = @benchmarkable Zygote.gradient(
        $f, $x)
    return
end
function __benchmark_reverse_pass_simple_chains(
        tag::String, end_tag::String, ::AutoZygote, f::F, x; kwargs...) where {F}
    SUITE[tag]["cpu"]["reverse"]["SimpleChains"][end_tag] = @benchmarkable Zygote.gradient(
        $f, $x)
    return
end
