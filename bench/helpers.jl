# TODO: Special Handling for GPU Arrays with @sync
function benchmark_forward_pass(tag::String, end_tag::String, model, x, ps_nt::NamedTuple,
        st; simple_chains = nothing)
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
        tag::String, end_tag::String, backends::NTuple, model, x, ps_nt::NamedTuple, st)
    # Not everyone can handle NamedTuples so convert to ComponentArray
    __f = @closure ps -> sum(abs2, first(Lux.apply(model, x, ps, st)))
    ps_ca = ComponentArray(ps_nt)
    return
end

function general_setup(model, x_dims)
    rng = StableRNG(0)
    ps, st = Lux.setup(rng, model)
    x_dims === nothing && return ps, st
    x = randn(rng, Float32, x_dims)
    return x, ps, st
end

@inline __typein(::Type{T}, x) where {T} = any(Base.Fix2(isa, T), x)

# TODO: Remove these once DifferentiationInterface has been released
function __benchmark_tapir_reverse_pass(tag::String, end_tag::String, f::F, x) where {F}
end
function __benchmark_tracker_reverse_pass(tag::String, end_tag::String, f::F, x) where {F}
end
function __benchmark_enzyme_reverse_pass(tag::String, end_tag::String, f::F, x) where {F}
end
function __benchmark_reversediff_reverse_pass(
        tag::String, end_tag::String, f::F, x) where {F}
end
function __benchmark_zygote_reverse_pass(tag::String, end_tag::String, f::F, x) where {F}
end
