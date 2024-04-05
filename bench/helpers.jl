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
