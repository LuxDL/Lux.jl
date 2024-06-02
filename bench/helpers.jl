function flatten_benchmark(results, prefix="", flat_results=[], depth=0)
    for (key, value) in results
        if value isa BenchmarkGroup
            flat_results = flatten_benchmark(value, "$prefix$key/", flat_results, depth + 1)
        elseif value isa Trial
            BenchmarkTools.rmskew!(value)

            push!(flat_results,
                Dict("benchmark" => "$prefix$key", "median" => median(value).time / 1e9,
                    "mean" => mean(value).time / 1e9, "min" => minimum(value).time / 1e9,
                    "max" => maximum(value).time / 1e9,
                    "allocs" => value.allocs, "gctime" => median(value).gctime))
        else
            error("Unexpected value type: $(typeof(value))")
        end
    end

    depth > 0 && return flat_results

    return Dict("commit" => readchomp(`$(git()) rev-parse HEAD`),
        "branch" => readchomp(`$(git()) rev-parse --abbrev-ref HEAD`),
        "julia_cmd" => string(Base.julia_cmd()),
        "date" => string(Dates.now()), "results" => flat_results)
end

function benchmark_forward_pass(tag::String, end_tag::String, model, x_dims;
        simple_chains=nothing, flux_model=nothing)
    benchmark_forward_pass(
        tag, end_tag, model, x_dims, LuxCPUDevice(); simple_chains, flux_model)

    if LuxCUDA.functional()
        benchmark_forward_pass(tag, end_tag, model, x_dims, LuxCUDADevice(); flux_model)
    else
        @warn "CUDA is not functional. Skipping..."
    end

    return
end

function benchmark_forward_pass(tag::String, end_tag::String, model, x_dims,
        ::LuxCPUDevice; simple_chains=nothing, flux_model=nothing)
    SUITE[tag]["cpu"]["forward"]["NamedTuple"][end_tag] = @benchmarkable Lux.apply(
        $model, x, ps_nt, st_test) setup=((x, ps_nt, st) = general_setup($model, $x_dims); st_test = Lux.testmode(st))

    SUITE[tag]["cpu"]["forward"]["ComponentArray"][end_tag] = @benchmarkable Lux.apply(
        $model, x, ps_ca, st_test) setup=((x, ps_nt, st) = general_setup($model, $x_dims); ps_ca = ComponentArray(ps_nt); st_test = Lux.testmode(st))

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

function benchmark_forward_pass(
        tag::String, end_tag::String, model, x_dims, dev::LuxCUDADevice; flux_model=nothing)
    # TODO: Needs to be implemented
end

function benchmark_reverse_pass(tag::String, end_tag::String, backends, model,
        x_dims; simple_chains=nothing, flux_model=nothing)
    benchmark_reverse_pass(
        tag, end_tag, backends, model, x_dims, LuxCPUDevice(); simple_chains, flux_model)

    if LuxCUDA.functional()
        benchmark_reverse_pass(
            tag, end_tag, backends, model, x_dims, LuxCUDADevice(); flux_model)
    else
        @warn "CUDA is not functional. Skipping..."
    end

    return
end

function benchmark_reverse_pass(tag::String, end_tag::String, backends, model, x_dims,
        dev; simple_chains=nothing, flux_model=nothing)
    for backend in backends
        benchmark_reverse_pass(tag, end_tag, backend, model, x_dims, dev)
    end

    if simple_chains !== nothing
        benchmark_reverse_pass_simple_chains(
            tag, end_tag, AutoZygote(), simple_chains(model), x_dims, dev)
    end

    if flux_model !== nothing
        benchmark_reverse_pass_flux(tag, end_tag, AutoZygote(), flux_model, x_dims, dev)
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

sumabsapply(model, x, p, st) = sum(abs2, first(Lux.apply(model, x, p, st)))

function benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoEnzyme, model, x_dims, ::LuxCPUDevice)
    SUITE[tag]["cpu"]["reverse"]["Enzyme"][end_tag] = @benchmarkable Enzyme.autodiff(
        Enzyme.Reverse, $sumabsapply, Enzyme.Active, Enzyme.Duplicated($model, dmodel),
        Enzyme.Const(x), Enzyme.Const(ps), Enzyme.Const(st)) setup=begin
        (x, ps, st) = general_setup($model, $x_dims)
        dmodel = Enzyme.make_zero($model)
        # Force jit compilation in initial run
        Enzyme.autodiff(
            Enzyme.Reverse, $sumabsapply, Enzyme.Active, Enzyme.Duplicated($model, dmodel),
            Enzyme.Const(x), Enzyme.Const(ps), Enzyme.Const(st))
    end
    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoEnzyme, model, x_dims, ::LuxCUDADevice)
    @warn "Enzyme supports CUDA but we haven't implemented it yet. Skipping..."
    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoTapir, model, x_dims, ::LuxCPUDevice)
    @warn "Tapir supports CPU but we haven't implemented it yet. Skipping..."
    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoTapir, model, x_dims, ::LuxCUDADevice)
    @warn "Tapir supports CUDA but we haven't implemented it yet. Skipping..."
    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoTracker, model, x_dims, ::LuxCPUDevice)
    SUITE[tag]["cpu"]["reverse"]["Tracker"][end_tag] = @benchmarkable begin
        ps_tracked = fmap(Tracker.param, ps)
        x_tracked = Tracker.param(x)
        loss = sum(abs2, first(Lux.apply($model, x_tracked, ps_tracked, st)))
        Tracker.back!(loss)
    end setup=begin
        (x, ps, st) = general_setup($model, $x_dims)
    end
    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoTracker, model, x_dims, ::LuxCUDADevice)
    @warn "Tracker supports CUDA but we haven't implemented it yet. Skipping..."
    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, ad::AutoReverseDiff, model, x_dims, ::LuxCPUDevice)
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
        SUITE[tag]["cpu"]["reverse"]["ReverseDiff"][end_tag] = @benchmarkable begin
            tape = ReverseDiff.InstructionTape()
            ∂ps = fmap(zero, ps)
            ps_tracked = fmap((p, g) -> ReverseDiff.TrackedArray(p, g, tape), ps, ∂ps)
            ∂x = zero(x)
            x_tracked = ReverseDiff.TrackedArray(x, ∂x, tape)
            loss = sum(abs2, first(Lux.apply($model, x_tracked, ps_tracked, st)))
            loss.deriv = true
            ReverseDiff.reverse_pass!(tape)
        end setup=begin
            (x, ps, st) = general_setup($model, $x_dims)
        end
    end
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, ad::AutoReverseDiff, model, x_dims, ::LuxCUDADevice)
    @warn "ReverseDiff doesn't support CUDA. Skipping..."
    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoZygote, model, x_dims, ::LuxCPUDevice)
    SUITE[tag]["cpu"]["reverse"]["Zygote"][end_tag] = @benchmarkable Zygote.gradient(
        f, $model, x, ps, st) setup=begin
        (x, ps, st) = general_setup($model, $x_dims)
        f = @closure((model, x, p, st)->sum(abs2, first(Lux.apply(model, x, p, st))))
    end
    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoZygote, model, x_dims, ::LuxCUDADevice)
    @warn "Zygote supports CUDA but we haven't implemented it yet. Skipping..."
    return
end

function benchmark_reverse_pass_simple_chains(
        tag::String, end_tag::String, ::AutoZygote, model, x_dims, ::LuxCPUDevice)
    SUITE[tag]["cpu"]["reverse"]["SimpleChains"][end_tag] = @benchmarkable Zygote.gradient(
        f, $model, x, ps, st) setup=begin
        (x, ps, st) = general_setup($model, $x_dims)
        f = @closure((model, x, p, st)->sum(abs2, first(Lux.apply(model, x, p, st))))
    end
    return
end

function benchmark_reverse_pass(
        tag::String, end_tag::String, ::AutoZygote, model, x_dims, ::LuxCUDADevice)
    @warn "SimpleChains doesn't support CUDA. Skipping..."
    return
end

function benchmark_reverse_pass_flux(
        tag::String, end_tag::String, ::AutoZygote, model, x_dims, ::LuxCPUDevice)
    SUITE[tag]["cpu"]["reverse"]["Flux"][end_tag] = @benchmarkable Zygote.gradient(f, m, x) setup=begin
        x = randn(StableRNG(0), Float32, $x_dims)
        m = $(model)()
        f = @closure((m, x)->sum(abs2, m(x)))
    end
    return
end

function benchmark_reverse_pass_flux(
        tag::String, end_tag::String, ::AutoZygote, model, x_dims, ::LuxCUDADevice)
    @warn "Flux supports CUDA but we haven't implemented it yet. Skipping..."
    return
end
