using ADTypes, BenchmarkTools, ComponentArrays, FastClosures, Functors, Logging, Lux,
      LuxCUDA, LuxDeviceUtils, StableRNGs
using SimpleChains: static
using Flux: Flux
using Enzyme, ReverseDiff, Tracker, Zygote

# Macros
## convenience macro to create a benchmark that requires synchronizing the GPU
## Mostly taken from `@benchmarkable` macro in BenchmarkTools.jl
macro async_benchmarkable(dev, ex...)
    core, setup, teardown, quote_vars, quote_vals, params = BenchmarkTools.benchmarkable_parts(ex)
    map!(esc, params, params)

    # extract any variable bindings shared between the core and setup expressions
    setup_vars = isa(setup, Expr) ? BenchmarkTools.collectvars(setup) : []
    core_vars = isa(core, Expr) ? BenchmarkTools.collectvars(core) : []
    out_vars = filter(var -> var in setup_vars, core_vars)

    core_new_expr = quote
        $(core)
        CUDA.synchronize(; blocking=false)
    end

    quote
        if $(esc(dev)) isa LuxCPUDevice
            #! format: off
            BenchmarkTools.generate_benchmark_definition(
                $__module__,
                $(Expr(:quote, out_vars)),
                $(Expr(:quote, setup_vars)),
                $(Expr(:quote, quote_vars)),
                $(esc(Expr(:tuple, Expr.(:quote, quote_vals)...))),
                $(esc(Expr(:quote, core))),
                $(esc(Expr(:quote, setup))),
                $(esc(Expr(:quote, teardown))),
                BenchmarkTools.Parameters($(params...)),
            )
            #! format: on
        elseif $(esc(dev)) isa LuxCUDADevice
            #! format: off
            BenchmarkTools.generate_benchmark_definition(
                $__module__,
                $(Expr(:quote, out_vars)),
                $(Expr(:quote, setup_vars)),
                $(Expr(:quote, quote_vars)),
                $(esc(Expr(:tuple, Expr.(:quote, quote_vals)...))),
                $(Expr(:quote, core_new_expr)),
                $(esc(Expr(:quote, setup))),
                $(esc(Expr(:quote, teardown))),
                BenchmarkTools.Parameters($(params...)),
            )
            #! format: on
        else
            error("Unknown device")
        end
    end
end

function general_setup(model, x_dims, dev=LuxCPUDevice())
    rng = StableRNG(0)
    ps, st = Lux.setup(rng, model) |> dev
    x_dims === nothing && return ps, st
    x = randn(rng, Float32, x_dims) |> dev
    return x, ps, st
end

sumabsapply(model, x, p, st) = sum(abs2, first(Lux.apply(model, x, p, st)))

# Forward Pass
function benchmark_forward_pass!(suite, tag::String, end_tag::String, model, x_dims;
        simple_chains=nothing, flux_model=nothing)
    benchmark_forward_pass!(
        suite, tag, end_tag, model, x_dims, LuxCPUDevice(); simple_chains, flux_model)

    if LuxCUDA.functional()
        benchmark_forward_pass!(
            suite, tag, end_tag, model, x_dims, LuxCUDADevice(); flux_model)
    else
        @warn "CUDA is not functional. Skipping..."
    end

    return
end

function benchmark_forward_pass!(suite, tag::String, end_tag::String, model, x_dims,
        dev; flux_model=nothing, simple_chains=nothing)
    dev_tag = dev isa LuxCPUDevice ? "cpu" : "cuda"

    suite[tag][dev_tag]["forward"]["NamedTuple"][end_tag] = @async_benchmarkable dev begin
        Lux.apply($model, x, ps_nt, st_test)
    end setup=begin
        (x, ps_nt, st) = general_setup($model, $x_dims, $dev)
        st_test = Lux.testmode(st)
    end

    suite[tag][dev_tag]["forward"]["ComponentArray"][end_tag] = @async_benchmarkable dev begin
        Lux.apply($model, x, ps_ca, st_test)
    end setup=begin
        (x, ps_nt, st) = general_setup($model, $x_dims, $dev)
        ps_ca = ComponentArray(ps_nt |> Lux.cpu_device()) |> $dev
        st_test = Lux.testmode(st)
    end

    if simple_chains !== nothing && dev isa LuxCPUDevice
        simple_chains_model = Logging.with_logger(Logging.NullLogger()) do
            simple_chains(model)
        end
        suite[tag][dev_tag]["forward"]["SimpleChains"][end_tag] = @async_benchmarkable dev begin
            Lux.apply($simple_chains_model, x, ps2, st2)
        end setup=begin
            (x, ps2, st2) = general_setup($simple_chains_model, $x_dims)
        end
    end

    if flux_model !== nothing
        flux_dev = dev isa LuxCPUDevice ? Flux.cpu : Flux.gpu
        suite[tag][dev_tag]["forward"]["Flux"][end_tag] = @async_benchmarkable dev begin
            fmodel(x)
        end setup=begin
            x = randn(StableRNG(0), Float32, $x_dims) |> $(flux_dev)
            fmodel = $(flux_model()) |> $(flux_dev)
        end
    end

    return
end

# Reverse Pass
function benchmark_reverse_pass!(suite, tag::String, end_tag::String, backends, model,
        x_dims; simple_chains=nothing, flux_model=nothing)
    benchmark_reverse_pass!(suite, tag, end_tag, backends, model, x_dims, LuxCPUDevice();
        simple_chains, flux_model)

    if LuxCUDA.functional()
        benchmark_reverse_pass!(
            suite, tag, end_tag, backends, model, x_dims, LuxCUDADevice(); flux_model)
    else
        @warn "CUDA is not functional. Skipping..."
    end

    return
end

function benchmark_reverse_pass!(
        suite, tag::String, end_tag::String, backends, model, x_dims,
        dev; simple_chains=nothing, flux_model=nothing)
    for backend in backends
        benchmark_reverse_pass!(suite, tag, end_tag, backend, model, x_dims, dev)
    end

    if simple_chains !== nothing
        simple_chains_model = Logging.with_logger(Logging.NullLogger()) do
            simple_chains(model)
        end
        benchmark_reverse_pass_simple_chains!(
            suite, tag, end_tag, AutoZygote(), simple_chains_model, x_dims, dev)
    end

    if flux_model !== nothing
        benchmark_reverse_pass_flux!(
            suite, tag, end_tag, AutoZygote(), flux_model, x_dims, dev)
    end

    return
end

function benchmark_reverse_pass!(
        suite, tag::String, end_tag::String, ::AutoEnzyme, model, x_dims, ::LuxCPUDevice)
    suite[tag]["cpu"]["reverse"]["Enzyme"][end_tag] = @benchmarkable Enzyme.autodiff(
        Enzyme.Reverse, $sumabsapply, Enzyme.Active, Enzyme.Duplicated($model, dmodel),
        Enzyme.Const(x), Enzyme.Const(ps), Enzyme.Const(st)) setup=begin
        (x, ps, st) = general_setup($model, $x_dims)
        dmodel = Enzyme.make_zero($model)
        Enzyme.autodiff( # Force jit compilation in initial run
            Enzyme.Reverse, $sumabsapply, Enzyme.Active, Enzyme.Duplicated($model, dmodel),
            Enzyme.Const(x), Enzyme.Const(ps), Enzyme.Const(st))
    end
    return
end

function benchmark_reverse_pass!(
        suite, tag::String, end_tag::String, ::AutoEnzyme, model, x_dims, ::LuxCUDADevice)
    @warn "Enzyme + Lux is not properly supported on CUDA. Skipping..." maxlog=1
    return
end

function benchmark_reverse_pass!(
        suite, tag::String, end_tag::String, ::AutoTapir, model, x_dims, dev)
    @warn "Tapir support is currently WIP. Skipping..." maxlog=1
    return
end

function benchmark_reverse_pass!(
        suite, tag::String, end_tag::String, ::AutoTracker, model, x_dims, dev)
    dev_tag = dev isa LuxCPUDevice ? "cpu" : "cuda"
    suite[tag][dev_tag]["reverse"]["Tracker"][end_tag] = @async_benchmarkable dev begin
        ps_tracked = fmap(Tracker.param, ps)
        x_tracked = Tracker.param(x)
        loss = sum(abs2, first(Lux.apply($model, x_tracked, ps_tracked, st)))
        Tracker.back!(loss)
    end setup=begin
        (x, ps, st) = general_setup($model, $x_dims, $dev)
    end
    return
end

function benchmark_reverse_pass!(suite, tag::String, end_tag::String,
        ad::AutoReverseDiff, model, x_dims, ::LuxCPUDevice)
    if ad.compile
        suite[tag]["cpu"]["reverse"]["ReverseDiff (compiled)"][end_tag] = @benchmarkable ReverseDiff.gradient!(
            ∂ps, tape, ps_ca) setup=begin
            (x, ps, st) = general_setup($model, $x_dims)
            ps_ca = ComponentArray(ps)
            ∂ps = similar(ps_ca)
            f = @closure(p->sum(abs2, first(Lux.apply($model, x, p, st))))
            tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, ps_ca))
        end
    else
        suite[tag]["cpu"]["reverse"]["ReverseDiff"][end_tag] = @benchmarkable begin
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

function benchmark_reverse_pass!(suite, tag::String, end_tag::String,
        ad::AutoReverseDiff, model, x_dims, ::LuxCUDADevice)
    @warn "ReverseDiff doesn't support CUDA. Skipping..." maxlog=1
    return
end

function benchmark_reverse_pass!(
        suite, tag::String, end_tag::String, ::AutoZygote, model, x_dims, dev)
    dev_tag = dev isa LuxCPUDevice ? "cpu" : "cuda"
    suite[tag][dev_tag]["reverse"]["Zygote"][end_tag] = @async_benchmarkable dev Zygote.gradient(
        f, $model, x, ps, st) setup=begin
        (x, ps, st) = general_setup($model, $x_dims, $dev)
        f = @closure((model, x, p, st)->sum(abs2, first(Lux.apply(model, x, p, st))))
    end
    return
end

function benchmark_reverse_pass_simple_chains!(
        suite, tag::String, end_tag::String, ::AutoZygote, model, x_dims, ::LuxCPUDevice)
    suite[tag]["cpu"]["reverse"]["SimpleChains"][end_tag] = @benchmarkable Zygote.gradient(
        f, $model, x, ps, st) setup=begin
        (x, ps, st) = general_setup($model, $x_dims)
        f = @closure((model, x, p, st)->sum(abs2, first(Lux.apply(model, x, p, st))))
    end
    return
end

function benchmark_reverse_pass_simple_chains!(
        suite, tag::String, end_tag::String, ::AutoZygote, model, x_dims, ::LuxCUDADevice)
    @warn "SimpleChains doesn't support CUDA. Skipping..." maxlog=1
    return
end

function benchmark_reverse_pass_flux!(
        suite, tag::String, end_tag::String, ::AutoZygote, model, x_dims, dev)
    flux_dev = dev isa LuxCPUDevice ? Flux.cpu : Flux.gpu
    dev_tag = dev isa LuxCPUDevice ? "cpu" : "cuda"
    suite[tag][dev_tag]["reverse"]["Flux"][end_tag] = @async_benchmarkable dev Zygote.gradient(
        f, m, x) setup=begin
        x = randn(StableRNG(0), Float32, $x_dims) |> $(flux_dev)
        m = $(model)() |> $(flux_dev)
        f = @closure((m, x)->sum(abs2, m(x)))
    end
    return
end
