using BenchmarkTools, ComponentArrays, Logging, Lux, LuxCUDA, LuxDeviceUtils, StableRNGs
using SimpleChains: static
using Flux: Flux

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
