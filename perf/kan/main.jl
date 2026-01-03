# Taken from https://github.com/vpuri3/KolmogorovArnold.jl/blob/0fc349813be15982365173bce0e9bf3a814a342a/examples/eg3.jl
using KolmogorovArnold
using Comonicon, BenchmarkTools, JSON3
using Random, LinearAlgebra
using Enzyme, Zygote, Lux
using OrderedCollections

# configure BLAS
ncores = min(Sys.CPU_THREADS, length(Sys.cpu_info()))
BLAS.set_num_threads(ncores)

# configure CUDA
using LuxCUDA
CUDA.allowscalar(false)

# configure Reactant
using Reactant
Reactant.set_default_backend("gpu")

rng = Random.default_rng()
Random.seed!(rng, 0)

function toy_loss_function(model, ps, st, x, y)
    pred, _ = model(x, ps, st)
    return MSELoss()(pred, y)
end

function setup_models(; kan_width::Int=128, grid_size::Int=32)
    wK, G = kan_width, grid_size

    basis_func = rbf      # rbf, rswaf
    normalizer = softsign # sigmoid(_fast), tanh(_fast), softsign

    kan1 = Chain(
        KDense(1, wK, G; use_base_act=true, basis_func, normalizer),
        KDense(wK, wK, G; use_base_act=true, basis_func, normalizer),
        KDense(wK, 1, G; use_base_act=true, basis_func, normalizer),
    )

    kan2 = Chain(
        KDense(1, wK, G; use_base_act=false, basis_func, normalizer),
        KDense(wK, wK, G; use_base_act=false, basis_func, normalizer),
        KDense(wK, 1, G; use_base_act=false, basis_func, normalizer),
    )

    return [("kan_base_act", kan1), ("kan_no_base_act", kan2)]
end

function run_cuda_benchmarks(; batch_size::Int=128, kwargs...)
    dev = gpu_device(; force=true)

    x = rand32(rng, 1, batch_size)
    y = x .^ 2

    models = setup_models(; kwargs...)
    timings = OrderedDict{String,OrderedDict{String,Float64}}()

    for (name, model) in models
        println("\nCUDA Benchmarking: $name")

        ps, st = Lux.setup(rng, model) |> dev
        x_cu = x |> dev
        y_cu = y |> dev

        println("Param count: $(Lux.parameterlength(ps))")
        println("State count: $(Lux.statelength(st))")

        # Forward pass timing
        fwd_time = @belapsed begin
            pred, _ = $(model)($(x_cu), $(ps), $(Lux.testmode(st)))
            CUDA.synchronize()
        end setup = begin
            GC.gc(true)
            CUDA.reclaim()
        end

        # Backward pass timing (using Zygote)
        fn = (ps, x) -> toy_loss_function(model, ps, st, x, y_cu)

        bwd_time = @belapsed begin
            Zygote.gradient($(fn), $(ps), $(x_cu))
            CUDA.synchronize()
        end setup = begin
            GC.gc(true)
            CUDA.reclaim()
        end

        timings[name] = OrderedDict{String,Float64}(
            "forward" => fwd_time, "backward" => bwd_time
        )

        display(timings[name])
    end

    return timings
end

function run_xla_benchmarks(; kwargs...)
    return run_reactant_benchmarks(;
        kwargs..., compile_options=Reactant.DefaultXLACompileOptions()
    )
end

function run_reactant_benchmarks(;
    batch_size::Int=128,
    compile_options=Reactant.CompileOptions(; optimization_passes=:all),
    kwargs...,
)
    dev = reactant_device(; force=true)

    x = rand32(rng, 1, batch_size)
    y = x .^ 2

    models = setup_models(; kwargs...)
    timings = OrderedDict{String,OrderedDict{String,Float64}}()

    for (name, model) in models
        println("\nReactant Benchmarking: $name")

        ps, st = Lux.setup(rng, model) |> dev
        x_ra = x |> dev
        y_ra = y |> dev

        println("Param count: $(Lux.parameterlength(ps))")
        println("State count: $(Lux.statelength(st))")

        # Forward pass timing
        fwd_time_result = Reactant.Profiler.profile_with_xprof(
            Lux.apply,
            model,
            x_ra,
            ps,
            Lux.testmode(st);
            nrepeat=10,
            warmup=1,
            compile_options,
        )
        fwd_time = fwd_time_result.profiling_result.runtime_ns / 1e9

        # Backward pass timing
        bwd_time_result = Reactant.Profiler.profile_with_xprof(
            Enzyme.gradient,
            Reverse,
            toy_loss_function,
            Const(model),
            ps,
            Const(st),
            Const(x_ra),
            Const(y_ra);
            nrepeat=10,
            warmup=1,
            compile_options,
        )
        bwd_time = bwd_time_result.profiling_result.runtime_ns / 1e9

        timings[name] = OrderedDict{String,Float64}(
            "forward" => fwd_time, "backward" => bwd_time
        )

        display(timings[name])
    end

    return timings
end

Comonicon.@main function main(;
    backend::String="all", batch_size::Int=1024, kan_width::Int=128, grid_size::Int=32
)
    results_path = joinpath(@__DIR__, "../results/kan/")
    mkpath(results_path)

    if backend in ("cuda", "all")
        println("\n" * "="^50)
        println("Running CUDA benchmarks...")
        println("="^50)

        cuda_timings = run_cuda_benchmarks(; batch_size, kan_width, grid_size)

        open(joinpath(results_path, "cudajl.json"), "w") do io
            JSON3.write(io, cuda_timings)
        end

        println("\nCUDA Results:")
        display(cuda_timings)
    end

    if backend in ("reactant", "all")
        println("\n" * "="^50)
        println("Running Reactant benchmarks...")
        println("="^50)

        reactant_timings = run_reactant_benchmarks(; batch_size, kan_width, grid_size)

        open(joinpath(results_path, "reactant.json"), "w") do io
            JSON3.write(io, reactant_timings)
        end

        println("\nReactant Results:")
        display(reactant_timings)
    end

    if backend in ("xla", "all")
        println("\n" * "="^50)
        println("Running XLA benchmarks...")
        println("="^50)

        xla_timings = run_xla_benchmarks(; batch_size, kan_width, grid_size)

        open(joinpath(results_path, "xla.json"), "w") do io
            JSON3.write(io, xla_timings)
        end

        println("\nXLA Results:")
        display(xla_timings)
    end

    return nothing
end
