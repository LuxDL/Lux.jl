using ADTypes: ADTypes, AutoEnzyme, AutoZygote
using Adapt: adapt
using Flux: Flux
using Lux: Lux, BatchNorm, Chain, Conv, Dense, Dropout, FlattenLayer, MaxPool
using LuxDeviceUtils: LuxCPUDevice, LuxCUDADevice, LuxAMDGPUDevice
using NNlib: relu
using SimpleChains: SimpleChains, static
using StableRNGs: StableRNG

# AD Backends
using Enzyme: Enzyme
using Zygote: Zygote

# Helper Functions
@inline synchronize(::LuxCUDADevice) = CUDA.synchronize()
@inline synchronize(::LuxAMDGPUDevice) = AMDGPU.synchronize()
@inline synchronize(::LuxCPUDevice) = nothing

@inline sumabs2(model, x, p, st) = sum(abs2, first(Lux.apply(model, x, p, st)))
@inline sumabs2(model, x) = sum(abs2, model(x))

function group_to_backend(group::String)
    group == "AMDGPU" && return LuxAMDGPUDevice()
    group == "CUDA" && return LuxCUDADevice()
    group == "CPU" && return LuxCPUDevice()
    error("Unknown backend: $group")
end

function group_to_flux_backend(group::String)
    group == "AMDGPU" && return Base.Fix1(adapt, Flux.FluxAMDGPUAdaptor())
    group == "CUDA" && return Base.Fix1(adapt, Flux.FluxCUDAAdaptor())
    group == "CPU" && return Base.Fix1(adapt, Flux.FluxCPUAdaptor())
    error("Unknown backend: $group")
end

function general_setup(model, x_dims)
    rng = StableRNG(0)
    ps, st = Lux.setup(rng, model)
    x_dims === nothing && return ps, st
    x = randn(rng, Float32, x_dims)
    return x, ps, st
end

function benchmark_forward_pass!(suite::BenchmarkGroup, group::String, tag, model,
        x_dims; simple_chains=nothing, flux_model=nothing)
    dev = group_to_backend(group)

    suite[tag]["Forward"]["Lux"] = @benchmarkable begin
        Lux.apply($model, x, ps, st_test)
        synchronize($dev)
    end setup=begin
        (x, ps, st) = general_setup($model, $x_dims) |> $dev
        st_test = Lux.testmode(st)
        Lux.apply($model, x, ps, st_test) # Warm up
    end

    if simple_chains !== nothing && group == "CPU"
        simple_chains_model = simple_chains(model)
        suite[tag]["Forward"]["SimpleChains"] = @benchmarkable begin
            Lux.apply($simple_chains_model, x, ps_simple_chains, st_simple_chains)
        end setup=begin
            (x, ps_simple_chains, st_simple_chains) = general_setup(
                $simple_chains_model, $x_dims)
            Lux.apply($simple_chains_model, x, ps_simple_chains, st_simple_chains) # Warm up
        end
    end

    if flux_model !== nothing
        suite[tag]["Forward"]["Flux"] = @benchmarkable begin
            fmodel(x)
            synchronize($dev)
        end setup=begin
            fdev = group_to_flux_backend($group)
            x = randn(StableRNG(0), Float32, $x_dims) |> fdev
            fmodel = $(flux_model()) |> fdev
            fmodel(x) # Warm up
        end
    end
end

function benchmark_reverse_pass!(suite::BenchmarkGroup, group::String, backends, tag,
        model, x_dims; simple_chains=nothing, flux_model=nothing)
    for backend in backends
        benchmark_reverse_pass!(suite, group, backend, tag, model, x_dims)
    end

    if simple_chains !== nothing && group == "CPU"
        simple_chains_model = simple_chains(model)
        benchmark_reverse_pass_simple_chains!(suite, tag, simple_chains_model, x_dims)
    end

    if flux_model !== nothing
        benchmark_reverse_pass_flux!(suite, group, tag, flux_model, x_dims)
    end
end

function benchmark_reverse_pass!(
        suite::BenchmarkGroup, group::String, ::AutoZygote, tag::String, model, x_dims)
    dev = group_to_backend(group)

    suite[tag]["Reverse"]["Zygote"] = @benchmarkable begin
        Zygote.gradient(sumabs2, $model, x, ps, st)
        synchronize($dev)
    end setup=begin
        (x, ps, st) = general_setup($model, $x_dims) |> $dev
        Zygote.gradient(sumabs2, $model, x, ps, st) # Warm up
    end
end

function benchmark_reverse_pass!(
        suite::BenchmarkGroup, group::String, ::AutoEnzyme, tag::String, model, x_dims)
    if group != "CPU"
        @error "Enzyme.jl currently only supports CPU"
        return
    end

    dev = group_to_backend(group)

    suite[tag]["Reverse"]["Enzyme"] = @benchmarkable begin
        Enzyme.autodiff(Enzyme.Reverse, sumabs2, Enzyme.Active, Enzyme.Const($model),
            Enzyme.Duplicated(x, dx), Enzyme.Duplicated(ps, dps), Enzyme.Const(st))
        synchronize($dev)
    end setup=begin
        (x, ps, st) = general_setup($model, $x_dims) |> $dev
        dps = Enzyme.make_zero(ps)
        dx = Enzyme.make_zero(x)
        Enzyme.autodiff(Enzyme.Reverse, sumabs2, Enzyme.Active, Enzyme.Const($model),
            Enzyme.Duplicated(x, dx), Enzyme.Duplicated(ps, dps), Enzyme.Const(st)) # Warm up
    end
end

function benchmark_reverse_pass_simple_chains!(
        suite::BenchmarkGroup, tag::String, model, x_dims)
    suite[tag]["Reverse"]["SimpleChains"] = @benchmarkable begin
        Zygote.gradient(sumabs2, $model, x, ps_simple_chains, st_simple_chains)
    end setup=begin
        (x, ps_simple_chains, st_simple_chains) = general_setup($model, $x_dims)
        Zygote.gradient(sumabs2, $model, x, ps_simple_chains, st_simple_chains) # Warm up
    end
end

function benchmark_reverse_pass_flux!(
        suite::BenchmarkGroup, group::String, tag::String, model, x_dims)
    dev = group_to_backend(group)

    suite[tag]["Reverse"]["Flux"] = @benchmarkable begin
        Zygote.gradient(sumabs2, fmodel, x)
        synchronize($dev)
    end setup=begin
        fdev = group_to_flux_backend($group)
        x = randn(StableRNG(0), Float32, $x_dims) |> fdev
        fmodel = $(model)() |> fdev
        Zygote.gradient(sumabs2, fmodel, x) # Warm up
    end
end

# Final Setup. Main entry point for benchmarks
function setup_benchmarks(suite::BenchmarkGroup, backend::String, num_cpu_threads::Int64)
    # Common Layers
    add_dense_benchmarks!(suite, backend)

    # Normalization Layers

    # Full Models
end
