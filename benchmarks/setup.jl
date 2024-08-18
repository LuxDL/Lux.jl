using ADTypes: ADTypes, AutoEnzyme, AutoZygote
using Adapt: adapt
using Lux: Lux, BatchNorm, Chain, Conv, CrossCor, Dense, Dropout, FlattenLayer, MaxPool
using LuxDeviceUtils: LuxCPUDevice, LuxCUDADevice, LuxAMDGPUDevice
using NNlib: relu, gelu
using Random: Random

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

function general_setup(model, x_dims)
    rng = Random.default_rng()  # don't use any other rng
    ps, st = Lux.setup(rng, model)
    x_dims === nothing && return ps, st
    x = randn(rng, Float32, x_dims)
    return x, ps, st
end

function benchmark_forward_pass!(suite::BenchmarkGroup, group::String, tag, model,
        x_dims)
    dev = group_to_backend(group)

    suite[tag]["Forward"]["Lux"] = @benchmarkable begin
        Lux.apply($model, x, ps, st_test)
        synchronize($dev)
    end setup=begin
        (x, ps, st) = general_setup($model, $x_dims) |> $dev
        st_test = Lux.testmode(st)
        Lux.apply($model, x, ps, st_test) # Warm up
    end
end

function benchmark_reverse_pass!(suite::BenchmarkGroup, group::String, backends, tag,
        model, x_dims)
    for backend in backends
        benchmark_reverse_pass!(suite, group, backend, tag, model, x_dims)
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

# loadparams custom
loadparams!(args...) = BenchmarkTools.loadparams!(args...), true

function loadparams!(group::BenchmarkGroup, paramsgroup::BenchmarkGroup, fields...)
    has_all_groups = true
    for (k, v) in group
        if haskey(paramsgroup, k)
            _, _has_all_groups = loadparams!(v, paramsgroup[k], fields...)
            !_has_all_groups && (has_all_groups = false)
        else
            has_all_groups = false
        end
    end
    return group, has_all_groups
end

# Final Setup. Main entry point for benchmarks
function setup_benchmarks(suite::BenchmarkGroup, backend::String, num_cpu_threads::Int64)
    # Common Layers
    add_dense_benchmarks!(suite, backend)
    add_conv_benchmarks!(suite, backend)

    # Full Models
    add_vgg16_benchmarks!(suite, backend)
end
