using ADTypes
using Adapt: adapt
using Lux
using LuxLib
using MLDataDevices
using MLDataDevices: AbstractDevice
using NNlib
using Random: Random
using StableRNGs: StableRNG

# AD Backends
using Enzyme: Enzyme
using Zygote: Zygote

# Helper Functions
synchronize(::CPUDevice) = nothing
synchronize(::AMDGPUDevice) = AMDGPU.synchronize()
synchronize(::CUDADevice) = CUDA.synchronize()
synchronize(::MetalDevice) = Metal.synchronize()
synchronize(::oneAPIDevice) = oneAPI.synchronize()

reclaim(::CPUDevice) = GC.gc()
reclaim(::AMDGPUDevice) = AMDGPU.HIP.reclaim()
reclaim(::CUDADevice) = CUDA.reclaim()
reclaim(::MetalDevice) = nothing  # Metal.reclaim()
reclaim(::oneAPIDevice) = nothing # oneAPI.reclaim()

function sumabs2(model::Lux.AbstractLuxLayer, x, p, st)
    return sum(abs2, first(Lux.apply(model, x, p, st)))
end
sumabs2(f::F, args...) where {F} = sum(abs2, f(args...))
sumabs2first(f::F, args...) where {F} = sum(abs2, first(f(args...)))

function benchmark_group_to_backend(benchmark_group::String)
    benchmark_group == "CPU" && return CPUDevice()
    benchmark_group == "AMDGPU" && return AMDGPUDevice()
    benchmark_group == "CUDA" && return CUDADevice()
    benchmark_group == "Metal" && return MetalDevice()
    benchmark_group == "oneAPI" && return oneAPIDevice()
    error("Unknown backend: $(benchmark_group)")
end

function general_setup(model, x_dims)
    rng = Random.default_rng()  # don't use any other rng
    ps, st = Lux.setup(rng, model)
    x_dims === nothing && return ps, st
    x = randn(rng, Float32, x_dims)
    return x, ps, st
end

# Main benchmark files
include("setups/layers.jl")
include("setups/models.jl")
include("setups/luxlib.jl")

function setup_benchmarks!(suite::BenchmarkGroup, backend::String, num_cpu_threads::Int64)
    dev = benchmark_group_to_backend(backend)
    cpu_or_gpu = backend == "CPU" ? "CPU" : "GPU"
    final_backend = backend == "CPU" ? string(num_cpu_threads, " ", "thread(s)") : backend

    # Model Benchmarks
    setup_dense_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_conv_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_vgg16_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_mlp_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_lenet_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    # Layer Benchmarks
    setup_dense_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_bias_activation_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_batchnorm_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_layernorm_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_groupnorm_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    return setup_batched_matmul_benchmarks!(suite, cpu_or_gpu, final_backend, dev)
end

function setup_forward_pass_benchmark!(
        suite::BenchmarkGroup, benchmark_name::String,
        cpu_or_gpu::String, backend::String, model, x_dims, dev::AbstractDevice
)
    return suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
        Lux.apply($model, x, ps, st_test)
        synchronize($dev)
    end setup=begin
        reclaim($dev)
        x, ps, st = general_setup($model, $x_dims) |> $dev
        st_test = Lux.testmode(st)
    end
end

function setup_reverse_pass_benchmark!(
        suite::BenchmarkGroup, benchmark_name::String,
        cpu_or_gpu::String, backend::String, ad_backends, model, x_dims, dev::AbstractDevice
)
    for ad_backend in ad_backends
        setup_reverse_pass_benchmark!(
            suite, benchmark_name, cpu_or_gpu, backend, ad_backend, model, x_dims, dev
        )
    end
    return
end

function setup_reverse_pass_benchmark!(
        suite::BenchmarkGroup, benchmark_name::String,
        cpu_or_gpu::String, backend::String, ::AutoZygote, model, x_dims, dev::AbstractDevice
)
    return suite[benchmark_name]["zygote"][cpu_or_gpu][backend] = @benchmarkable begin
        Zygote.gradient(sumabs2, $model, x, ps, st)
        synchronize($dev)
    end setup=begin
        reclaim($dev)
        x, ps, st = general_setup($model, $x_dims) |> $dev
        Zygote.gradient(sumabs2, $model, x, ps, st) # Warm up
    end
end

function setup_reverse_pass_benchmark!(
        suite::BenchmarkGroup, benchmark_name::String,
        cpu_or_gpu::String, backend::String, ::AutoEnzyme, model, x_dims, dev::AbstractDevice
)
    cpu_or_gpu != "CPU" && return  # TODO: Remove once Enzyme.jl supports GPUs

    return suite[benchmark_name]["enzyme"][cpu_or_gpu][backend] = @benchmarkable begin
        Enzyme.autodiff(
            Enzyme.Reverse, sumabs2, Enzyme.Active, Enzyme.Const($model),
            Enzyme.Duplicated(x, dx), Enzyme.Duplicated(ps, dps), Enzyme.Const(st)
        )
        synchronize($dev)
    end setup=begin
        reclaim($dev)
        x, ps, st = general_setup($model, $x_dims) |> $dev
        dps = Enzyme.make_zero(ps)
        dx = Enzyme.make_zero(x)
        Enzyme.autodiff(
            Enzyme.Reverse, sumabs2, Enzyme.Active, Enzyme.Const($model),
            Enzyme.Duplicated(x, dx), Enzyme.Duplicated(ps, dps), Enzyme.Const(st)
        ) # Warm up
    end
end
