using MLDataDevices, StableRNGs, Random

synchronize(::CPUDevice) = nothing
synchronize(::AMDGPUDevice) = AMDGPU.synchronize()
synchronize(::CUDADevice) = CUDA.synchronize()
synchronize(::MetalDevice) = Metal.synchronize()
synchronize(::oneAPIDevice) = oneAPI.synchronize()

function benchmark_group_to_backend(benchmark_group::String)
    benchmark_group == "CPU" && return CPUDevice()
    benchmark_group == "AMDGPU" && return AMDGPUDevice()
    benchmark_group == "CUDA" && return CUDADevice()
    benchmark_group == "Metal" && return MetalDevice()
    benchmark_group == "oneAPI" && return oneAPIDevice()
    error("Unknown backend: $(benchmark_group)")
end

function setup_benchmarks!(suite::BenchmarkGroup, backend::String, num_cpu_threads::Int64)
    dev = benchmark_group_to_backend(backend)
    cpu_or_gpu = backend == "CPU" ? "CPU" : "GPU"
    final_backend = backend == "CPU" ? string(num_cpu_threads, " ", "thread(s)") : backend

    setup_dense_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_bias_activation_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_batchnorm_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_layernorm_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_groupnorm_benchmarks!(suite, cpu_or_gpu, final_backend, dev)

    setup_batched_matmul_benchmarks!(suite, cpu_or_gpu, final_backend, dev)
end

# Dense
function setup_dense_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
        backend::String, dev::MLDataDevices.AbstractDevice)
    for bias in [true, false], activation in [identity, relu, gelu], N in [2, 32, 512]
        benchmark_name = "dense($N, bias=$bias, act=$activation)($N x 128)"
        suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
            fused_dense_bias_activation($activation, w, x, b)
            synchronize($dev)
        end setup=begin
            rng = StableRNG(123)
            x = randn(rng, Float32, $N, 128) |> $(dev)
            w = randn(rng, Float32, $N, $N) |> $(dev)
            b = ($bias ? randn(rng, Float32, $N) : nothing) |> $(dev)
        end
    end
end

# Bias Activation
function setup_bias_activation_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
        backend::String, dev::MLDataDevices.AbstractDevice)
    for activation in [tanh, relu, gelu], N in [2, 32, 512]
        benchmark_name = "bias_activation($N, act=$activation)($N x 128)"
        suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
            bias_activation($activation, x, b)
            synchronize($dev)
        end setup=begin
            rng = StableRNG(123)
            x = randn(rng, Float32, $N, 128) |> $(dev)
            b = randn(rng, Float32, $N) |> $(dev)
        end
    end
end

# BatchNorm
function setup_batchnorm_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
        backend::String, dev::MLDataDevices.AbstractDevice)
    for activation in [identity, relu, gelu], ndims in (2, 4)
        shapes = [(ntuple(Returns(16), ndims - 2)..., 4, 32),
            (ntuple(Returns(16), ndims - 2)..., 32, 32)]
        for shape in shapes, affine in (true, false)
            benchmark_name = "batchnorm($ndims, act=$activation, affine=$affine)(\
                              $(join(shape, " x ")))"

            suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
                batchnorm(
                    x, scale, bias, running_mean, running_var, Val(false), $activation)
                synchronize($dev)
            end setup=begin
                rng = StableRNG(123)
                x = randn(rng, Float32, $(shape...)) |> $(dev)
                scale = $affine ? randn(rng, Float32, $(shape[end - 1])) |> $(dev) : nothing
                bias = $affine ? randn(rng, Float32, $(shape[end - 1])) |> $(dev) : nothing
                running_mean = rand(rng, Float32, $(shape[end - 1])) |> $(dev)
                running_var = rand(rng, Float32, $(shape[end - 1])) |> $(dev)
            end
        end
    end
end

# LayerNorm
function setup_layernorm_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
        backend::String, dev::MLDataDevices.AbstractDevice)
    for activation in [identity, relu, gelu], ndims in (2, 4)
        shapes = [(ntuple(Returns(16), ndims - 2)..., 4, 32),
            (ntuple(Returns(16), ndims - 2)..., 32, 32)]
        for shape in shapes, affine in (true, false)
            benchmark_name = "layernorm($ndims, act=$activation, affine=$affine)(\
                              $(join(shape, " x ")))"

            suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
                layernorm(x, scale, bias, $activation, 1:($ndims - 1))
                synchronize($dev)
            end setup=begin
                rng = StableRNG(123)
                x = randn(rng, Float32, $(shape...)) |> $(dev)
                scale = $affine ?
                        randn(rng, Float32, $(shape[1:(end - 1)]..., 1)) |> $(dev) : nothing
                bias = $affine ?
                       randn(rng, Float32, $(shape[1:(end - 1)]..., 1)) |> $(dev) : nothing
            end
        end
    end
end

# GroupNorm
function setup_groupnorm_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
        backend::String, dev::MLDataDevices.AbstractDevice)
    for activation in [identity, relu, gelu], ndims in (2, 4)
        shapes = [(ntuple(Returns(16), ndims - 2)..., 4, 32),
            (ntuple(Returns(16), ndims - 2)..., 32, 32)]
        for shape in shapes, affine in (true, false)
            benchmark_name = "groupnorm($ndims, act=$activation, affine=$affine)(\
                              $(join(shape, " x ")))"

            suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
                groupnorm(x, scale, bias, 4, $activation)
                synchronize($dev)
            end setup=begin
                rng = StableRNG(123)
                x = randn(rng, Float32, $(shape...)) |> $(dev)
                scale = $affine ? randn(rng, Float32, $(shape[end - 1])) |> $(dev) : nothing
                bias = $affine ? randn(rng, Float32, $(shape[end - 1])) |> $(dev) : nothing
            end
        end
    end
end

# Batched Matrix Multiplication
function setup_batched_matmul_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
        backend::String, dev::MLDataDevices.AbstractDevice)
    if dev isa MetalDevice || dev isa oneAPIDevice
        @warn "Skipping batched_matmul benchmarks for $(dev)..."
        return
    end

    for N in [2, 16, 128, 512], Bsize in [4, 32, 128, 512]
        benchmark_name = "batchedmm($N, Bsize=$Bsize)"

        suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
            batched_matmul(x, x)
            synchronize($dev)
        end setup=begin
            rng = StableRNG(123)
            x = randn(rng, Float32, $N, $N, $Bsize) |> $(dev)
        end
    end
end
