function dense_setup(N::Int, bias::Bool, dev::MLDataDevices.AbstractDevice)
    rng = StableRNG(123)
    x = randn(rng, Float32, N, 128) |> dev
    w = randn(rng, Float32, N, N) |> dev
    b = (bias ? randn(rng, Float32, N) : nothing) |> dev
    return x, w, b
end

function setup_dense_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
                                 backend::String, dev::MLDataDevices.AbstractDevice)
    for bias in [true, false], activation in [identity, relu, gelu], N in [2, 32, 512]
        benchmark_name = "dense($N, bias=$bias, act=$activation)($N x 128)"
        suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
            fused_dense_bias_activation($activation, w, x, b)
            synchronize($dev)
        end setup=begin x, w, b = dense_setup($N, $bias, $dev) end

        suite[benchmark_name]["zygote"][cpu_or_gpu][backend] = @benchmarkable begin
            Zygote.gradient(sumabs2, fused_dense_bias_activation, $activation, w, x, b)
            synchronize($dev)
        end setup=begin
            x, w, b = dense_setup($N, $bias, $dev)
            Zygote.gradient(sumabs2, fused_dense_bias_activation, $activation, w, x, b)
        end
    end
    return
end

# Bias Activation
function bias_activation_setup(N::Int, dev::MLDataDevices.AbstractDevice)
    rng = StableRNG(123)
    x = randn(rng, Float32, N, 128) |> dev
    b = randn(rng, Float32, N) |> dev
    return x, b
end

function setup_bias_activation_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
                                           backend::String,
                                           dev::MLDataDevices.AbstractDevice)
    for activation in [tanh, relu, gelu], N in [2, 32, 512]
        benchmark_name = "bias_activation($N, act=$activation)($N x 128)"
        suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
            bias_activation($activation, x, b)
            synchronize($dev)
        end setup=begin
            reclaim($dev)
            x, b = bias_activation_setup($N, $dev)
        end

        suite[benchmark_name]["zygote"][cpu_or_gpu][backend] = @benchmarkable begin
            Zygote.gradient(sumabs2, bias_activation, $activation, x, b)
            synchronize($dev)
        end setup=begin
            reclaim($dev)
            x, b = bias_activation_setup($N, $dev)
            Zygote.gradient(sumabs2, bias_activation, $activation, x, b)
        end
    end
    return
end

# BatchNorm
function batchnorm_setup(shape::Dims, affine::Bool, dev::MLDataDevices.AbstractDevice)
    rng = StableRNG(123)
    x = randn(rng, Float32, shape...) |> dev
    scale = (affine ? randn(rng, Float32, shape[end - 1]) : nothing) |> dev
    bias = (affine ? randn(rng, Float32, shape[end - 1]) : nothing) |> dev
    running_mean = rand(rng, Float32, shape[end - 1]) |> dev
    running_var = rand(rng, Float32, shape[end - 1]) |> dev
    return x, scale, bias, running_mean, running_var
end

function setup_batchnorm_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
                                     backend::String, dev::MLDataDevices.AbstractDevice)
    for activation in [identity, relu, gelu], ndims in (2, 4)
        shapes = [
            (ntuple(Returns(16), ndims - 2)..., 4, 32),
            (ntuple(Returns(16), ndims - 2)..., 32, 32),
        ]
        for shape in shapes, affine in (true, false)
            benchmark_name = "batchnorm($ndims, act=$activation, affine=$affine)(\
                              $(join(shape, " x ")))"

            suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
                batchnorm(x, scale, bias, running_mean, running_var, Val(false),
                          $activation)
                synchronize($dev)
            end setup=begin x, scale, bias, running_mean, running_var = batchnorm_setup($shape,
                                                                                        $affine,
                                                                                        $dev) end

            suite[benchmark_name]["zygote"][cpu_or_gpu][backend] = @benchmarkable begin
                Zygote.gradient(sumabs2first, batchnorm, x, scale, bias,
                                running_mean, running_var, Val(true), $activation)
                synchronize($dev)
            end setup=begin
                reclaim($dev)
                x, scale, bias, running_mean, running_var = batchnorm_setup($shape, $affine,
                                                                            $dev)
                Zygote.gradient(sumabs2first, batchnorm, x, scale, bias,
                                running_mean, running_var, Val(true), $activation)
            end
        end
    end
    return
end

# LayerNorm
function layernorm_setup(shape::Dims, affine::Bool, dev::MLDataDevices.AbstractDevice)
    rng = StableRNG(123)
    x = randn(rng, Float32, shape...) |> dev
    scale = (affine ? randn(rng, Float32, shape[1:(end - 1)]..., 1) : nothing) |> dev
    bias = (affine ? randn(rng, Float32, shape[1:(end - 1)]..., 1) : nothing) |> dev
    return x, scale, bias
end

function setup_layernorm_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
                                     backend::String, dev::MLDataDevices.AbstractDevice)
    for activation in [identity, relu, gelu], ndims in (2, 4)
        shapes = [
            (ntuple(Returns(16), ndims - 2)..., 4, 32),
            (ntuple(Returns(16), ndims - 2)..., 32, 32),
        ]
        for shape in shapes, affine in (true, false)
            benchmark_name = "layernorm($ndims, act=$activation, affine=$affine)(\
                              $(join(shape, " x ")))"

            suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
                layernorm(x, scale, bias, $activation, 1:($ndims - 1))
                synchronize($dev)
            end setup=begin
                reclaim($dev)
                x, scale, bias = layernorm_setup($shape, $affine, $dev)
            end

            suite[benchmark_name]["zygote"][cpu_or_gpu][backend] = @benchmarkable begin
                Zygote.gradient(sumabs2, layernorm, x, scale, bias, $activation,
                                1:($ndims - 1))
                synchronize($dev)
            end setup=begin
                reclaim($dev)
                x, scale, bias = layernorm_setup($shape, $affine, $dev)
                Zygote.gradient(sumabs2, layernorm, x, scale, bias, $activation,
                                1:($ndims - 1))
            end
        end
    end
    return
end

# GroupNorm
function groupnorm_setup(shape::Dims, affine::Bool, dev::MLDataDevices.AbstractDevice)
    rng = StableRNG(123)
    x = randn(rng, Float32, shape...) |> dev
    scale = (affine ? randn(rng, Float32, shape[end - 1]) : nothing) |> dev
    bias = (affine ? randn(rng, Float32, shape[end - 1]) : nothing) |> dev
    return x, scale, bias
end

function setup_groupnorm_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
                                     backend::String, dev::MLDataDevices.AbstractDevice)
    for activation in [identity, relu, gelu], ndims in (2, 4)
        shapes = [
            (ntuple(Returns(16), ndims - 2)..., 4, 32),
            (ntuple(Returns(16), ndims - 2)..., 32, 32),
        ]
        for shape in shapes, affine in (true, false)
            benchmark_name = "groupnorm($ndims, act=$activation, affine=$affine)(\
                              $(join(shape, " x ")))"

            suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
                groupnorm(x, scale, bias, 4, $activation)
                synchronize($dev)
            end setup=begin
                reclaim($dev)
                x, scale, bias = groupnorm_setup($shape, $affine, $dev)
            end

            suite[benchmark_name]["zygote"][cpu_or_gpu][backend] = @benchmarkable begin
                Zygote.gradient(sumabs2, groupnorm, x, scale, bias, 4, $activation)
                synchronize($dev)
            end setup=begin
                reclaim($dev)
                x, scale, bias = groupnorm_setup($shape, $affine, $dev)
                Zygote.gradient(sumabs2, groupnorm, x, scale, bias, 4, $activation)
            end
        end
    end
    return
end

# Batched Matrix Multiplication
function batchedmm_setup(N::Int, Bsize::Int, dev::MLDataDevices.AbstractDevice)
    rng = StableRNG(123)
    x = randn(rng, Float32, N, N, Bsize) |> dev
    return x
end

function setup_batched_matmul_benchmarks!(suite::BenchmarkGroup, cpu_or_gpu::String,
                                          backend::String,
                                          dev::MLDataDevices.AbstractDevice)
    for N in [2, 16, 128, 512], Bsize in [4, 32, 128, 512]
        benchmark_name = "batchedmm($N, Bsize=$Bsize)"

        suite[benchmark_name]["forward"][cpu_or_gpu][backend] = @benchmarkable begin
            batched_matmul(x, x)
            synchronize($dev)
        end setup=begin
            reclaim($dev)
            x = batchedmm_setup($N, $Bsize, $dev)
        end

        suite[benchmark_name]["zygote"][cpu_or_gpu][backend] = @benchmarkable begin
            Zygote.gradient(sumabs2, batched_matmul, x, x)
            synchronize($dev)
        end setup=begin
            reclaim($dev)
            x = batchedmm_setup($N, $Bsize, $dev)
            Zygote.gradient(sumabs2, batched_matmul, x, x)
        end
    end
    return
end
