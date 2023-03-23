using CUDA, Test, Zygote
using LuxLib

include("../test_utils.jl")

function _setup_groupnorm(T, sz, groups; track_stats::Bool)
    x = randn(T, sz)
    scale = randn(T, sz[end - 1])
    bias = randn(T, sz[end - 1])

    if track_stats
        running_mean = randn(T, groups)
        running_var = abs2.(randn(T, groups))
        return x, scale, bias, running_mean, running_var
    else
        return x, scale, bias
    end
end

function _groupnorm_generic_fallback(x, scale, bias, running_mean, running_var, training,
                                     momentum, epsilon, groups)
    sz = size(x)
    N = ndims(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    x_, xmean, xvar = LuxLib._normalization(x_reshaped, running_mean, running_var, scale,
                                            bias, collect(1:(N - 1)), training, momentum,
                                            epsilon)

    return reshape(x_, sz)
end

@testset "GroupNorm KernelAbstractions" begin
    if cpu_testing()
        for T in (Float32, Float64),
            sz in ((16, 16, 6, 4), (32, 32, 6, 4), (64, 64, 12, 4)),
            groups in (2, 3)

            println("GN_CPU: $T $(sz) $groups")

            _f = (args...) -> groupnorm(args...; groups, epsilon)

            epsilon = T(1e-5)
            x, scale, bias = _setup_groupnorm(T, sz, groups; track_stats=false)
            @time y = _f(x, scale, bias)

            @inferred groupnorm(x, scale, bias; groups, epsilon)
            run_JET_tests(_f, x, scale, bias; opt_broken=true)
            @test y isa Array{T, 4}
            @test size(y) == sz

            Zygote.gradient(sum ∘ _f, x, scale, bias)  # Compile
            @time gs_x, gs_scale, gs_bias = Zygote.gradient(sum ∘ _f, x, scale, bias)

            # Use the generic implementation to test the KA implementation
            __f = (args...) -> _groupnorm_generic_fallback(args..., nothing, nothing,
                                                           Val(true), T(0.9), epsilon,
                                                           groups)
            @time y_ = __f(x, scale, bias)

            Zygote.gradient(sum ∘ __f, x, scale, bias)  # Compile
            @time gs_x_, gs_scale_, gs_bias_ = Zygote.gradient(sum ∘ __f, x, scale, bias)

            # The KA implementation reorders operations manually for maximal
            # performance. Hence equality cannot be guaranteed.
            @test isapprox(y, y_; atol=1.0f-3, rtol=1.0f-3)
            @test isapprox(gs_x, gs_x_; atol=1.0f-3, rtol=1.0f-3)
            @test isapprox(gs_scale, gs_scale_; atol=1.0f-3, rtol=1.0f-3)
            @test isapprox(gs_bias, gs_bias_; atol=1.0f-3, rtol=1.0f-3)
        end
    end

    if gpu_testing()
        for T in (Float32, Float64),
            sz in ((16, 16, 6, 4), (32, 32, 6, 4), (64, 64, 12, 4)),
            groups in (2, 3)

            println("GN_GPU: $T $(sz) $groups")

            _f = (args...) -> groupnorm(args...; groups, epsilon)

            epsilon = T(1e-5)
            x, scale, bias = _setup_groupnorm(T, sz, groups; track_stats=false)

            x, scale, bias = (x, scale, bias) .|> cu
            x = x .|> T
            scale = scale .|> T
            bias = bias .|> T

            CUDA.@time y = _f(x, scale, bias)

            @inferred groupnorm(x, scale, bias; groups, epsilon)
            run_JET_tests(_f, x, scale, bias; opt_broken=true)
            @test y isa CuArray{T, 4}
            @test size(y) == sz

            Zygote.gradient(sum ∘ _f, x, scale, bias)  # Compile
            CUDA.@time gs_x, gs_scale, gs_bias = Zygote.gradient(sum ∘ _f, x, scale, bias)

            # Use the generic implementation to test the KA implementation
            __f = (args...) -> _groupnorm_generic_fallback(args..., nothing, nothing,
                                                           Val(true), T(0.9), epsilon,
                                                           groups)

            CUDA.@time y_ = __f(x, scale, bias)

            Zygote.gradient(sum ∘ __f, x, scale, bias)  # Compile
            CUDA.@time gs_x_, gs_scale_, gs_bias_ = Zygote.gradient(sum ∘ __f, x, scale,
                                                                    bias)

            # The KA implementation reorders operations manually for maximal
            # performance. Hence equality cannot be guaranteed.
            @test isapprox(y, y_; atol=1.0f-3, rtol=1.0f-3)
            @test isapprox(gs_x, gs_x_; atol=1.0f-3, rtol=1.0f-3)
            @test isapprox(gs_scale, gs_scale_; atol=1.0f-3, rtol=1.0f-3)
            @test isapprox(gs_bias, gs_bias_; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end

@testset "GroupNorm Generic Fallback" begin
    if cpu_testing()
        for T in (Float16, Float32, Float64),
            sz in ((4, 4, 6, 2), (8, 8, 6, 2), (16, 16, 12, 2)),
            groups in (2, 3),
            training in (Val(true), Val(false))

            println("GN_CPU: $T $(sz) $groups $training")

            _f = (args...) -> groupnorm(args...; groups, epsilon, training, momentum=T(0.9))

            epsilon = T(1e-5)
            x, scale, bias, rm, rv = _setup_groupnorm(T, sz, groups; track_stats=true)
            @time y, nt = _f(x, scale, bias, rm, rv)

            @inferred groupnorm(x, scale, bias, rm, rv; groups, epsilon, training,
                                momentum=T(0.9))
            run_JET_tests(_f, x, scale, bias, rm, rv; opt_broken=true)
            @test y isa Array{T, 4}
            @test size(y) == sz
            @test size(nt.running_mean) == (groups,)
            @test size(nt.running_var) == (groups,)

            Zygote.gradient(sum ∘ first ∘ _f, x, scale, bias, rm, rv)  # Compile
            @time gs_x, gs_scale, gs_bias, _, _ = Zygote.gradient(sum ∘ first ∘ _f, x,
                                                                  scale, bias, rm, rv)

            if T != Float16
                __f = (args...) -> sum(first(groupnorm(args..., rm, rv; groups, epsilon,
                                                       training, momentum=T(0.9))))
                test_gradient_correctness_fdm(__f, x, scale, bias; atol=1.0f-2, rtol=1.0f-2)
            end
        end
    end

    if gpu_testing()
        for T in (Float16, Float32, Float64),
            sz in ((4, 4, 6, 2), (8, 8, 6, 2), (16, 16, 12, 2)),
            groups in (2, 3),
            training in (Val(true), Val(false))

            println("GN_GPU: $T $(sz) $groups $training")

            _f = (args...) -> groupnorm(args...; groups, epsilon, training, momentum=T(0.9))

            epsilon = T(1e-5)
            x, scale, bias, rm, rv = _setup_groupnorm(T, sz, groups; track_stats=true)

            x, scale, bias, rm, rv = (x, scale, bias, rm, rv) .|> cu
            x = x .|> T
            scale = scale .|> T
            bias = bias .|> T
            rm = rm .|> T
            rv = rv .|> T

            CUDA.@time y, nt = _f(x, scale, bias, rm, rv)

            @inferred groupnorm(x, scale, bias, rm, rv; groups, epsilon, training,
                                momentum=T(0.9))
            run_JET_tests(_f, x, scale, bias, rm, rv; opt_broken=true)
            @test y isa CuArray{T, 4}
            @test size(y) == sz
            @test size(nt.running_mean) == (groups,)
            @test size(nt.running_var) == (groups,)

            Zygote.gradient(sum ∘ first ∘ _f, x, scale, bias, rm, rv)  # Compile
            CUDA.@time gs_x, gs_scale, gs_bias, _, _ = Zygote.gradient(sum ∘ first ∘ _f, x,
                                                                       scale, bias, rm, rv)

            __f = (args...) -> sum(first(groupnorm(args..., rm, rv; groups, epsilon,
                                                   training, momentum=T(0.9))))
            # FiniteDifferences for GPU seems broken
            # test_gradient_correctness_fdm(__f, x, scale, bias; atol=1.0f-2, rtol=1.0f-2)
        end
    end
end
