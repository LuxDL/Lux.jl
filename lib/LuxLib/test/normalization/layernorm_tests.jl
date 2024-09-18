@testsetup module LayerNormSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, NNlib, Statistics
using LuxTestUtils: check_approx

function setup_layernorm(gen_f, aType, T, x_size, affine_shape, expand_dims::Bool=true)
    x = gen_f(T, x_size) |> aType
    if affine_shape !== nothing
        if expand_dims
            scale = gen_f(T, (affine_shape..., 1)) |> aType
            bias = gen_f(T, (affine_shape..., 1)) |> aType
        else
            scale = gen_f(T, affine_shape) |> aType
            bias = gen_f(T, affine_shape) |> aType
        end
        return x, scale, bias
    else
        return x, nothing, nothing
    end
end

function run_layernorm_testing(gen_f, aType, T, x_size, affine_shape, act, ongpu, mode)
    @testset for dims in (Colon(), nothing)
        if dims === nothing
            affine_shape === nothing && continue
            length(x_size) ≤ length(affine_shape) && continue
            x, scale, bias = setup_layernorm(gen_f, aType, T, x_size, affine_shape, false)
        else
            x, scale, bias = setup_layernorm(gen_f, aType, T, x_size, affine_shape)
        end

        run_layernorm_testing_core(
            aType, T, x_size, affine_shape, act, dims, x, scale, bias)
    end
end

function run_layernorm_testing_core(
        aType, T, x_size, affine_shape, act, dims, x, scale, bias)
    epsilon = LuxLib.Utils.default_epsilon(T)
    _f = (args...) -> layernorm(args..., act, dims, epsilon)

    @test @inferred(layernorm(x, scale, bias, act, dims, epsilon)) isa Any
    @jet layernorm(x, scale, bias, act, dims, epsilon)

    y = _f(x, scale, bias)

    @test y isa aType{T, length(x_size)}
    @test size(y) == x_size

    if affine_shape === nothing && act === identity
        @test check_approx(mean(y; dims), 0; atol=1e-3, rtol=1e-3)
        @test check_approx(std(y; dims), 1; atol=1e-1, rtol=1e-1)
    end

    fp16 = T == Float16
    atol = fp16 ? 1.0f-2 : 1.0f-3
    rtol = fp16 ? 1.0f-2 : 1.0f-3

    soft_fail = fp16 ? fp16 : [AutoFiniteDiff()]
    if affine_shape !== nothing
        __f = (args...) -> sum(_f(args...))
        @test_gradients(__f, x, scale, bias; atol, rtol, soft_fail)
    else
        __f = x -> sum(_f(x, scale, bias))
        @test_gradients(__f, x; atol, rtol, soft_fail)
    end

    if anonact !== act
        lfn = (x, sc, b, act, dim, ϵ) -> sum(layernorm(x, sc, b, act, dim, ϵ))
        @test @inferred(Zygote.gradient(lfn, x, scale, bias, act, dims, epsilon)) isa Any
    end
end

anonact = x -> x^3

const ALL_TEST_CONFIGS = Any[]

for T in (Float16, Float32, Float64),
    x_shape in ((3, 3, 2, 1), (2, 2, 2, 1), (2, 3, 2, 2)),
    affine_shape in (nothing, x_shape[1:3], (1, 1, 1), (1, 1, x_shape[3])),
    act in (identity, relu, tanh_fast, sigmoid_fast, anonact)

    push!(ALL_TEST_CONFIGS, (T, x_shape, affine_shape, act))
end

const TEST_BLOCKS = collect(Iterators.partition(
    ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 5)))

export ALL_TEST_CONFIGS, TEST_BLOCKS, run_layernorm_testing

end

@testitem "Layer Norm: Group 1" tags=[:layer_norm] setup=[SharedTestSetup, LayerNormSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $T, size $x_shape, $act" for (T, x_shape, affine_shape, act) in TEST_BLOCKS[1]
            run_layernorm_testing(
                generate_fixed_array, aType, T, x_shape, affine_shape, act, ongpu, mode)
        end
    end
end

@testitem "Layer Norm: Group 2" tags=[:layer_norm] setup=[SharedTestSetup, LayerNormSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $T, size $x_shape, $act" for (T, x_shape, affine_shape, act) in TEST_BLOCKS[2]
            run_layernorm_testing(
                generate_fixed_array, aType, T, x_shape, affine_shape, act, ongpu, mode)
        end
    end
end

@testitem "Layer Norm: Group 3" tags=[:layer_norm] setup=[SharedTestSetup, LayerNormSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $T, size $x_shape, $act" for (T, x_shape, affine_shape, act) in TEST_BLOCKS[3]
            run_layernorm_testing(
                generate_fixed_array, aType, T, x_shape, affine_shape, act, ongpu, mode)
        end
    end
end

@testitem "Layer Norm: Group 4" tags=[:layer_norm] setup=[SharedTestSetup, LayerNormSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $T, size $x_shape, $act" for (T, x_shape, affine_shape, act) in TEST_BLOCKS[4]
            run_layernorm_testing(
                generate_fixed_array, aType, T, x_shape, affine_shape, act, ongpu, mode)
        end
    end
end

@testitem "Layer Norm: Group 5" tags=[:layer_norm] setup=[SharedTestSetup, LayerNormSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $T, size $x_shape, $act" for (T, x_shape, affine_shape, act) in TEST_BLOCKS[5]
            run_layernorm_testing(
                generate_fixed_array, aType, T, x_shape, affine_shape, act, ongpu, mode)
        end
    end
end

@testitem "Layer Norm: Error Checks" tags=[:layer_norm] setup=[SharedTestSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        x = rand(2, 3) |> aType

        @test_throws ArgumentError layernorm(x, nothing, nothing, identity, nothing, 1e-5)

        sc = rand(2, 1) |> aType
        b = rand(2, 1) |> aType

        @test_throws AssertionError layernorm(x, sc, b, identity, nothing, 1e-5)
    end
end
