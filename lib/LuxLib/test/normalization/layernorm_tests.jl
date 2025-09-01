@testsetup module LayerNormSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, NNlib, Statistics
using LuxTestUtils: check_approx

function setup_layernorm(gen_f, aType, T, x_size, affine_shape, expand_dims::Bool=true)
    x = aType(gen_f(T, x_size))
    if affine_shape !== nothing
        if expand_dims
            scale = aType(gen_f(T, (affine_shape..., 1)))
            bias = aType(gen_f(T, (affine_shape..., 1)))
        else
            scale = aType(gen_f(T, affine_shape))
            bias = aType(gen_f(T, affine_shape))
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
            aType, T, x_size, affine_shape, act, dims, x, scale, bias
        )
    end
end

sumabs2layernorm(args...) = sum(abs2, layernorm(args...))

function run_layernorm_testing_core(
    aType, T, x_size, affine_shape, act, dims, x, scale, bias
)
    epsilon = LuxLib.Utils.default_epsilon(T)
    _f = (args...) -> layernorm(args..., act, dims, epsilon)

    @test @inferred(layernorm(x, scale, bias, act, dims, epsilon)) isa Any
    @jet layernorm(x, scale, bias, act, dims, epsilon)

    y = _f(x, scale, bias)

    @test y isa aType{T,length(x_size)}
    @test size(y) == x_size

    if affine_shape === nothing && act === identity
        @test check_approx(mean(y; dims), 0; atol=1.0e-3, rtol=1.0e-3)
        @test check_approx(std(y; dims), 1; atol=1.0e-1, rtol=1.0e-1)
    end

    atol = 1.0f-3
    rtol = 1.0f-3

    @test_gradients(
        sumabs2layernorm,
        x,
        scale,
        bias,
        act,
        dims,
        epsilon;
        atol,
        rtol,
        soft_fail=[AutoFiniteDiff()]
    )
end

anonact = x -> x^3

const ALL_TEST_CONFIGS = Any[]

for T in (Float32, Float64),
    # x_shape in ((3, 3, 2, 1), (2, 2, 2, 1), (2, 3, 2, 2)),
    x_shape in ((3, 3, 2, 1), (2, 2, 2, 1)),
    # affine_shape in (nothing, x_shape[1:3], (1, 1, 1), (1, 1, x_shape[3])),
    affine_shape in (nothing, (1, 1, x_shape[3])),
    act in (identity, sigmoid_fast, anonact)

    push!(ALL_TEST_CONFIGS, (T, x_shape, affine_shape, act))
end

const TEST_BLOCKS = collect(
    Iterators.partition(ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 2))
)

export ALL_TEST_CONFIGS, TEST_BLOCKS, run_layernorm_testing

end

@testitem "Layer Norm: Group 1" tags = [:normalization] setup = [
    SharedTestSetup, LayerNormSetup
] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "eltype $T, size $x_shape, $act" for (T, x_shape, affine_shape, act) in
                                                      TEST_BLOCKS[1]
            !fp64 && T == Float64 && continue
            run_layernorm_testing(
                generate_fixed_array, aType, T, x_shape, affine_shape, act, ongpu, mode
            )
        end
    end
end

@testitem "Layer Norm: Group 2" tags = [:normalization] setup = [
    SharedTestSetup, LayerNormSetup
] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "eltype $T, size $x_shape, $act" for (T, x_shape, affine_shape, act) in
                                                      TEST_BLOCKS[2]
            !fp64 && T == Float64 && continue
            run_layernorm_testing(
                generate_fixed_array, aType, T, x_shape, affine_shape, act, ongpu, mode
            )
        end
    end
end

@testitem "Layer Norm: Error Checks" tags = [:normalization] setup = [SharedTestSetup] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        !fp64 && continue

        x = aType(rand(2, 3))

        @test_throws ArgumentError layernorm(x, nothing, nothing, identity, nothing, 1.0e-5)

        sc = aType(rand(2, 1))
        b = aType(rand(2, 1))

        @test_throws AssertionError layernorm(x, sc, b, identity, nothing, 1.0e-5)
    end
end
