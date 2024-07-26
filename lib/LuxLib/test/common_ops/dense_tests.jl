@testsetup module DenseSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, Enzyme, NNlib
using LuxTestUtils: @jet, @test_gradients
using DispatchDoctor: allow_unstable

anonact = x -> x^3

function run_dense_testing(gen_f, Tw, Tx, M, N, hasbias, activation, aType, mode, on_gpu)
    bias = hasbias ? gen_f(Tw, M) |> aType : nothing
    w = gen_f(Tw, M, N) |> aType
    x = gen_f(Tx, N, 3) |> aType

    y = fused_dense_bias_activation(activation, w, x, bias)
    y_generic = LuxLib.__generic_dense_bias_activation(activation, w, x, bias)

    @test y ≈ y_generic
    @test eltype(y) == promote_type(Tw, Tx)

    @test @inferred(fused_dense_bias_activation(activation, w, x, bias)) isa Any
    @jet fused_dense_bias_activation(activation, w, x, bias)

    __f = (σ, w, x, b) -> sum(abs2, fused_dense_bias_activation(σ, w, x, b))

    if activation !== anonact
        @test @inferred(Zygote.gradient(__f, activation, w, x, bias)) isa Any
    else
        @test length(@inferred(Zygote.gradient(__f, activation, w, x, bias)))==4 broken=true
    end

    fp16 = Tx == Float16 || Tw == Float16
    atol = fp16 ? 1.0f-1 : 1.0f-3
    rtol = fp16 ? 1.0f-1 : 1.0f-3

    if !on_gpu
        _, ∂w_zyg, ∂x_zyg, ∂b_zyg = Zygote.gradient(__f, activation, w, x, bias)

        ∂w_enz = Enzyme.make_zero(w)
        ∂x_enz = Enzyme.make_zero(x)
        ∂b = if hasbias
            ∂b_enz = Enzyme.make_zero(bias)
            Duplicated(bias, ∂b_enz)
        else
            Const(nothing)
        end
        Enzyme.autodiff(Reverse, __f, Active, Const(activation),
            Duplicated(w, ∂w_enz), Duplicated(x, ∂x_enz), ∂b)

        @test ∂w_zyg≈∂w_enz rtol=rtol atol=atol
        @test ∂x_zyg≈∂x_enz rtol=rtol atol=atol
        hasbias && @test ∂b_zyg≈∂b.dval rtol=rtol atol=atol
    end

    allow_unstable() do
        @eval @test_gradients $__f $activation $w $x $bias gpu_testing=$on_gpu atol=$atol rtol=$rtol skip_reverse_diff=$(Tx !=
                                                                                                                         Tw) skip_finite_differences=$(Tx !=
                                                                                                                                                       Tw)
    end
end

const ALL_TEST_CONFIGS = Iterators.product(
    ((Float16, Float16), (Float32, Float16), (Float32, Float32),
        (Float32, Float64), (Float64, Float64)),
    (4, 8),
    (4, 8),
    (true, false),
    (identity, tanh, tanh_fast, sigmoid, sigmoid_fast, relu, gelu, anonact))

const TEST_BLOCKS = collect(Iterators.partition(
    ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 5)))

export ALL_TEST_CONFIGS, TEST_BLOCKS, run_dense_testing

end

@testitem "Fused Dense: Group 1" tags=[:dense] setup=[SharedTestSetup, DenseSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $Tw x $Tx, size $M x $N, bias $hasbias, activation $activation" for ((Tx, Tw), M, N, hasbias, activation) in TEST_BLOCKS[1]
            run_dense_testing(__generate_fixed_array, Tw, Tx, M, N,
                hasbias, activation, aType, mode, on_gpu)
        end
    end
end

@testitem "Fused Dense: Group 2" tags=[:dense] setup=[SharedTestSetup, DenseSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $Tw x $Tx, size $M x $N, bias $hasbias, activation $activation" for ((Tx, Tw), M, N, hasbias, activation) in TEST_BLOCKS[2]
            run_dense_testing(__generate_fixed_array, Tw, Tx, M, N,
                hasbias, activation, aType, mode, on_gpu)
        end
    end
end

@testitem "Fused Dense: Group 3" tags=[:dense] setup=[SharedTestSetup, DenseSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $Tw x $Tx, size $M x $N, bias $hasbias, activation $activation" for ((Tx, Tw), M, N, hasbias, activation) in TEST_BLOCKS[3]
            run_dense_testing(__generate_fixed_array, Tw, Tx, M, N,
                hasbias, activation, aType, mode, on_gpu)
        end
    end
end

@testitem "Fused Dense: Group 4" tags=[:dense] setup=[SharedTestSetup, DenseSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $Tw x $Tx, size $M x $N, bias $hasbias, activation $activation" for ((Tx, Tw), M, N, hasbias, activation) in TEST_BLOCKS[4]
            run_dense_testing(__generate_fixed_array, Tw, Tx, M, N,
                hasbias, activation, aType, mode, on_gpu)
        end
    end
end

@testitem "Fused Dense: Group 5" tags=[:dense] setup=[SharedTestSetup, DenseSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $Tw x $Tx, size $M x $N, bias $hasbias, activation $activation" for ((Tx, Tw), M, N, hasbias, activation) in TEST_BLOCKS[5]
            run_dense_testing(__generate_fixed_array, Tw, Tx, M, N,
                hasbias, activation, aType, mode, on_gpu)
        end
    end
end

@testitem "Fused Dense: StaticArrays" tags=[:dense] begin
    using StaticArrays

    x = @SArray rand(2, 4)
    weight = @SArray rand(3, 2)
    bias = @SArray rand(3)

    @test @inferred(fused_dense_bias_activation(relu, weight, x, bias)) isa SArray
end
