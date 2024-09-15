@testsetup module DenseSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, NNlib, StableRNGs

anonact = x -> x^3

dense_simple(act, w, x, ::Nothing) = act.(w * x)
dense_simple(act, w, x, b) = act.(w * x .+ b)

function run_dense_testing(Tw, Tx, M, N, hasbias, activation, aType, mode, ongpu)
    rng = StableRNG(1234)

    bias = hasbias ? randn(rng, Tw, M) |> aType : nothing
    w = randn(rng, Tw, M, N) |> aType
    x = randn(rng, Tx, N, 3) |> aType

    if activation === tanh_fast || activation === tanh || activation === gelu
        bias = bias === nothing ? nothing : (bias .* eltype(bias)(0.001))
        w = w .* eltype(w)(0.001)
        x = x .* eltype(x)(0.001)
    end

    y = fused_dense_bias_activation(activation, w, x, bias)
    y_generic = bias === nothing ? activation.(w * x) : activation.(w * x .+ bias)

    @test y ≈ y_generic
    @test eltype(y) == promote_type(Tw, Tx)

    @test @inferred(fused_dense_bias_activation(activation, w, x, bias)) isa Any
    @jet fused_dense_bias_activation(activation, w, x, bias)

    fp16 = Tx == Float16 || Tw == Float16
    atol = fp16 ? 1.0f-1 : 1.0f-3
    rtol = fp16 ? 1.0f-1 : 1.0f-3

    __f = (σ, w, x, b) -> sum(abs2, fused_dense_bias_activation(σ, w, x, b))

    if !fp16 && activation !== anonact
        @test @inferred(Zygote.gradient(__f, activation, w, x, bias)) isa Any
    end

    skip_backends = []
    Tw != Tx && push!(skip_backends, AutoReverseDiff())
    fp16 && push!(skip_backends, AutoFiniteDiff())
    fp16 && push!(skip_backends, AutoTracker())

    __f_grad = let activation = activation
        (w, x, b) -> __f(activation, w, x, b)
    end
    test_gradients(__f_grad, w, x, bias; atol, rtol, skip_backends, soft_fail=fp16)

    y_simple = dense_simple(activation, w, x, bias)
    y_zyg = fused_dense_bias_activation(activation, w, x, bias)
    @test y_simple≈y_zyg atol=atol rtol=rtol

    _, ∂w_true, ∂x_true, ∂b_true = Zygote.gradient(
        sum ∘ dense_simple, activation, w, x, bias)
    _, ∂w_zyg, ∂x_zyg, ∂b_zyg = Zygote.gradient(
        sum ∘ fused_dense_bias_activation, activation, w, x, bias)
    @test ∂w_true≈∂w_zyg atol=atol rtol=rtol
    @test ∂x_true≈∂x_zyg atol=atol rtol=rtol
    if bias !== nothing
        @test ∂b_true≈∂b_zyg atol=atol rtol=rtol
    end
end

const ALL_TEST_CONFIGS = Iterators.product(
    ((Float16, Float16), (Float32, Float16), (Float32, Float32),
        (Float32, Float64), (Float64, Float64)),
    (4, 32),
    (4, 32),
    (true, false),
    (identity, tanh, tanh_fast, sigmoid, sigmoid_fast, relu, gelu, anonact))

const TEST_BLOCKS = collect(Iterators.partition(
    ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 5)))

export ALL_TEST_CONFIGS, TEST_BLOCKS, run_dense_testing

end

@testitem "Fused Dense: Group 1" tags=[:dense] setup=[SharedTestSetup, DenseSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $Tw x $Tx, size $M x $N, bias $hasbias, activation $activation" for ((Tx, Tw), M, N, hasbias, activation) in TEST_BLOCKS[1]
            run_dense_testing(Tw, Tx, M, N, hasbias, activation, aType, mode, ongpu)
        end
    end
end

@testitem "Fused Dense: Group 2" tags=[:dense] setup=[SharedTestSetup, DenseSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $Tw x $Tx, size $M x $N, bias $hasbias, activation $activation" for ((Tx, Tw), M, N, hasbias, activation) in TEST_BLOCKS[2]
            run_dense_testing(Tw, Tx, M, N, hasbias, activation, aType, mode, ongpu)
        end
    end
end

@testitem "Fused Dense: Group 3" tags=[:dense] setup=[SharedTestSetup, DenseSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $Tw x $Tx, size $M x $N, bias $hasbias, activation $activation" for ((Tx, Tw), M, N, hasbias, activation) in TEST_BLOCKS[3]
            run_dense_testing(Tw, Tx, M, N, hasbias, activation, aType, mode, ongpu)
        end
    end
end

@testitem "Fused Dense: Group 4" tags=[:dense] setup=[SharedTestSetup, DenseSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $Tw x $Tx, size $M x $N, bias $hasbias, activation $activation" for ((Tx, Tw), M, N, hasbias, activation) in TEST_BLOCKS[4]
            run_dense_testing(Tw, Tx, M, N, hasbias, activation, aType, mode, ongpu)
        end
    end
end

@testitem "Fused Dense: Group 5" tags=[:dense] setup=[SharedTestSetup, DenseSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $Tw x $Tx, size $M x $N, bias $hasbias, activation $activation" for ((Tx, Tw), M, N, hasbias, activation) in TEST_BLOCKS[5]
            run_dense_testing(Tw, Tx, M, N, hasbias, activation, aType, mode, ongpu)
        end
    end
end

@testitem "Fused Dense: StaticArrays" tags=[:dense] begin
    using StaticArrays, NNlib

    x = @SArray rand(2, 4)
    weight = @SArray rand(3, 2)
    bias = @SArray rand(3)

    @test @inferred(fused_dense_bias_activation(relu, weight, x, bias)) isa SArray
end

@testitem "Fused Dense: CPU No Scalar Indexing" tags=[:dense] begin
    using JLArrays, NNlib

    x = JLArray(rand(Float32, 2, 4))
    weight = JLArray(rand(Float32, 3, 2))
    bias = JLArray(rand(Float32, 3))

    @test @inferred(fused_dense_bias_activation(relu, weight, x, bias)) isa JLArray
    @test LuxLib.internal_operation_mode(x) isa LuxLib.GenericBroadcastOp
end

@testitem "`LuxLib.Impl.matmul(add)` allocations" tags=[:dense] setup=[SharedTestSetup] begin
    using BenchmarkTools, Statistics

    if BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
        @testset "size $N" for N in (1, 4, 32, 256, 1024)
            x = rand(Float32, N, N)

            trial_opt = median(@benchmark(LuxLib.Impl.matmul($x, $x)))
            trial_baseline = median(@benchmark($x*$x))

            @test trial_opt.allocs ≤ trial_baseline.allocs
            @test trial_opt.memory ≤ trial_baseline.memory

            bias = rand(Float32, N)

            trial_opt = median(@benchmark(LuxLib.Impl.matmuladd($x, $x, $bias)))
            trial_baseline = median(@benchmark(muladd($x, $x, $bias)))

            @test trial_opt.allocs ≤ trial_baseline.allocs
            @test trial_opt.memory ≤ trial_baseline.memory
        end
    end
end

@testitem "Enzyme.Forward patch: dense" tags=[:dense] setup=[SharedTestSetup] skip=:(using LuxTestUtils; !LuxTestUtils.ENZYME_TESTING_ENABLED) begin
    using LuxLib, Random, LuxTestUtils, Enzyme

    x = rand(Float32, 2, 2)

    f(x) = sum(abs2, LuxLib.Impl.matmul(x, x))

    # Just test that we don't crash
    @test length(Enzyme.gradient(Forward, f, x)) == 4
end

@testitem "Enzyme rules for fused dense" tags=[:dense] setup=[SharedTestSetup] skip=:(using LuxTestUtils; !LuxTestUtils.ENZYME_TESTING_ENABLED) begin
    using LuxLib, NNlib, Zygote, Enzyme

    # These are mostly for testing the CUDA rules since we don't enable the CUDA tests
    # in LuxTestUtils currently
    function fused_dense!(y, act, weight, x, b)
        op = LuxLib.internal_operation_mode((y, weight, x, b))
        LuxLib.Impl.fused_dense!(y, op, act, weight, x, b)
        return
    end

    function matmuladd!(C, A, B, bias)
        op = LuxLib.internal_operation_mode((C, A, B, bias))
        LuxLib.Impl.matmuladd!(C, op, A, B, bias)
        return
    end

    rng = StableRNG(1234)

    ALL_ACTS = [identity, tanh, tanh_fast, sigmoid, sigmoid_fast,
        relu, gelu, x -> x^3, x -> gelu(x)]

    @testset "$mode" for (mode, aType, ongpu) in MODES
        mode ∈ ("cpu", "cuda") || continue

        y = zeros(Float32, 2, 2) |> aType
        weight = randn(rng, Float32, 2, 2) |> aType
        x = randn(rng, Float32, 2, 2) |> aType
        @testset for (act, hasbias) in Iterators.product(ALL_ACTS, (true, false))
            b = hasbias ? aType(randn(rng, Float32, 2)) : nothing

            dy = randn(rng, Float32, 2, 2) |> aType

            dweight = zeros(Float32, 2, 2) |> aType
            dx = zeros(Float32, 2, 2) |> aType
            db = hasbias ? aType(zeros(Float32, 2)) : nothing

            b_enz = hasbias ? Duplicated(b, db) : Const(b)

            Enzyme.autodiff(Reverse, fused_dense!, Duplicated(y, copy(dy)), Const(act),
                Duplicated(weight, dweight), Duplicated(x, dx), b_enz)

            _, pb_f = Zygote.pullback(fused_dense_bias_activation, act, weight, x, b)
            _, dweight_zyg, dx_zyg, db_zyg = pb_f(dy)

            @test dweight≈dweight_zyg atol=1e-3 rtol=1e-3
            @test dx≈dx_zyg atol=1e-3 rtol=1e-3
            if hasbias
                @test db≈db_zyg atol=1e-3 rtol=1e-3
            end

            act === identity || !hasbias || continue

            Enzyme.autodiff(Reverse, matmuladd!, Duplicated(y, copy(dy)),
                Duplicated(weight, dweight), Duplicated(x, dx), b_enz)

            _, pb_f = Zygote.pullback(matmuladd, weight, x, b)
            dweight_zyg, dx_zyg, db_zyg = pb_f(dy)

            @test dweight≈dweight_zyg atol=1e-3 rtol=1e-3
            @test dx≈dx_zyg atol=1e-3 rtol=1e-3
            @test db≈db_zyg atol=1e-3 rtol=1e-3
        end
    end
end
