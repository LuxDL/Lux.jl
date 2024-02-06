using Lux, Setfield, Test

include("../test_utils.jl")

function zero_dense_params_1(l, ps, st, name)
    if l isa Dense && occursin("model.layers.chain", name)
        @set! ps.weight = zero.(ps.weight)
        @set! ps.bias = zero.(ps.bias)
    end
    return l, ps, st
end

function zero_dense_params_2(l, ps, st, name)
    if l isa Dense && occursin("c.layers.chain", name)
        @set! ps.weight = zero.(ps.weight)
        @set! ps.bias = zero.(ps.bias)
    end
    return l, ps, st
end

function zero_dense_params_3(l, ps, st, name)
    if l isa Dense
        @set! ps.weight = zero.(ps.weight)
        @set! ps.bias = zero.(ps.bias)
    end
    return l, ps, st
end

@testset "$mode" for (mode, aType, device, ongpu) in MODES
    c = Parallel(+;
        chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3), dense_2=Dense(3 => 5)),
        dense_3=Dense(5 => 1))

    rng = get_stable_rng(12345)
    ps, st = Lux.setup(rng, c) .|> device

    c_, ps_, st_ = Lux.Experimental.layer_map(zero_dense_params_1, c, ps, st)

    @test all(iszero, ps_.chain.dense_1.weight)
    @test all(iszero, ps_.chain.dense_1.bias)
    @test all(iszero, ps_.chain.dense_2.weight)
    @test all(iszero, ps_.chain.dense_2.bias)
    @test !all(iszero, ps_.dense_3.weight)
    @test all(iszero, ps_.dense_3.bias)

    c_, ps_, st_ = Lux.Experimental.@layer_map zero_dense_params_2 c ps st

    @test all(iszero, ps_.chain.dense_1.weight)
    @test all(iszero, ps_.chain.dense_1.bias)
    @test all(iszero, ps_.chain.dense_2.weight)
    @test all(iszero, ps_.chain.dense_2.bias)
    @test !all(iszero, ps_.dense_3.weight)
    @test all(iszero, ps_.dense_3.bias)

    # Custom Layers -- See https://github.com/avik-pal/Lux.jl/issues/187
    struct SimpleCustom{L1, L2} <: Lux.AbstractExplicitContainerLayer{(:dense, :conv)}
        dense::L1
        conv::L2
    end

    l = SimpleCustom(Dense(3 => 2), Conv((3,), 3 => 2))

    ps, st = Lux.setup(rng, l) .|> device

    l_, ps_, st_ = Lux.Experimental.@layer_map zero_dense_params_3 l ps st

    @test all(iszero, ps_.dense.weight)
    @test all(iszero, ps_.dense.bias)
    @test !all(iszero, ps_.conv.weight)
    @test all(iszero, ps_.conv.bias)
end
