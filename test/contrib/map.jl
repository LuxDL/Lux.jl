using Lux, Random, Setfield, Test

c = Parallel(+;
             chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3), dense_2=Dense(3 => 5)),
             dense_3=Dense(5 => 1))

rng = Random.default_rng()
ps, st = Lux.setup(rng, c)

function zero_dense_params(l, ps, st, name)
    if l isa Dense && occursin("model.chain", name)
        @set! ps.weight = zero.(ps.weight)
        @set! ps.bias = zero.(ps.bias)
    end
    return l, ps, st
end

c_, ps_, st_ = Lux.layer_map(zero_dense_params, c, ps, st)

@test ps_.chain.dense_1.weight == zeros(3, 2)
@test ps_.chain.dense_1.bias == zeros(3, 1)
@test ps_.chain.dense_2.weight == zeros(5, 3)
@test ps_.chain.dense_2.bias == zeros(5, 1)
@test ps_.dense_3.weight != zeros(1, 5)
@test ps_.dense_3.bias == zeros(1, 1)

function zero_dense_params(l, ps, st, name)
    if l isa Dense && occursin("c.chain", name)
        @set! ps.weight = zero.(ps.weight)
        @set! ps.bias = zero.(ps.bias)
    end
    return l, ps, st
end

c_, ps_, st_ = Lux.@layer_map zero_dense_params c ps st

@test ps_.chain.dense_1.weight == zeros(3, 2)
@test ps_.chain.dense_1.bias == zeros(3, 1)
@test ps_.chain.dense_2.weight == zeros(5, 3)
@test ps_.chain.dense_2.bias == zeros(5, 1)
@test ps_.dense_3.weight != zeros(1, 5)
@test ps_.dense_3.bias == zeros(1, 1)

# Custom Layers -- See https://github.com/avik-pal/Lux.jl/issues/187
struct SimpleCustom{L1, L2} <: Lux.AbstractExplicitContainerLayer{(:dense, :conv)}
    dense::L1
    conv::L2
end

l = SimpleCustom(Dense(3 => 2), Conv((3,), 3 => 2))

ps, st = Lux.setup(rng, l)

function zero_dense_params(l, ps, st, name)
    if l isa Dense
        @set! ps.weight = zero.(ps.weight)
        @set! ps.bias = zero.(ps.bias)
    end
    return l, ps, st
end

l_, ps_, st_ = Lux.@layer_map zero_dense_params l ps st

@test ps_.dense.weight == zeros(Float32, 2, 3)
@test ps_.dense.bias == zeros(Float32, 2, 1)
@test ps_.conv.weight != zeros(Float32, 3, 3, 2)
@test ps_.conv.bias == zeros(Float32, 1, 2, 1)
