using Functors, LuxCore, Optimisers, Random, Test

rng = LuxCore._default_rng()

# Define some custom layers
struct Dense <: LuxCore.AbstractExplicitLayer
    in::Int
    out::Int
end

function LuxCore.initialparameters(rng::AbstractRNG, l::Dense)
    return (w=randn(rng, l.out, l.in), b=randn(rng, l.out))
end

(::Dense)(x, ps, st) = x, st  # Dummy Forward Pass

struct Chain{L} <: LuxCore.AbstractExplicitContainerLayer{(:layers,)}
    layers::L
end

function (c::Chain)(x, ps, st)
    y, st1 = c.layers[1](x, ps.layer_1, st.layer_1)
    y, st2 = c.layers[2](y, ps.layer_2, st.layer_2)
    return y, (layers = (st1, st2))
end

struct Chain2{L1, L2} <: LuxCore.AbstractExplicitContainerLayer{(:layer1, :layer2)}
    layer1::L1
    layer2::L2
end

function (c::Chain2)(x, ps, st)
    y, st1 = c.layer1(x, ps.layer1, st.layer1)
    y, st2 = c.layer1(y, ps.layer2, st.layer2)
    return y, (; layer1=st1, layer2=st2)
end

@testset "AbstractExplicitLayer Interface" begin
    @testset "Custom Layer" begin
        model = Dense(5, 6)
        x = randn(rng, Float32, 5)
        ps, st = LuxCore.setup(rng, model)

        @test LuxCore.parameterlength(ps) == LuxCore.parameterlength(model)
        @test LuxCore.statelength(st) == LuxCore.statelength(model)

        @test LuxCore.apply(model, x, ps, st) == model(x, ps, st)

        @test_nowarn println(model)
    end

    @testset "Default Fallbacks" begin
        struct NoParamStateLayer <: LuxCore.AbstractExplicitLayer end

        layer = NoParamStateLayer()
        @test LuxCore.initialparameters(rng, layer) == NamedTuple()
        @test LuxCore.initialstates(rng, layer) == NamedTuple()

        @test LuxCore.parameterlength(zeros(10, 2)) == 20
        @test LuxCore.statelength(zeros(10, 2)) == 20
        @test LuxCore.statelength(Val(true)) == 1
        @test LuxCore.statelength((zeros(10), zeros(5, 2))) == 20
        @test LuxCore.statelength((layer_1=zeros(10), layer_2=zeros(5, 2))) == 20

        @test LuxCore.initialparameters(rng, NamedTuple()) == NamedTuple()
        @test_throws MethodError LuxCore.initialparameters(rng, ())
        @test LuxCore.initialparameters(rng, nothing) == NamedTuple()

        @test LuxCore.initialstates(rng, NamedTuple()) == NamedTuple()
        @test_throws MethodError LuxCore.initialstates(rng, ())
        @test LuxCore.initialstates(rng, nothing) == NamedTuple()
    end
end

@testset "AbstractExplicitContainerLayer Interface" begin
    model = Chain((; layer_1=Dense(5, 5), layer_2=Dense(5, 6)))
    x = randn(rng, Float32, 5)
    ps, st = LuxCore.setup(rng, model)

    @test LuxCore.parameterlength(ps) ==
          LuxCore.parameterlength(model) ==
          LuxCore.parameterlength(model.layers[1]) +
          LuxCore.parameterlength(model.layers[2])
    @test LuxCore.statelength(st) ==
          LuxCore.statelength(model) ==
          LuxCore.statelength(model.layers[1]) + LuxCore.statelength(model.layers[2])

    @test LuxCore.apply(model, x, ps, st) == model(x, ps, st)

    @test_nowarn println(model)

    model = Chain2(Dense(5, 5), Dense(5, 6))
    x = randn(rng, Float32, 5)
    ps, st = LuxCore.setup(rng, model)

    @test LuxCore.parameterlength(ps) ==
          LuxCore.parameterlength(model) ==
          LuxCore.parameterlength(model.layer1) + LuxCore.parameterlength(model.layer2)
    @test LuxCore.statelength(st) ==
          LuxCore.statelength(model) ==
          LuxCore.statelength(model.layer1) + LuxCore.statelength(model.layer2)

    @test LuxCore.apply(model, x, ps, st) == model(x, ps, st)

    @test_nowarn println(model)
end

@testset "update_state API" begin
    st = (layer_1=(training=Val(true), val=1),
        layer_2=(layer_1=(val=2,), layer_2=(training=Val(true),)))

    st_ = LuxCore.testmode(st)

    @test st_.layer_1.training == Val(false) &&
          st_.layer_2.layer_2.training == Val(false) &&
          st_.layer_1.val == st.layer_1.val &&
          st_.layer_2.layer_1.val == st.layer_2.layer_1.val

    st = st_
    st_ = LuxCore.trainmode(st)

    @test st_.layer_1.training == Val(true) &&
          st_.layer_2.layer_2.training == Val(true) &&
          st_.layer_1.val == st.layer_1.val &&
          st_.layer_2.layer_1.val == st.layer_2.layer_1.val

    st_ = LuxCore.update_state(st, :val, -1)
    @test st_.layer_1.training == st.layer_1.training &&
          st_.layer_2.layer_2.training == st.layer_2.layer_2.training &&
          st_.layer_1.val == -1 &&
          st_.layer_2.layer_1.val == -1
end

@testset "Functor Compatibilty" begin
    @testset "Basic Usage" begin
        model = Chain((; layer_1=Dense(5, 10), layer_2=Dense(10, 5)))

        children, reconstructor = Functors.functor(model)

        @test children isa NamedTuple
        @test fieldnames(typeof(children)) == (:layers,)
        @test children.layers isa NamedTuple
        @test fieldnames(typeof(children.layers)) == (:layer_1, :layer_2)
        @test children.layers.layer_1 isa Dense
        @test children.layers.layer_2 isa Dense
        @test children.layers.layer_1.in == 5
        @test children.layers.layer_1.out == 10
        @test children.layers.layer_2.in == 10
        @test children.layers.layer_2.out == 5

        new_model = reconstructor((; layers=(; layer_1=Dense(10, 5), layer_2=Dense(5, 10))))

        @test new_model isa Chain
        @test new_model.layers.layer_1.in == 10
        @test new_model.layers.layer_1.out == 5
        @test new_model.layers.layer_2.in == 5
        @test new_model.layers.layer_2.out == 10
    end

    @testset "Method Ambiguity" begin
        # Needed if defining a layer that works with both Flux and Lux -- See DiffEqFlux.jl
        # See https://github.com/SciML/DiffEqFlux.jl/pull/750#issuecomment-1373874944

        struct CustomLayer{M, P} <: LuxCore.AbstractExplicitContainerLayer{(:model,)}
            model::M
            p::P
        end

        @functor CustomLayer (p,)

        l = CustomLayer(x -> x, nothing)  # Dummy Struct

        @test_nowarn Optimisers.trainable(l)
    end
end
