using Optimisers, Random, EnzymeCore, MLDataDevices, LuxCore, Test, Functors, Setfield

rng = LuxCore.Internal.default_rng()

include("common.jl")

@testset "update_state API" begin
    st = (
        layer_1=(training=Val(true), val=1),
        layer_2=(layer_1=(val=2,), layer_2=(training=Val(true),)),
    )

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

@testset "Functor Compatibility" begin
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

        new_model = reconstructor((;
            layers=(; layer_1=Dense(10, 5), layer_2=Dense(5, 10)),
        ))

        @test new_model isa Chain
        @test new_model.layers.layer_1.in == 10
        @test new_model.layers.layer_1.out == 5
        @test new_model.layers.layer_2.in == 5
        @test new_model.layers.layer_2.out == 10

        model = ChainWrapper((; layer_1=Dense(5, 10), layer_2=Dense(10, 5)))

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

        new_model = reconstructor((;
            layers=(; layer_1=Dense(10, 5), layer_2=Dense(5, 10)),
        ))

        @test new_model isa ChainWrapper
        @test new_model.layers.layer_1.in == 10
        @test new_model.layers.layer_1.out == 5
        @test new_model.layers.layer_2.in == 5
        @test new_model.layers.layer_2.out == 10
    end

    @testset "Method Ambiguity" begin
        # Needed if defining a layer that works with both Flux and Lux -- See DiffEqFlux.jl
        # See https://github.com/SciML/DiffEqFlux.jl/pull/750#issuecomment-1373874944
        struct CustomLayer{M,P} <: AbstractLuxContainerLayer{(:model,)}
            model::M
            p::P
        end

        @functor CustomLayer (p,)

        l = CustomLayer(x -> x, nothing)  # Dummy Struct

        @test_nowarn Optimisers.trainable(l)
    end
end

@testset "Display Name" begin
    struct StructWithoutName <: AbstractLuxLayer end

    model = StructWithoutName()

    @test LuxCore.display_name(model) == "StructWithoutName"

    struct StructWithName{N} <: AbstractLuxLayer
        name::N
    end

    model = StructWithName("Test")

    @test LuxCore.display_name(model) == "Test"

    model = StructWithName(nothing)

    @test LuxCore.display_name(model) == "StructWithName"

    @test LuxCore.display_name(rand(20)) == "Array"
end

@testset "initialparameter/initialstate for Default Containers" begin
    models1 = [
        Chain((; layer_1=Dense(5, 10), layer_2=Dense(10, 5))),
        Chain2(Dense(5, 10), Dense(10, 5)),
        [Dense(5, 10), Dense(10, 5)],
    ]
    models2 = [
        Chain((; layer_1=Dense(5, 10), layer_2=Dense(10, 5))),
        Chain2(Dense(5, 10), Dense(10, 5)),
        (Dense(5, 10), Dense(10, 5)),
    ]

    for models in (models1, models2)
        ps, st = LuxCore.setup(rng, models)
        @test length(ps) == length(models)
        @test length(st) == length(models)
        @test typeof(ps[1]) == typeof(LuxCore.initialparameters(rng, models[1]))
        @test typeof(ps[2]) == typeof(LuxCore.initialparameters(rng, models[2]))
        @test typeof(ps[3][1]) == typeof(LuxCore.initialparameters(rng, models[3][1]))
        @test typeof(ps[3][2]) == typeof(LuxCore.initialparameters(rng, models[3][2]))
        @test typeof(st[1]) == typeof(LuxCore.initialstates(rng, models[1]))
        @test typeof(st[2]) == typeof(LuxCore.initialstates(rng, models[2]))
        @test typeof(st[3][1]) == typeof(LuxCore.initialstates(rng, models[3][1]))
        @test typeof(st[3][2]) == typeof(LuxCore.initialstates(rng, models[3][2]))
    end
end

@testset "Convenience Checks" begin
    models1 = [
        Chain((; layer_1=Dense(5, 10), layer_2=Dense(10, 5))),
        Chain2(Dense(5, 10), Dense(10, 5)),
        [Dense(5, 10), Dense(10, 5)],
    ]

    @test LuxCore.contains_lux_layer(models1)

    models2 = [1, 2, 3, 4]

    @test !LuxCore.contains_lux_layer(models2)

    models3 = [1, 2, 3, (; a=Dense(5, 10), b=Dense(10, 5))]

    @test LuxCore.contains_lux_layer(models3)
end

@testset "replicate" begin
    rng = Random.default_rng()
    @test LuxCore.replicate(rng) === rng
    @test LuxCore.replicate(rng) == rng

    rng = Xoshiro(1234)
    @test LuxCore.replicate(rng) !== rng
    @test LuxCore.replicate(rng) == rng
end

@testset "empty fleaves" begin
    @test length(fleaves(NamedTuple())) == 0
    @test !LuxCore.check_fmap_condition(isodd, nothing, NamedTuple())
end

@testset "Common Lux + Enzyme Mistakes" begin
    d = Dense(2, 2)

    @test_throws ArgumentError Active(d)
    @test_throws ArgumentError Duplicated(d, d)
    @test_throws ArgumentError DuplicatedNoNeed(d, d)
    @test_throws ArgumentError BatchDuplicated(d, (d, d))
    @test_throws ArgumentError BatchDuplicatedNoNeed(d, (d, d))
    @test Const(d) isa Const
end

@testset "Device Transfer Warnings" begin
    my_layer = Dense(2, 2)

    dev = cpu_device()
    @test_logs (
        :warn,
        "Lux layers are stateless and hence don't participate in device \
         transfers. Apply this function on the parameters and states generated \
         using `LuxCore.setup`.",
    ) dev(my_layer)
end

@testset "nested `training` key: Issue Lux.jl#849" begin
    st = (
        encoder=(layer_1=NamedTuple(), layer_2=(; training=Val{true}())),
        μ=NamedTuple(),
        logσ=NamedTuple(),
        decoder=(
            layer_1=NamedTuple(),
            layer_2=NamedTuple(),
            layer_3=NamedTuple(),
            layer_4=(running_mean=Float32[0.0, 0.0], training=Val{true}()),
        ),
        rng=Xoshiro(),
        training=Val{true}(),
    )

    @test st.encoder.layer_2.training isa Val{true}
    @test st.decoder.layer_4.training isa Val{true}
    @test st.training isa Val{true}

    st_test = LuxCore.testmode(st)

    @test st_test.encoder.layer_2.training isa Val{false}
    @test st_test.decoder.layer_4.training isa Val{false}
    @test st_test.training isa Val{false}

    st_train = LuxCore.trainmode(st_test)

    @test st_train.encoder.layer_2.training isa Val{true}
    @test st_train.decoder.layer_4.training isa Val{true}
    @test st_train.training isa Val{true}
end
