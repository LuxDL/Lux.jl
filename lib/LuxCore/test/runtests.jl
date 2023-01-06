using Functors, LuxCore, Optimisers, Random, Test

@testset "LuxCore.jl" begin
    rng = LuxCore._default_rng()

    @testset "AbstractExplicitLayer Interface" begin
        struct Dense <: LuxCore.AbstractExplicitLayer
            in::Int
            out::Int
        end

        function LuxCore.initialparameters(rng::AbstractRNG, l::Dense)
            return (w=randn(rng, l.out, l.in), b=randn(rng, l.out))
        end

        model = Dense(5, 6)
        ps, st = LuxCore.setup(rng, model)

        @test LuxCore.parameterlength(ps) == LuxCore.parameterlength(model)
        @test LuxCore.parameterlength(zeros(10, 2)) == 20
        @test LuxCore.statelength(st) == LuxCore.statelength(model)
        @test LuxCore.statelength(zeros(10, 2)) == 20
        @test LuxCore.statelength(Val(true)) == 1
        @test LuxCore.statelength((zeros(10), zeros(5, 2))) == 20
        @test LuxCore.statelength((layer_1=zeros(10), layer_2=zeros(5, 2))) == 20
    end

    @testset "update_state" begin
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

    # NOTE(@avik-pal): Custom Layers and Functors are tested in test/core.jl (in Lux)
end

@testset "@functor method ambiguity" begin
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
