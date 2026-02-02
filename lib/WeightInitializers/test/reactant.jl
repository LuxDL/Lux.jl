using WeightInitializers, Test, Reactant

@testset "Initialization inside compile" begin
    rrng = Reactant.ReactantRNG()

    @testset "Concrete: $(op)" for op in (zeros32, ones32)
        gen_arr = op(rrng, 3, 4)
        @test eltype(gen_arr) == Float32
        @test size(gen_arr) == (3, 4)
        @test gen_arr isa Reactant.ConcreteRArray{Float32,2}
    end

    @testset "Traced: $(op)" for op in (zeros32, ones32, rand32, randn32)
        gen_arr = @jit op(rrng, 3, 4)
        @test eltype(gen_arr) == Float32
        @test size(gen_arr) == (3, 4)
        @test gen_arr isa Reactant.ConcreteRArray{Float32,2}
    end
end
