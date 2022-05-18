using CUDA, Functors, JET, Lux, NNlib, Random, Statistics, Test, Zygote
# using Metalhead

# Some Helper Functions
function gradtest(model, input, ps, st)
    y, pb = Zygote.pullback(p -> model(input, p, st)[1], ps)
    gs = pb(ones(Float32, size(y)))
      
    # if we make it to here with no error, success!
    return true
end

function run_model(m::Lux.AbstractExplicitLayer, x, mode=:test)
    if mode == :test
        ps, st = Lux.setup(Random.default_rng(), m)
        st = Lux.testmode(st)
        return Lux.apply(m, x, ps, st)[1]
    end
end

function Base.isapprox(nt1::NamedTuple{fields}, nt2::NamedTuple{fields}) where {fields}
    checkapprox(xy) = xy[1] ≈ xy[2]
    checkapprox(t::Tuple{Nothing,Nothing}) = true
    all(checkapprox, zip(values(nt1), values(nt2)))
end

# Main Tests
@testset "Lux" begin
    @testset "Layers" begin
        @testset "Basic" begin
            include("layers/basic.jl")
        end
        @testset "Normalization" begin
            include("layers/normalize.jl")
        end
        @testset "Recurrent" begin
            include("layers/recurrent.jl")
        end
    end

    @testset "Functional Operations" begin
        include("functional.jl")
    end

    # Might not want to run always
    # @testset "Metalhead Models" begin
    #     @testset "ConvNets -- ImageNet" begin
    #         include("models/convnets.jl")
    #     end
    # end
end
