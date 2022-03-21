using Test, Random
import Flux: relu
import ExplicitFluxLayers: apply, setup, Dense

@testset "ExplicitFluxLayers" begin
    @testset "Linear" begin
        include("linear.jl")
    end
end
