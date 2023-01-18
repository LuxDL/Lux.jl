using SafeTestsets, Test

@time begin
    @testset "Boltz.jl" begin
        @time @safetestset "Vision Models" begin include("vision.jl") end
    end
end
