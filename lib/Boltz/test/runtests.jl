using SafeTestsets, Test

@time begin @testset verbose=true "Boltz.jl" begin @time @safetestset "Vision Models" begin include("vision.jl") end end end
