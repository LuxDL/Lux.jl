using SafeTestsets, Test

@testset "Layers" begin
    @time @safetestset "Basic" begin include("layers/basic.jl") end
    @time @safetestset "Convolution" begin include("layers/conv.jl") end
    @time @safetestset "Normalization" begin include("layers/normalize.jl") end
    @time @safetestset "Recurrent" begin include("layers/recurrent.jl") end
    @time @safetestset "Dropout" begin include("layers/dropout.jl") end
end

@time @safetestset "Functional Operations" begin include("functional.jl") end

@testset "Metalhead Models" begin
    @time @safetestset "ConvNets -- ImageNet" begin include("models/convnets.jl") end
end
