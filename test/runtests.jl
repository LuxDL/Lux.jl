using SafeTestsets, Test

@testset "Lux.jl" begin
    @time @safetestset "Utils" begin include("utils.jl") end

    @time @safetestset "Core" begin include("core.jl") end

    @time @safetestset "Adapt" begin include("adapt.jl") end

    @testset "Layers" begin
        @time @safetestset "Basic" begin include("layers/basic.jl") end
        @time @safetestset "Containers" begin include("layers/containers.jl") end
        @time @safetestset "Convolution" begin include("layers/conv.jl") end
        @time @safetestset "Normalization" begin include("layers/normalize.jl") end
        @time @safetestset "Recurrent" begin include("layers/recurrent.jl") end
        @time @safetestset "Dropout" begin include("layers/dropout.jl") end
    end

    @time @safetestset "NNlib" begin include("nnlib.jl") end

    @time @safetestset "Automatic Differentiation" begin include("autodiff.jl") end

    @testset "Experimental" begin
        @time @safetestset "Map" begin include("contrib/map.jl") end
        @time @safetestset "Training" begin include("contrib/training.jl") end
        @time @safetestset "Freeze" begin include("contrib/freeze.jl") end
        @time @safetestset "Shared Parameters" begin include("contrib/share_parameters.jl") end
    end

    @testset "Extensions" begin
        # Most CA tests are already included in the other tests
        @time @safetestset "ComponentArrays" begin include("ext/LuxComponentArraysExt.jl") end

        @time @safetestset "Flux" begin include("ext/LuxFluxTransformExt.jl") end
    end
end
