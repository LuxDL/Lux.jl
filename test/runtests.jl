using SafeTestsets, Test, TestSetExtensions

const GROUP = get(ENV, "GROUP", "All")

@testset ExtendedTestSet "Lux.jl" begin
    @safetestset "Utils" include("utils.jl")

    @safetestset "Core" include("core.jl")

    @testset "Layers" begin
        @safetestset "Basic" include("layers/basic.jl")
        @safetestset "Containers" include("layers/containers.jl")
        @safetestset "Convolution" include("layers/conv.jl")
        @safetestset "Normalization" include("layers/normalize.jl")
        @safetestset "Recurrent" include("layers/recurrent.jl")
        @safetestset "Dropout" include("layers/dropout.jl")
    end

    @testset "Experimental" begin
        @safetestset "Map" include("contrib/map.jl")
        @safetestset "Training" include("contrib/training.jl")
        @safetestset "Freeze" include("contrib/freeze.jl")
        @safetestset "Shared Parameters" include("contrib/share_parameters.jl")
        @safetestset "Debugging Tools" include("contrib/debug.jl")
        # Tests for StatefulLuxLayer is embedded into @compact tests
        @safetestset "Stateful & Compact Layers" include("contrib/compact.jl")
    end

    @safetestset "Aqua Tests" include("aqua.jl")

    @safetestset "Miscellaneous Tests" include("misc.jl")

    @testset "Extensions" begin
        # Most CA tests are already included in the other tests
        @safetestset "ComponentArrays" include("ext/LuxComponentArraysExt.jl")
        @safetestset "Flux" include("ext/LuxFluxTransformExt.jl")
    end
end
