using Pkg, SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    return Pkg.develop(PackageSpec(; path=subpkg_path))
end

function activate_subpkg_env(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.activate(subpkg_path)
    Pkg.develop(PackageSpec(; path=subpkg_path))
    return Pkg.instantiate()
end

groups = if GROUP == "All"
    ["Lux", "Boltz", "LuxLib", "Flux2Lux"]
else
    [GROUP]
end

@time begin for group in groups
    @info "Testing GROUP $group"
    if group == "Lux"
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
            end
        end
    else
        dev_subpkg(group)
        subpkg_path = joinpath(dirname(@__DIR__), "lib", group)
        Pkg.test(PackageSpec(; name=group, path=subpkg_path))
    end
end end
