using Pkg, SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

_get_lib_path(subpkg) = joinpath(dirname(@__DIR__), "lib", subpkg)

function _dev_pkg(path)
    @info "Pkg.develop $path"
    return Pkg.develop(PackageSpec(; path))
end

groups = if GROUP == "All"
    ["Lux", "Boltz", "LuxLib", "Flux2Lux"]
else
    [GROUP]
end

cross_dependencies = Dict("Lux" => [_get_lib_path("LuxLib")],
                          "Boltz" => [_get_lib_path("LuxLib"), dirname(@__DIR__)],
                          "LuxLib" => [],
                          "Flux2Lux" => [_get_lib_path("LuxLib"), dirname(@__DIR__)])

const OVERRIDE_INTER_DEPENDENCIES = get(ENV, "OVERRIDE_INTER_DEPENDENCIES", "false") ==
                                    "true"

@time begin for group in groups
    @info "Testing GROUP $group"

    if group == "Lux"
        if !OVERRIDE_INTER_DEPENDENCIES
            # Use unreleased versions of inter-dependencies
            _dev_pkg.(cross_dependencies[group])
        end

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
        subpkg_path = _get_lib_path(group)
        _dev_pkg(subpkg_path)

        if !OVERRIDE_INTER_DEPENDENCIES
            # Use unreleased versions of inter-dependencies
            _dev_pkg.(cross_dependencies[group])
        end

        # this should inherit the GROUP envvar
        run_coverage = get(ENV, "COVERAGE", "false")
        Pkg.test(PackageSpec(; name=group, path=subpkg_path);
                 coverage=(run_coverage == "true"))
    end
end end
