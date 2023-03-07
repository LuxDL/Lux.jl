using Pkg

const GROUP = get(ENV, "GROUP", "All")

_get_lib_path(subpkg) = joinpath(dirname(@__DIR__), "lib", subpkg)

function _dev_pkg(path)
    @info "Pkg.develop $path"
    return Pkg.develop(PackageSpec(; path))
end

groups = if GROUP == "All"
    ["Lux", "Boltz", "LuxLib", "LuxCore", "LuxCUDA", "LuxAMDGPU"]
else
    [GROUP]
end

cross_dependencies = Dict("Lux" => [_get_lib_path("LuxLib"), _get_lib_path("LuxCore")],
                          "Boltz" => [
                              _get_lib_path("LuxLib"),
                              _get_lib_path("LuxCore"),
                              # dirname(@__DIR__),
                          ], "LuxLib" => [], "LuxCore" => [], "LuxCUDA" => [],
                          "LuxAMDGPU" => [])

const OVERRIDE_INTER_DEPENDENCIES = get(ENV, "OVERRIDE_INTER_DEPENDENCIES", "true") ==
                                    "true"

for group in groups
    @info "Testing GROUP $group"

    pkg_path = group == "Lux" ? dirname(@__DIR__) : _get_lib_path(group)
    Pkg.activate(pkg_path)

    if !OVERRIDE_INTER_DEPENDENCIES
        # Use unreleased versions of inter-dependencies
        _dev_pkg.(cross_dependencies[group])
    end

    Pkg.update()

    # this should inherit the GROUP envvar
    run_coverage = get(ENV, "COVERAGE", "false")
    Pkg.test(PackageSpec(; name=group, path=pkg_path); coverage=(run_coverage == "true"))
end
