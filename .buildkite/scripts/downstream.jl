using Pkg
using TOML

repo = ARGS[1]
if contains(repo, "#")
    repo, group = split(repo, "#")
else
    group = get(ARGS, 2, "CUDA")
end

pkg_name = endswith(repo, ".jl") ? String(repo[1:(end - 3)]) : String(repo)

lux_dir = normpath(joinpath(@__DIR__, "..", ".."))

local_pkgs = Dict{String,String}("Lux" => lux_dir)
lib_dir = joinpath(lux_dir, "lib")
if isdir(lib_dir)
    for d in readdir(lib_dir; join=true)
        if isdir(d) && isfile(joinpath(d, "Project.toml"))
            proj = try
                TOML.parsefile(joinpath(d, "Project.toml"))
            catch
                Dict{String,Any}()
            end
            if haskey(proj, "name")
                local_pkgs[proj["name"]] = d
            end
        end
    end
end

println("--- :julia: Running downstream tests for $(pkg_name) (group: $(group))")

mktempdir() do tmp_dir
    Pkg.activate(tmp_dir)

    withenv("JULIA_PKG_DEVDIR" => tmp_dir) do
        Pkg.develop(PackageSpec(; name=pkg_name))
    end

    pkg_dir = joinpath(tmp_dir, pkg_name)
    if !isdir(pkg_dir)
        if isdir(joinpath(tmp_dir, pkg_name * ".jl"))
            pkg_dir = joinpath(tmp_dir, pkg_name * ".jl")
        else
            candidates = filter(readdir(tmp_dir)) do d
                pkg_cand = joinpath(tmp_dir, d)
                return isdir(pkg_cand) && isfile(joinpath(pkg_cand, "Project.toml"))
            end
            if !isempty(candidates)
                pkg_dir = joinpath(tmp_dir, first(candidates))
            else
                error(
                    "Could not find cloned package directory for $(pkg_name) in $(tmp_dir)"
                )
            end
        end
    end

    Pkg.activate(pkg_dir)

    downstream_proj = TOML.parsefile(joinpath(pkg_dir, "Project.toml"))
    downstream_deps = Set{String}(keys(get(downstream_proj, "deps", Dict())))

    withenv("JULIA_PKG_PRECOMPILE_AUTO" => 0, "GROUP" => group, "BACKEND_GROUP" => group) do
        try
            direct_specs = PackageSpec[]
            if "Lux" in downstream_deps || !any(in(downstream_deps), keys(local_pkgs))
                println("+++ :julia: Developing local Lux from $(lux_dir)")
                push!(direct_specs, PackageSpec(; path=lux_dir))
            end
            for (p_name, p_path) in local_pkgs
                p_name == "Lux" && continue
                if p_name in downstream_deps
                    println("+++ :julia: Developing local $(p_name) from $(p_path)")
                    push!(direct_specs, PackageSpec(; path=p_path))
                end
            end
            !isempty(direct_specs) && Pkg.develop(direct_specs)

            println("+++ :julia: Updating downstream dependencies")
            Pkg.update()

            println("+++ :julia: Running tests")
            Pkg.test(;
                julia_args=`--code-coverage=user`,
                test_fn=() -> begin
                    sandbox_proj_file = Base.active_project()
                    sandbox_dir = dirname(sandbox_proj_file)
                    sandbox_proj = TOML.parsefile(sandbox_proj_file)
                    sandbox_names = Set{String}(keys(get(sandbox_proj, "deps", Dict())))
                    manifest_file = joinpath(sandbox_dir, "Manifest.toml")
                    if isfile(manifest_file)
                        manifest = TOML.parsefile(manifest_file)
                        union!(sandbox_names, keys(get(manifest, "deps", Dict())))
                    end
                    sandbox_specs = [
                        PackageSpec(; path=p_path) for
                        (p_name, p_path) in local_pkgs if p_name in sandbox_names
                    ]
                    !isempty(sandbox_specs) && Pkg.develop(sandbox_specs)
                end,
            )
        catch err
            err isa Pkg.Resolve.ResolverError || rethrow()
            @info "Not compatible with this release. No problem." exception = err
            return nothing
        end
    end
end

println("+++ :julia: Finished Downstream Test")
