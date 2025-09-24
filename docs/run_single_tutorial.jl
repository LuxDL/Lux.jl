using Pkg

storage_dir = joinpath(@__DIR__, "..", "tutorial_deps")
!isdir(storage_dir) && mkpath(storage_dir)

# Run this as `run_single_tutorial.jl <tutorial_name> <output_dir> <path/to/script>` <should_run>
name = ARGS[1]
output_directory = ARGS[2]
path = ARGS[3]
should_run = parse(Bool, ARGS[4])

project_path = dirname(Pkg.project().path)

pkg_log_path = joinpath(storage_dir, "$(name)_pkg.log")
push!(LOAD_PATH, "@literate")  # Should have the Literate and InteractiveUtils packages

const DRAFT_MODE = parse(Bool, get(ENV, "LUX_DOCS_DRAFT_BUILD", "false"))

io = open(pkg_log_path, "w")
warn_old_version = try
    should_run && Pkg.develop(; path=joinpath(@__DIR__, ".."), io)
    false
catch err
    err isa Pkg.Resolve.ResolverError || rethrow()
    @warn "Failed to install the latest version of Lux.jl. This is possible when the \
           downstream packages haven't been updated to support latest releases yet." err =
        err
    true
end
if should_run
    Pkg.instantiate(; io)
    Pkg.precompile(; io)
end
close(io)

using Literate

function preprocess_and_replace_includes(str)
    return replace(
        str,
        r"""include\("([^"]+)"\)""" =>
            s -> read(
                joinpath(project_path, match(r"""include\("([^"]+)"\)""", s).captures[1]),
                String,
            ),
    )
end

# Generate the script for users to download
assets_dir = joinpath(@__DIR__, "src", "public", "examples", name)
mkpath(assets_dir)

example_dir = dirname(path)
has_project_toml = false
for file in readdir(example_dir)
    if basename(file) == "Project.toml"
        global has_project_toml = true
        cp(joinpath(example_dir, file), joinpath(assets_dir, file); force=true)
    end
end
Literate.script(path, assets_dir; name, preprocess=preprocess_and_replace_includes)
Literate.notebook(
    path, assets_dir; name, execute=false, preprocess=preprocess_and_replace_includes
)

rel_path_to_assets = joinpath("..", "..", "public", "examples", name)

function preprocess(path, str)
    str = preprocess_and_replace_includes(str)

    if warn_old_version
        str =
            """
            # !!! danger "Using older version of Lux.jl"

            #     This tutorial cannot be run on the latest Lux.jl release due to downstream
            #     packages not being updated yet.

            \n\n""" * str
    end

    str = replace(str, "__DIR = @__DIR__" => "__DIR = \"$(dirname(path))\"")

    if !should_run
        str =
            """
            # !!! danger "Not Run on CI"

            #     This tutorial is not run on CI to reduce the computational burden. If you
            #     encounter any issues, please open an issue on the
            #     [Lux.jl](https://github.com/LuxDL/Lux.jl) repository.

            \n\n""" * str
    else
        str = str * """
        # ## Appendix

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        if @isdefined(MLDataDevices)
            if @isdefined(CUDA) && MLDataDevices.functional(CUDADevice)
                println()
                CUDA.versioninfo()
            end

            if @isdefined(AMDGPU) && MLDataDevices.functional(AMDGPUDevice)
                println()
                AMDGPU.versioninfo()
            end
        end

        nothing #hide
        """
    end

    # TODO: make vitepress compatible
    # script_download = """
    # # [![download julia script](https://img.shields.io/badge/download-$(name).jl-9558B2?logo=julia)]($(joinpath(rel_path_to_assets, "$(name).jl")))"""
    # project_toml_download = if has_project_toml
    #     """
    #     # [![download project toml](https://img.shields.io/badge/download-Project.toml-9C4221?logo=toml)]($(joinpath(rel_path_to_assets, "Project.toml")))"""
    # else
    #     ""
    # end
    # notebook_download = """
    # # [![download notebook](https://img.shields.io/badge/download-$(name).ipynb-FFB13B)]($(joinpath(rel_path_to_assets, "$(name).ipynb")))"""

    # str = """
    # $(script_download)
    # $(project_toml_download)
    # $(notebook_download)
    # \n\n""" * str

    return str
end

# For displaying generated Latex
function postprocess(path, str)
    return replace(
        str, "````\n__REPLACEME__\$" => "\$\$", "\$__REPLACEME__\n````" => "\$\$"
    )
end

Literate.markdown(
    path,
    output_directory;
    execute=should_run,
    name,
    flavor=if should_run || DRAFT_MODE
        Literate.DocumenterFlavor()
    else
        Literate.CommonMarkFlavor()
    end,
    preprocess=Base.Fix1(preprocess, path),
    postprocess=Base.Fix1(postprocess, path),
)
