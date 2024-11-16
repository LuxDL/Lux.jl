using Pkg

storage_dir = joinpath(@__DIR__, "..", "tutorial_deps")
!isdir(storage_dir) && mkpath(storage_dir)

# Run this as `run_single_tutorial.jl <tutorial_name> <output_dir> <path/to/script>`
name = ARGS[1]
pkg_log_path = joinpath(storage_dir, "$(name)_pkg.log")
output_directory = ARGS[2]
path = ARGS[3]
push!(LOAD_PATH, "@literate")  # Should have the Literate and InteractiveUtils packages

io = open(pkg_log_path, "w")
warn_old_version = try
    Pkg.develop(; path=joinpath(@__DIR__, ".."), io)
    false
catch err
    err isa Pkg.Resolve.ResolverError || rethrow()
    @warn "Failed to install the latest version of Lux.jl. This is possible when the \
           downstream packages haven't been updated to support latest releases yet." err=err
    true
end
Pkg.instantiate(; io)
close(io)

using Literate

function preprocess(path, str)
    if warn_old_version
        str = """
        # !!! danger "Using older version of Lux.jl"

        #     This tutorial cannot be run on the latest Lux.jl release due to downstream
        #     packages not being updated yet.

        \n\n""" * str
    end
    new_str = replace(str, "__DIR = @__DIR__" => "__DIR = \"$(dirname(path))\"")
    appendix_code = """
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
    return new_str * appendix_code
end

# For displaying generated Latex
function postprocess(path, str)
    return replace(
        str, "````\n__REPLACEME__\$" => "\$\$", "\$__REPLACEME__\n````" => "\$\$")
end

Literate.markdown(
    path, output_directory; execute=true, name, flavor=Literate.DocumenterFlavor(),
    preprocess=Base.Fix1(preprocess, path), postprocess=Base.Fix1(postprocess, path)
)
