using Pkg

io = open(ENV["PKG_LOG_PATH"], "w")
Pkg.develop(; path=ENV["LUX_PATH"], io)
Pkg.instantiate(; io)
close(io)

using Literate

function preprocess(path, str)
    new_str = replace(str, "__DIR = @__DIR__" => "__DIR = \"$(dirname(path))\"")
    appendix_code = """
    # ## Appendix

    using InteractiveUtils
    InteractiveUtils.versioninfo()

    if @isdefined(LuxCUDA) && CUDA.functional()
        println()
        CUDA.versioninfo()
    end

    if @isdefined(LuxAMDGPU) && LuxAMDGPU.functional()
        println()
        AMDGPU.versioninfo()
    end

    nothing#hide
    """
    return new_str * appendix_code
end

# For displaying generated Latex
function postprocess(path, str)
    return replace(
        str, "````\n__REPLACEME__\$" => "\$\$", "\$__REPLACEME__\n````" => "\$\$")
end

Literate.markdown(ENV["EXAMPLE_PATH"], ENV["OUTPUT_DIRECTORY"]; execute=true,
    name=ENV["EXAMPLE_NAME"], flavor=Literate.DocumenterFlavor(),
    preprocess=Base.Fix1(preprocess, ENV["EXAMPLE_PATH"]),
    postprocess=Base.Fix1(postprocess, ENV["EXAMPLE_PATH"]))
