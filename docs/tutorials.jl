using Distributed

addprocs(parse(Int, get(ENV, "LUX_DOCUMENTATION_NWORKERS", "1"));
    enable_threaded_blas=true, env=["JULIA_NUM_THREADS" => "$(Threads.nthreads())"])

@everywhere const LUX_DOCUMENTATION_NWORKERS = parse(
    Int, get(ENV, "LUX_DOCUMENTATION_NWORKERS", "1"))
@info "Lux Tutorial Build Running tutorials with $(LUX_DOCUMENTATION_NWORKERS) workers."
@everywhere const CUDA_MEMORY_LIMIT = 100 ÷ LUX_DOCUMENTATION_NWORKERS

@everywhere using Literate

@everywhere get_example_path(p) = joinpath(@__DIR__, "..", "examples", p)

BEGINNER_TUTORIALS = ["Basics/main.jl", "PolynomialFitting/main.jl",
    "SimpleRNN/main.jl", "SimpleChains/main.jl"]
INTERMEDIATE_TUTORIALS = ["NeuralODE/main.jl", "BayesianNN/main.jl", "HyperNet/main.jl"]
ADVANCED_TUTORIALS = ["GravitationalWaveForm/main.jl"]

TUTORIALS = [collect(enumerate(Iterators.product(["beginner"], BEGINNER_TUTORIALS)))...,
    collect(enumerate(Iterators.product(["intermediate"], INTERMEDIATE_TUTORIALS)))...,
    collect(enumerate(Iterators.product(["advanced"], ADVANCED_TUTORIALS)))...]

const storage_dir = joinpath(@__DIR__, "..", "tutorial_deps")
mkpath(storage_dir)

@info "Starting tutorial build"

try
    pmap(TUTORIALS) do (i, (d, p))
        println("Running tutorial $(i): $(p) on worker $(myid())")
        OUTPUT = joinpath(@__DIR__, "src", "tutorials")
        p_ = get_example_path(p)
        name = "$(i)_$(first(rsplit(p, "/")))"
        tutorial_proj = dirname(p_)
        pkg_log_path = joinpath(storage_dir, "$(name)_pkg.log")
        lux_path = joinpath(@__DIR__, "..")

        withenv("JULIA_DEBUG" => "Literate",
            "PKG_LOG_PATH" => pkg_log_path, "LUX_PATH" => lux_path,
            "JULIA_CUDA_HARD_MEMORY_LIMIT" => "$(CUDA_MEMORY_LIMIT)%",
            "OUTPUT_DIRECTORY" => joinpath(OUTPUT, d),
            "EXAMPLE_PATH" => p_, "EXAMPLE_NAME" => name,
            "JULIA_NUM_THREADS" => Threads.nthreads()) do
            cmd = `$(Base.julia_cmd()) --color=yes --project=$(tutorial_proj) -e \
                'using Pkg;
                    io=open(ENV["PKG_LOG_PATH"], "w");
                    Pkg.develop(; path=ENV["LUX_PATH"], io);
                    Pkg.instantiate(; io);
                    Pkg.precompile(; io);
                    eval(Meta.parse("using " * join(keys(Pkg.project().dependencies), ", ")));
                    close(io)'`
            @info "Running Command: $(cmd)"
            run(cmd)
            cmd = `$(Base.julia_cmd()) --color=yes --project=$(tutorial_proj) -e \
                'using Literate;
                    function preprocess(path, str)
                        new_str = replace(str, "__DIR = @__DIR__" => "__DIR = \"$(dirname(path))\"")
                        appendix_code = "\n# ## Appendix\nusing InteractiveUtils\nInteractiveUtils.versioninfo()\nif @isdefined(LuxCUDA) && CUDA.functional(); println(); CUDA.versioninfo(); end\nif @isdefined(LuxAMDGPU) && LuxAMDGPU.functional(); println(); AMDGPU.versioninfo(); end\nnothing#hide"
                        return new_str * appendix_code
                    end;
                    Literate.markdown(ENV["EXAMPLE_PATH"], ENV["OUTPUT_DIRECTORY"];
                        execute=false, name=ENV["EXAMPLE_NAME"],
                        flavor=Literate.DocumenterFlavor(),
                        preprocess=Base.Fix1(preprocess, ENV["EXAMPLE_PATH"]))'`
            @info "Running Command: $(cmd)"
            run(cmd)
            return
        end
    end
catch e
    rmprocs(workers()...)
    rethrow(e)
end
