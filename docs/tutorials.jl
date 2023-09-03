using Literate

preprocess(path, str) = replace(str, "__DIR = @__DIR__" => "__DIR = \"$(dirname(path))\"")

get_example_path(p) = joinpath(@__DIR__, "..", "examples", p)
OUTPUT = joinpath(@__DIR__, "src", "tutorials")

BEGINNER_TUTORIALS = ["Basics/main.jl", "PolynomialFitting/main.jl", "SimpleRNN/main.jl"]
INTERMEDIATE_TUTORIALS = ["NeuralODE/main.jl", "BayesianNN/main.jl", "HyperNet/main.jl"]
ADVANCED_TUTORIALS = ["GravitationalWaveForm/main.jl"]

withenv("JULIA_DEBUG" => "Literate") do
    # tasks = []
    for (d, paths) in (("beginner", BEGINNER_TUTORIALS),
            ("intermediate", INTERMEDIATE_TUTORIALS),
            ("advanced", ADVANCED_TUTORIALS)), (i, p) in enumerate(paths)
        name = "$(i)_$(first(rsplit(p, "/")))"
        p_ = get_example_path(p)
        # Literate.markdown(p_, joinpath(OUTPUT, d); execute=true, name=name, documenter=true,
        #     preprocess=Base.Fix1(preprocess, p_))
        # Using `@spawn` causes non-deterministic failures
        jl_expr = "using Literate; preprocess(path, str) = replace(str, \"__DIR = @__DIR__\" => \"__DIR = \\\"\$(dirname(path))\\\"\"); Literate.markdown(\"$(p_)\", \"$(joinpath(OUTPUT, d))\"; execute=true, name=\"$name\", documenter=true, preprocess=Base.Fix1(preprocess, \"$(p_)\"))"
        cm = `julia --project=$(@__DIR__) -e $(jl_expr)`
        run(cm)
        # task = Threads.@spawn run(cm)
        # push!(tasks, task)
    end
    # return wait.(tasks)
end
