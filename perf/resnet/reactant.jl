using Comonicon, BenchmarkTools, JSON3
using Lux, Enzyme, Reactant, Random
using OrderedCollections

Reactant.set_default_backend("gpu")

include("resnet.jl")

function toy_loss_function(model, ps, st, x, y)
    return first(MSELoss()(model, ps, st, (x, y)))
end

Comonicon.@main function main(;
    optimize::String="all",
    batch_size::Vector{Int}=[1, 4, 32, 128],
    model_size::Vector{Int}=[18, 34, 50, 101],
)
    dev = reactant_device(; force=true)

    timings = OrderedDict{Int,OrderedDict{Int,OrderedDict{String,Float64}}}()

    for m in model_size
        println("model_size=$m")

        model = ResNet(m)
        ps, st = Lux.setup(Random.default_rng(), model) |> dev

        println("Param count: $(Lux.parameterlength(ps))")
        println("State count: $(Lux.statelength(st))")

        timings[m] = OrderedDict{Int,OrderedDict{String,Float64}}()

        for b in batch_size
            x = rand(Float32, 224, 224, 3, b) |> dev
            y = rand(Float32, 1000, b) |> dev

            time = Reactant.Profiler.profile_with_xprof(
                Lux.apply,
                model,
                x,
                ps,
                Lux.testmode(st);
                nrepeat=10,
                warmup=1,
                compile_options=Reactant.CompileOptions(;
                    optimization_passes=Symbol(optimize)
                ),
            )

            fwd_time = time.profiling_result.runtime_ns / 1e9

            if b == 1
                bwd_time = -1.0 # batchnorm cannot support batch size 1
            else
                time = Reactant.Profiler.profile_with_xprof(
                    Enzyme.gradient,
                    Reverse,
                    toy_loss_function,
                    Const(model),
                    ps,
                    Const(st),
                    Const(x),
                    Const(y);
                    nrepeat=10,
                    warmup=1,
                    compile_options=Reactant.CompileOptions(;
                        optimization_passes=Symbol(optimize)
                    ),
                )

                bwd_time = time.profiling_result.runtime_ns / 1e9
            end

            timings[m][b] = OrderedDict{String,Float64}(
                "forward" => fwd_time, "backward" => bwd_time
            )
        end

        display(timings[m])
    end

    results_path = joinpath(@__DIR__, "../results/resnet/")
    mkpath(results_path)

    open(joinpath(results_path, "reactant.json"), "w") do io
        JSON3.write(io, timings)
    end

    display(timings)
    return nothing
end
