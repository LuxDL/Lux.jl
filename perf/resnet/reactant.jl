using Comonicon, BenchmarkTools, JSON3
using Lux, Enzyme, Reactant, Random

Reactant.set_default_backend("gpu")

include("resnet.jl")

function toy_loss_function(model, ps, st, x, y)
    return first(MSELoss()(model, ps, st, (x, y)))
end

Comonicon.@main function main(;
    optimize::String="all",
    batch_size::Vector{Int}=[1, 4, 32, 128],
    model_size::Vector{Int}=[18, 34, 50, 101, 152],
)
    dev = reactant_device(; force=true)

    timings = Dict{Int,Dict{Int,Dict{String,Float64}}}()

    for m in model_size
        println("model_size=$m")

        model = ResNet(m)
        ps, st = Lux.setup(Random.default_rng(), model) |> dev

        println("Param count: $(Lux.parameterlength(ps))")
        println("State count: $(Lux.statelength(st))")

        timings[m] = Dict{Int,Dict{String,Float64}}()

        for b in batch_size
            x = rand(Float32, 224, 224, 3, b) |> dev
            y = rand(Float32, 1000, b) |> dev

            model_compiled = Reactant.with_config(;
                dot_general_precision=PrecisionConfig.DEFAULT,
                convolution_precision=PrecisionConfig.DEFAULT,
            ) do
                Reactant.compile(
                    model,
                    (x, ps, Lux.testmode(st));
                    sync=true,
                    optimize=Symbol(optimize),
                )
            end

            fwd_time = @belapsed begin
                $(model_compiled)($(x), $(ps), $(Lux.testmode(st)))
            end setup = begin
                GC.gc(true)
            end

            if b == 1
                bwd_time = -1.0 # batchnorm cannot support batch size 1
            else
                grad_compiled = Reactant.with_config(;
                    dot_general_precision=PrecisionConfig.DEFAULT,
                    convolution_precision=PrecisionConfig.DEFAULT,
                ) do
                    Reactant.compile(
                        Enzyme.gradient,
                        (
                            Reverse,
                            toy_loss_function,
                            Const(model),
                            ps,
                            Const(st),
                            Const(x),
                            Const(y),
                        );
                        sync=true,
                        optimize=Symbol(optimize),
                    )
                end
                bwd_time = @belapsed $(grad_compiled)(
                    $Reverse,
                    $(toy_loss_function),
                    $(Const(model)),
                    $(ps),
                    $(Const(st)),
                    $(Const(x)),
                    $(Const(y)),
                ) setup = begin
                    GC.gc(true)
                end
            end

            timings[m][b] = Dict{String,Float64}(
                "forward" => fwd_time, "backward" => bwd_time
            )
        end

        println(timings[m])
    end

    results_path = joinpath(@__DIR__, "../results/resnet/")
    mkpath(results_path)

    open(joinpath(results_path, "reactant.json"), "w") do io
        JSON3.write(io, timings)
    end

    display(timings)
    return nothing
end
