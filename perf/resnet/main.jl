using ArgParse, BenchmarkTools
import Metalhead
using Lux, Enzyme, Reactant, Random, Boltz

Reactant.set_default_backend("gpu")

function parse_commandline()
    s = ArgParseSettings()

    #! format: off
    @add_arg_table! s begin
        "--batch-size"
            help = "Batch size"
            arg_type = Vector{Int}
            default = [1, 4, 32, 128]

        "--model-size"
            help = "Model size"
            arg_type = Int
            default = 50
    end
    #! format: on

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    dev = xla_device()

    model = Vision.ResNet(parsed_args["model-size"])
    ps, st = Lux.setup(Random.default_rng(), model)
    ps_ra = ps |> dev
    st_ra = Lux.testmode(st) |> dev

    println("Param count: $(Lux.parameterlength(ps_ra))")
    println("State count: $(Lux.statelength(st_ra))")

    timings = Dict{Int, Float64}()

    for b in parsed_args["batch-size"]
        println("batch_size=$b")

        x = rand(Float32, 224, 224, 3, b) |> dev

        model_compiled = @compile model(x, ps_ra, st_ra)

        timings[b] = @belapsed begin
            y, _ = $(model_compiled)($(x), $(ps_ra), $(st_ra))
            Reactant.synchronize(y)
        end

        println("Best timing: $(timings[b]) s")
    end

    println(timings)
end

main()
