using ArgParse, BenchmarkTools
import Metalhead
using Lux, LuxCUDA, Random, Boltz

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
    dev = gpu_device(; force=true)

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

        timings[b] = @belapsed begin
            y, _ = $(model)($(x), $(ps_ra), $(st_ra))
            CUDA.synchronize()
        end

        println("Best timing: $(timings[b]) s")
    end

    println(timings)
end

main()
