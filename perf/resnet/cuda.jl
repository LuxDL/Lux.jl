using Comonicon, BenchmarkTools, JSON3
using Lux, LuxCUDA, Random, Zygote
using OrderedCollections

include("resnet.jl")

function toy_loss_function(model, ps, st, x, y)
    return first(MSELoss()(model, ps, st, (x, y)))
end

Comonicon.@main function main(;
    batch_size::Vector{Int}=[1, 4, 32, 128],
    model_size::Vector{Int}=[18, 34, 50, 101], # 152 is too large for even for 40GB GPU
)
    dev = gpu_device(; force=true)

    timings = OrderedDict{Int,OrderedDict{Int,OrderedDict{String,Float64}}}

    for m in model_size
        println("model_size=$m")
        model = ResNet(m)
        ps, st = Lux.setup(Random.default_rng(), model) |> dev

        println("Param count: $(Lux.parameterlength(ps))")
        println("State count: $(Lux.statelength(st))")

        timings[m] = OrderedDict{Int,OrderedDict{String,Float64}}()

        for b in batch_size
            x = rand(Float32, 224, 224, 3, b) |> dev

            fwd_time = @belapsed begin
                y, _ = $(model)($(x), $(ps), $(Lux.testmode(st)))
                CUDA.synchronize()
            end setup = begin
                GC.gc(true)
                CUDA.reclaim()
            end

            y = rand(Float32, 1000, b) |> dev

            fn = (ps, x) -> toy_loss_function(model, ps, st, x, y)

            if b == 1
                bwd_time = -1.0 # batchnorm cannot support batch size 1
            else
                bwd_time = @belapsed begin
                    Zygote.gradient($(fn), $(ps), $(x))
                    CUDA.synchronize()
                end setup = begin
                    GC.gc(true)
                    CUDA.reclaim()
                end
            end

            timings[m][b] = OrderedDict{String,Float64}(
                "forward" => fwd_time, "backward" => bwd_time
            )
        end

        display(timings[m])
    end

    results_path = joinpath(@__DIR__, "../results/resnet/")
    mkpath(results_path)

    open(joinpath(results_path, "cudajl.json"), "w") do io
        JSON3.write(io, timings)
    end

    display(timings)
    return nothing
end

main()
