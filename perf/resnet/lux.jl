using Comonicon, BenchmarkTools
using Lux, LuxCUDA, Random

include("resnet.jl")

Comonicon.@main function main(;
        batch_size::Vector{Int} = [1, 4, 32, 128], model_size::Int = 50
    )
    dev = gpu_device(; force = true)

    model = ResNet(model_size)
    ps, st = Lux.setup(Random.default_rng(), model) |> dev

    println("Param count: $(Lux.parameterlength(ps))")
    println("State count: $(Lux.statelength(st))")

    timings = Dict{Int, Float64}()

    for b in batch_size
        println("batch_size=$b")

        x = rand(Float32, 224, 224, 3, b) |> dev

        timings[b] = @belapsed begin
            y, _ = $(model)($(x), $(ps), $(Lux.testmode(st)))
            CUDA.synchronize()
        end

        println("Best timing: $(timings[b]) s")
    end

    println(timings)
end

main()
