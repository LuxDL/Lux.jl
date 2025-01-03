using Comonicon, BenchmarkTools
using Lux, Enzyme, Reactant, Random

Reactant.set_default_backend("gpu")

include("resnet.jl")

Comonicon.@main function main(;
        optimize::String="all", batch_size::Vector{Int}=[1, 4, 32, 128],
        model_size::Int=50
)
    dev = reactant_device(; force=true)

    model = ResNet(model_size)
    ps, st = Lux.setup(Random.default_rng(), model) |> dev

    println("Param count: $(Lux.parameterlength(ps))")
    println("State count: $(Lux.statelength(st))")

    timings = Dict{Int, Float64}()

    for b in batch_size
        println("batch_size=$b")

        x = rand(Float32, 224, 224, 3, b) |> dev

        model_compiled = Reactant.compile(
            model, (x, ps, Lux.testmode(st)); sync=true, optimize=Symbol(optimize)
        )

        timings[b] = @belapsed $(model_compiled)($(x), $(ps), $(Lux.testmode(st)))

        println("Best timing: $(timings[b]) s")
    end

    println(timings)
end
