using BenchmarkTools, CUDA, Functors, Flux, ExplicitFluxLayers, Random

function run_benchmark_flux!(model, input_dims, lfn; model_name::String=string(typeof(model).name.name), btimes=Dict())
    if "Flux" ∉ keys(btimes)
        btimes["Flux"] = Dict()
    end
    merge!(
        btimes["Flux"],
        Dict(
            model_name => Dict(
                "Forward Pass CPU" => Dict(),
                "Forward + Backward Pass CPU" => Dict(),
                "Forward Pass GPU" => Dict(),
                "Forward + Backward Pass GPU" => Dict(),
            ),
        ),
    )

    println("Flux $model_name")

    # CPU Timings
    ps = Flux.params(model)
    for batchsize in 2 .^ (0:2:10)
        x = randn(Float32, input_dims..., batchsize)
        t = @belapsed $model($x)
        btimes["Flux"][model_name]["Forward Pass CPU"]["Batch Size = $batchsize"] = t
        println("    Forward Pass CPU | Batch Size = $batchsize: $(t)s")
        t = @belapsed gradient(() -> $lfn($model($x)), $ps)
        btimes["Flux"][model_name]["Forward + Backward Pass CPU"]["Batch Size = $batchsize"] = t
        println("    Forward Pass + Backward Pass CPU | Batch Size = $batchsize: $(t)s")
    end

    # GPU Timings
    if CUDA.functional()
        model = gpu(model)
        ps = Flux.params(model)
        for batchsize in 2 .^ (0:2:10)
            x = gpu(randn(Float32, input_dims..., batchsize))
            t = @belapsed CUDA.@sync $model($x)
            btimes["Flux"][model_name]["Forward Pass GPU"]["Batch Size = $batchsize"] = t
            println("    Forward Pass GPU | Batch Size = $batchsize: $(t)s")
            t = @belapsed CUDA.@sync gradient(() -> $lfn($model($x)), $ps)
            btimes["Flux"][model_name]["Forward + Backward Pass GPU"]["Batch Size = $batchsize"] = t
            println("    Forward Pass + Backward Pass GPU | Batch Size = $batchsize: $(t)s")
        end
    end

    return btimes
end

function run_benchmark_efl!(model, input_dims, lfn; model_name::String=string(typeof(model).name.name), btimes=Dict())
    if "ExplicitFluxLayers" ∉ keys(btimes)
        btimes["ExplicitFluxLayers"] = Dict()
    end
    merge!(
        btimes["ExplicitFluxLayers"],
        Dict(
            model_name => Dict(
                "Forward Pass CPU" => Dict(),
                "Forward + Backward Pass CPU" => Dict(),
                "Forward Pass GPU" => Dict(),
                "Forward + Backward Pass GPU" => Dict(),
            ),
        ),
    )

    println("ExplicitFluxLayers $model_name")

    # CPU Timings
    ps, st = ExplicitFluxLayers.setup(model)
    for batchsize in 2 .^ (0:2:10)
        x = randn(Float32, input_dims..., batchsize)
        t = @belapsed $model($x, $ps, $st)
        btimes["ExplicitFluxLayers"][model_name]["Forward Pass CPU"]["Batch Size = $batchsize"] = t
        println("    Forward Pass CPU | Batch Size = $batchsize: $(t)s")
        t = @belapsed gradient(p -> $lfn($model($x, p, $st)), $ps)
        btimes["ExplicitFluxLayers"][model_name]["Forward + Backward Pass CPU"]["Batch Size = $batchsize"] = t
        println("    Forward Pass + Backward Pass CPU | Batch Size = $batchsize: $(t)s")
    end

    # GPU Timings
    if CUDA.functional()
        ps = fmap(gpu, ps)
        st = fmap(gpu, st)
        for batchsize in 2 .^ (0:2:10)
            x = gpu(randn(Float32, input_dims..., batchsize))
            t = @belapsed CUDA.@sync $model($x, $ps, $st)
            btimes["ExplicitFluxLayers"][model_name]["Forward Pass GPU"]["Batch Size = $batchsize"] = t
            println("    Forward Pass GPU | Batch Size = $batchsize: $(t)s")
            t = @belapsed CUDA.@sync gradient(p -> $lfn($model($x, p, $st)), $ps)
            btimes["ExplicitFluxLayers"][model_name]["Forward + Backward Pass GPU"]["Batch Size = $batchsize"] = t
            println("    Forward Pass + Backward Pass GPU | Batch Size = $batchsize: $(t)s")
        end
    end

    return btimes
end
