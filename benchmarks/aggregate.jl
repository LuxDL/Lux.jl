using BenchmarkTools

const GPU_BACKENDS = ["CUDA"]
const NUM_CPU_THREADS = [8, 128]

# Start with CPU benchmarks for 4 threads and add other results
const CPU_results_8threads_filepath = joinpath(
    @__DIR__, "results", "CPUbenchmarks8threads.json")
@assert(ispath(CPU_results_8threads_filepath))

const RESULTS = BenchmarkTools.load(CPU_results_8threads_filepath)[1]
@assert RESULTS isa BenchmarkTools.BenchmarkGroup

for benchmark in keys(RESULTS), mode in keys(RESULTS[benchmark])
    for end_key in keys(RESULTS[benchmark][mode])
        delete!(RESULTS[benchmark][mode], end_key)
    end
end

for n in NUM_CPU_THREADS
    filename = string("CPUbenchmarks", n, "threads.json")
    filepath = joinpath(@__DIR__, "results", filename)
    if !ispath(filepath)
        @warn "No file found at path: $(filepath)"
    else
        nthreads_results = BenchmarkTools.load(filepath)[1]
        if nthreads_results isa BenchmarkTools.BenchmarkGroup
            for benchmark in keys(RESULTS),
                mode in keys(RESULTS[benchmark]),
                end_tag in keys(nthreads_results[benchmark][mode])

                RESULTS[benchmark][mode]["CPU " * string(n, " ", "thread(s)")][end_tag] = nthreads_results[benchmark][mode][end_tag]
            end
        else
            @warn "Unexpected file format for file at path: $(filepath)"
        end
    end
end

for backend in GPU_BACKENDS
    filename = string(backend, "benchmarks.json")
    filepath = joinpath(@__DIR__, "results", filename)
    if !ispath(filepath)
        @warn "No file found at path: $(filepath)"
    else
        backend_results = BenchmarkTools.load(filepath)[1]
        if backend_results isa BenchmarkTools.BenchmarkGroup
            for benchmark in keys(RESULTS),
                mode in keys(RESULTS[benchmark]),
                end_tag in keys(backend_results[benchmark][mode])

                RESULTS[benchmark][mode][backend][end_tag] = backend_results[benchmark][mode][end_tag]
            end
        else
            @warn "Unexpected file format for file at path: $(filepath)"
        end
    end
end

BenchmarkTools.save(joinpath(@__DIR__, "results", "combinedbenchmarks.json"), RESULTS)
