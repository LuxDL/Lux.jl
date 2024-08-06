using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @btime, @benchmarkable
using InteractiveUtils: versioninfo
using LinearAlgebra: BLAS
using Statistics: median
using Suppressor: @suppress
using ThreadPinning: pinthreads

pinthreads(:cores)

BLAS.set_num_threads(Threads.nthreads() ÷ 2)

@info sprint(versioninfo)
@info "BLAS threads: $(BLAS.get_num_threads())"

const SUITE = BenchmarkGroup()

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

# To run benchmarks on a specific GPU backend, add AMDGPU / CUDA / Metal / oneAPI
# to benchmarks/Project.toml and change BENCHMARK_GROUP to the backend name
const BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", "CPU")
const FORCE_BENCHMARK_TUNING = parse(Bool, get(ENV, "FORCE_BENCHMARK_TUNING", "false"))
const BENCHMARK_CPU_THREADS = Threads.nthreads()

# Number of CPU threads to benchmarks on
if BENCHMARK_CPU_THREADS > Threads.nthreads()
    @error "More CPU threads were requested than are available. Change the \
            JULIA_NUM_THREADS environment variable or pass \
            --threads=$(BENCHMARK_CPU_THREADS) as a julia argument"
end

if BENCHMARK_GROUP == "AMDGPU"
    using AMDGPU # ] add AMDGPU to benchmarks/Project.toml
    AMDGPU.versioninfo()
    @info "Running AMDGPU benchmarks" maxlog=1
elseif BENCHMARK_GROUP == "CUDA"
    using LuxCUDA # ] add LuxCUDA to benchmarks/Project.toml
    CUDA.versioninfo()
    @info "Running CUDA benchmarks" maxlog=1
else
    @info "Running CPU benchmarks with $(BENCHMARK_CPU_THREADS) thread(s)" maxlog=1
end

# Main benchmark files
include("setups/layers.jl")
include("setups/models.jl")

include("setup.jl")
setup_benchmarks(SUITE, BENCHMARK_GROUP, BENCHMARK_CPU_THREADS)

filepath = joinpath(@__DIR__, "results")
mkpath(filepath)
filename = BENCHMARK_GROUP == "CPU" ?
           string("CPUbenchmarks", BENCHMARK_CPU_THREADS, "threads.json") :
           string(BENCHMARK_GROUP, "benchmarks.json")
params_filename = BENCHMARK_GROUP == "CPU" ?
                  string("CPUbenchmarks", BENCHMARK_CPU_THREADS, "threads_params.json") :
                  string(BENCHMARK_GROUP, "benchmarks_params.json")

try
    params_file_url = "https://raw.githubusercontent.com/LuxDL/Lux.jl/gh-pages/benchmark-params/$(params_filename)"
    @info "Downloaded params file from $params_file_url"
    global download_path = download(params_file_url, params_filename)
catch err
    @error "Failed to download params file from \
            https://raw.githubusercontent.com/LuxDL/Lux.jl/gh-pages/benchmark-params/$(params_filename)" err
    global download_path = nothing
end

if download_path !== nothing && !FORCE_BENCHMARK_TUNING
    @info "Loading params from $download_path"
    _, has_all_groups = loadparams!(
        SUITE, BenchmarkTools.load(download_path)[1], :evals)
    if !has_all_groups
        @warn "Some groups did not have all benchmarks. Tuning benchmarks."
        BenchmarkTools.tune!(SUITE; verbose=true)
    end
else
    @info "Tuning benchmarks"
    BenchmarkTools.tune!(SUITE; verbose=true)
end
results = BenchmarkTools.run(SUITE; verbose=true)

BenchmarkTools.save(joinpath(filepath, params_filename), BenchmarkTools.params(SUITE))
@info "Saved params to $(joinpath(filepath, params_filename))"

BenchmarkTools.save(joinpath(filepath, filename), median(results))
@info "Saved results to $(joinpath(filepath, filename))"
