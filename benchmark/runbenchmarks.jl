using PkgJogger, InteractiveUtils, LinearAlgebra, Lux, ThreadPinning, Printf
using LuxCUDA

BLAS.set_num_threads(Threads.nthreads() ÷ 2)
ThreadPinning.pinthreads(:cores)
ThreadPinning.threadinfo()

@info sprint(versioninfo)
if CUDA.functional()
    @info sprint(CUDA.versioninfo)
end
@info "BLAS threads: $(BLAS.get_num_threads())"

@jog Lux

# TODO: Judge and post to GitHub
try
    list = PkgJogger.list_benchmarks(joinpath(JogLux.BENCHMARK_DIR, "trial"))
    @info "Previous Benchmarks" list=list
    JogLux.benchmark(; verbose=true, save=true, ref=:latest)
catch e
    @warn "Failed to load previous benchmarks" exception=(e, catch_backtrace())
    JogLux.benchmark(; verbose=true, save=true)
end
