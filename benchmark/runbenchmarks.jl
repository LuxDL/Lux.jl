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

try
    PkgJogger.list_benchmarks(JogLux.BENCHMARK_DIR)
    JogLux.benchmark(; verbose=true, save=true, ref=:latest)
catch e
    JogLux.benchmark(; verbose=true, save=true)
end
