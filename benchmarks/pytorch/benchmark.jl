using Lux, Metalhead, BenchmarkTools, Random, CUDA, Flux, Yota

model = EFL.transform(AlexNet().layers)

ps, st = EFL.setup(Random.default_rng(), model) .|> gpu;

# Numbers are from a V100

x = randn(MersenneTwister(0), Float32, 224, 224, 3, 32) |> gpu;

@benchmark CUDA.@sync $model($x, $ps, $st)


x = randn(MersenneTwister(0), Float32, 224, 224, 3, 1) |> gpu;

loss_function(p) = sum(model(x, p, st)[1])

@benchmark CUDA.@sync gradient($loss_function, $ps)




# BenchmarkTools.Trial: 105 samples with 1 evaluation.
#  Range (min … max):  43.343 ms … 99.216 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     44.498 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   47.737 ms ±  7.778 ms  ┊ GC (mean ± σ):  0.70% ± 2.51%

#    █                                                           
#   ▆██▆▃▃▂▁▃▁▁▁▃▁▂▁▁▁▁▃▂▁▃▂▁▂▃▃▂▁▂▃▁▁▁▂▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂ ▂
#   43.3 ms         Histogram: frequency by time        75.3 ms <

#  Memory estimate: 469.73 KiB, allocs estimate: 2715.

@benchmark CUDA.@sync grad($loss_function, $ps)
# BenchmarkTools.Trial: 105 samples with 1 evaluation.
#  Range (min … max):  40.542 ms … 105.025 ms  ┊ GC (min … max): 0.00% … 6.57%
#  Time  (median):     41.906 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   47.705 ms ±  10.329 ms  ┊ GC (mean ± σ):  1.09% ± 3.20%

#   ▆█▅                                                           
#   ████▆▅▆▅▁▁▁█▁▁▅▅▁▁▁▁▅▅▆█▅▅▁▁▅▅█▆▅▅▆█▅▁▅▁▁▁▁▅▁▁▅▁▁▁▁▅▁▁▁▁▅▁▁▅ ▅
#   40.5 ms       Histogram: log(frequency) by time      75.1 ms <

#  Memory estimate: 365.91 KiB, allocs estimate: 4087.



# Adjoints
d = EFL.Dense(2, 2; bias=false)
psd, std = EFL.setup(Random.default_rng(), d) .|> gpu
x = randn(Float32, 2, 1) |> gpu
@code_adjoint d(x, psd, std)
# Zygote.Adjoint(1: (%5, %6 :: Zygote.Context, %1, %2, %3, %4)
#   %7 = Zygote._pullback(%6, Zygote.literal_getproperty, %3, Val{:weight}())
#   %8 = Base.getindex(%7, 1)
#   %9 = Base.getindex(%7, 2)
#   %10 = Zygote._pullback(%6, Lux.fast_matmul, %8, %2)
#   %11 = Base.getindex(%10, 1)
#   %12 = Base.getindex(%10, 2)
#   %13 = Zygote._pullback(%6, Core.tuple, %11, %4)
#   %14 = Base.getindex(%13, 1)
#   %15 = Base.getindex(%13, 2)
#   return %14, 1: (%1)
#   %2 = (@15)(%1)
#   %3 = Zygote.gradindex(%2, 2)
#   %4 = Zygote.gradindex(%2, 3)
#   %5 = (@12)(%3)
#   %6 = Zygote.gradindex(%5, 2)
#   %7 = Zygote.gradindex(%5, 3)
#   %8 = (@9)(%6)
#   %9 = Zygote.gradindex(%8, 2)
#   %10 = Zygote.tuple(nothing, %7, %9, %4)
#   return %10)

y, back = pullback(d, x, psd, std)
@code_warntype back.back(y)
# MethodInstance for (::typeof(∂(λ)))(::Tuple{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, NamedTuple{(), Tuple{}}})
#     from (j::Zygote.Pullback{T})(Δ) where T in Zygote at compiler/interface2.jl:43
#   Static Parameters
#     T = Tuple{Lux.Dense{false, typeof(identity), typeof(Flux.glorot_uniform), typeof(Flux.zeros32)}, CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, NamedTuple{(:weight,), Tuple{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}}, NamedTuple{(), Tuple{}}}
#   Arguments
#     #self#::typeof(∂(λ))
#     Δ::Tuple{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, NamedTuple{(), Tuple{}}}
#   Body::Tuple{Nothing, CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, Union{Nothing, NamedTuple{(:weight,), Tuple{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}}}, NamedTuple{(), Tuple{}}}
#   1 ─       $(Expr(:meta, :inline))
#   │   %2  = Base.getfield(#self#, :t)::Tuple{typeof(∂(fast_matmul)), Zygote.var"#1630#back#159"{typeof(identity)}, Zygote.var"#1761#back#222"{Zygote.var"#back#221"{:weight, Zygote.Context, NamedTuple{(:weight,), Tuple{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}}, CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}}}
#   │   %3  = Base.getindex(%2, 3)::Zygote.var"#1761#back#222"{Zygote.var"#back#221"{:weight, Zygote.Context, NamedTuple{(:weight,), Tuple{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}}, CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}}
#   │   %4  = Base.getindex(%2, 2)::Core.Const(Zygote.var"#1630#back#159"{typeof(identity)}(identity))
#   │   %5  = Base.getindex(%2, 1)::typeof(∂(fast_matmul))
#   │   %6  = (%4)(Δ)::Tuple{Nothing, CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, NamedTuple{(), Tuple{}}}
#   │   %7  = Zygote.gradindex(%6, 2)::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
#   │   %8  = Zygote.gradindex(%6, 3)::Core.Const(NamedTuple())
#   │   %9  = (%5)(%7)::Tuple{Nothing, CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}
#   │   %10 = Zygote.gradindex(%9, 2)::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
#   │   %11 = Zygote.gradindex(%9, 3)::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
#   │   %12 = (%3)(%10)::Union{Nothing, Tuple{Nothing, NamedTuple{(:weight,), Tuple{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}}, Nothing}}
#   │   %13 = Zygote.gradindex(%12, 2)::Union{Nothing, NamedTuple{(:weight,), Tuple{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}}}
#   │   %14 = Zygote.tuple(nothing, %11, %13, %8)::Tuple{Nothing, CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, Union{Nothing, NamedTuple{(:weight,), Tuple{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}}}, NamedTuple{(), Tuple{}}}
#   └──       return %14