using Lux, Flux, Random

make_immutable(x::AbstractArray) = ImmutableArray(copy(x))
make_immutable(x) = x

# Trial 1
model = Lux.Chain(
    Lux.BatchNorm(128),
    Lux.Dense(128, 256, tanh),
    Lux.BatchNorm(256),
    Lux.Chain(
        Lux.Dense(256, 1, tanh),
        Lux.Dense(1, 10)
    )
)

# Parameter and State Variables
ps, st = Lux.setup(MersenneTwister(0), model) .|> NamedTuple
ps_immutable = fmap(make_immutable, ps);
st_immutable = fmap(make_immutable, st);

# Dummy Input
x = randn(Float32, 128, 1024);
x_immutable = make_immutable(x);

# Run the model
@benchmark $model($x, $ps, $st)
@benchmark $model($x_immutable, $ps_immutable, $st_immutable)


# Trial 2
model = Lux.Dense(128, 256)

# Parameter and State Variables
ps, st = Lux.setup(MersenneTwister(0), model) .|> NamedTuple
ps_immutable = fmap(make_immutable, ps);
st_immutable = fmap(make_immutable, st);

# Run the model
@benchmark $model($x, $ps, $st)
@benchmark $model($x_immutable, $ps_immutable, $st_immutable)



# Trial 3
model = Lux.Dense(128, 256; bias=false)

# Parameter and State Variables
ps, st = Lux.setup(MersenneTwister(0), model) .|> NamedTuple
ps_immutable = fmap(make_immutable, ps);
st_immutable = fmap(make_immutable, st);

# Run the model
@benchmark $model($x, $ps, $st)
@benchmark $model($x_immutable, $ps_immutable, $st_immutable)



# julia> # Run the model
#        @benchmark $model($x, $ps, $st)
# BenchmarkTools.Trial: 7358 samples with 1 evaluation.
#  Range (min … max):  306.121 μs … 49.334 ms  ┊ GC (min … max): 0.00% … 6.06%
#  Time  (median):     412.842 μs              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   670.244 μs ±  1.303 ms  ┊ GC (mean ± σ):  3.75% ± 7.40%

#   █▇▆▆▄▃▃▂▂ ▁                                                  ▂
#   █████████████▇▇▇▇█▇▆▆▇▇▇▆▅▅▆▅▄▅▅▆▇▆▅▅▅▅▃▅▆▅▅▄▅▁▃▄▁▄▁▃▃▃▄▁▃▃▄ █
#   306 μs        Histogram: log(frequency) by time      5.23 ms <

#  Memory estimate: 1.00 MiB, allocs estimate: 2.

# julia> @benchmark $model($x_immutable, $ps_immutable, $st_immutable)
# BenchmarkTools.Trial: 5351 samples with 1 evaluation.
#  Range (min … max):  311.569 μs … 26.576 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     434.144 μs              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   919.238 μs ±  1.729 ms  ┊ GC (mean ± σ):  2.95% ± 7.45%

#   █▆▅▄▃▂▂▁▁▁   ▁                                               ▁
#   ██████████████▇█▇▇▇██▇▆▆▆▇▆▆▆▆▆▆▅▅▆▄▄▅▆▄▃▄▄▁▁▄▄▁▄▅▅▅▁▃▄▄▁▄▄▃ █
#   312 μs        Histogram: log(frequency) by time      8.66 ms <

#  Memory estimate: 1.00 MiB, allocs estimate: 5.


# Trial 4
model = Lux.Chain(
    Lux.Dense(128, 256; bias=false),
    Lux.Chain(
        Lux.Dense(256, 512; bias=false),
        Lux.Dense(512, 10; bias=false)
    )
)

# Parameter and State Variables
ps, st = Lux.setup(MersenneTwister(0), model) .|> NamedTuple
ps_immutable = fmap(make_immutable, ps);
st_immutable = fmap(make_immutable, st);

# Run the model
@benchmark $model($x, $ps, $st)
@benchmark $model($x_immutable, $ps_immutable, $st_immutable)


# julia> # Run the model
#        @benchmark $model($x, $ps, $st)
# BenchmarkTools.Trial: 1938 samples with 1 evaluation.
#  Range (min … max):  1.449 ms … 26.290 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     2.070 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   2.556 ms ±  1.763 ms  ┊ GC (mean ± σ):  3.14% ± 8.81%

#    █▄▁  ▁                                                     
#   ▇███▇▇█▆▄▄▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▁▂▂▂▁▂▁▁▂▁▁▂▂▂▁▁▂▂ ▃
#   1.45 ms        Histogram: frequency by time        10.4 ms <

#  Memory estimate: 3.04 MiB, allocs estimate: 6.

# julia> @benchmark $model($x_immutable, $ps_immutable, $st_immutable)
# BenchmarkTools.Trial: 1983 samples with 1 evaluation.
#  Range (min … max):  1.504 ms … 54.664 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     1.977 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   2.498 ms ±  1.957 ms  ┊ GC (mean ± σ):  3.18% ± 9.03%

#   ▁█▄▃▁                                                       
#   █████▆▆▅▄▄▃▃▃▃▃▃▃▃▃▃▃▃▂▂▂▂▃▂▂▂▂▂▂▂▂▂▂▁▂▁▂▂▂▁▁▂▁▂▂▂▂▂▂▁▂▂▂▂ ▃
#   1.5 ms         Histogram: frequency by time        8.55 ms <

#  Memory estimate: 3.04 MiB, allocs estimate: 18.