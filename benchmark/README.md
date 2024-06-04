# Lux.jl Continuous Benchmarking

Currently we use the BenchmarkTools.jl package to benchmark the performance of Lux.jl over
time.

This is built using https://github.com/benchmark-action/github-action-benchmark/ so it
allows for nice visualizations of the benchmark results in github pages and produces
warnings on PRs if the benchmark regress.

## Current Benchmarks

1. Small VGG Net for CIFAR-10
