# Lux.jl Continuous Benchmarking

Currently we use the BenchmarkTools.jl package to benchmark the performance of Lux.jl over
time.

This is built using https://github.com/benchmark-action/github-action-benchmark/ so it
allows for nice visualizations of the benchmark results in github pages and produces
warnings on PRs if the benchmarks regress.

## Current Benchmarks

TODO

## Roadmap for Continuous Benchmarking

- [ ] Generate a minimal set of benchmarks
  - [ ] Some basic ones are present in https://github.com/FluxML/Flux.jl/tree/master/perf
  - [ ] Benchmark different AD backends -- use `DifferentiationInterface.jl` for this part.
        Main backends are Zygote.jl, Tracker.jl, Enzyme.jl, Tapir.jl and ReverseDiff.jl.
- [ ] Comparative benchmarking against Flux.jl
- [ ] Migrate to Chairmarks.jl for benchmarking
- [ ] Migrate to Buildkite to allow for CUDA benchmarks
