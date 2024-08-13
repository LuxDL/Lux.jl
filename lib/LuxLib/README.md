# LuxLib

[![GitHub Discussions](https://img.shields.io/github/discussions/LuxDL/Lux.jl?color=white&logo=github&label=Discussions)](https://github.com/LuxDL/Lux.jl/discussions)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://lux.csail.mit.edu/dev/api/Building_Blocks/LuxLib)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://lux.csail.mit.edu/stable/api/Building_Blocks/LuxLib)

[![CI](https://github.com/LuxDL/LuxLib.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/LuxLib.jl/actions/workflows/CI.yml)
[![Buildkite](https://img.shields.io/buildkite/650bceb9ffcb044bee9c21e591728aaac2d8b57fae466e99cd/main?label=gpu)](https://buildkite.com/julialang/luxlib-dot-jl)
[![Benchmarks](https://github.com/LuxDL/LuxLib.jl/actions/workflows/Benchmark.yml/badge.svg)](https://luxdl.github.io/LuxLib.jl/benchmarks/)
[![codecov](https://codecov.io/gh/LuxDL/LuxLib.jl/branch/main/graph/badge.svg?token=1ZY0A2NPEM)](https://codecov.io/gh/LuxDL/LuxLib.jl)

[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FLuxLib&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/LuxLib)
[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FLuxLib&query=total_requests&&label=Total%20Downloads)](https://juliapkgstats.com/pkg/LuxLib)

[![JET Testing](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Backend for [Lux.jl](http://lux.csail.mit.edu/).

## Tutorials

This is a developer-facing project and most users **should not** depend on it directly. As
such, we don't have tutorials for this package. Instead, we recommend you check out the
[Lux tutorials](http://lux.csail.mit.edu/).

## What's the distinction from [NNlib.jl](https://github.com/FluxML/NNlib.jl)?

This is currently a place to hold more specialized kernels and layer implementations for
Lux.jl. Anyone is free to move these to NNlib.jl (this package is MIT licensed), but I
probably don't have the time to do so myself. But incase you do, open an issue here and let
me know I will delete the code from this package.

## Changelog

### Updating from v0.2 to v0.3

`groupnorm` with statistics tracking support has been removed.

### Updating from v0.1 to v0.2

Support for `CUDA` has been moved to a weak dependency. If you want to use `CUDA`, you need
to install and load `LuxCUDA` as `using LuxCUDA` or `import LuxCUDA`.
