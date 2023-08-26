# LuxLib

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://lux.csail.mit.edu/dev/api/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://lux.csail.mit.edu/stable/api/)

[![CI](https://github.com/LuxDL/LuxLib.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/LuxLib.jl/actions/workflows/CI.yml)
[![Buildkite](https://img.shields.io/buildkite/650bceb9ffcb044bee9c21e591728aaac2d8b57fae466e99cd/main?label=gpu)](https://buildkite.com/julialang/luxlib-dot-jl)
[![codecov](https://codecov.io/gh/LuxDL/LuxLib.jl/branch/main/graph/badge.svg?token=1ZY0A2NPEM)](https://codecov.io/gh/LuxDL/LuxLib.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/LuxLib)](https://pkgs.genieframework.com?packages=LuxLib)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Backend for [Lux.jl](http://lux.csail.mit.edu/).

## Tutorials

This is a developer-facing project and most users **should not** depend on it directly. As
such, we don't have tutorials for this package. Instead, we recommend you check out the
[Lux tutorials](http://lux.csail.mit.edu/stable/).

## What's the distinction from NNlib.jl?

Think of this package as a temporary location for functionalities that will move into
NNlib.jl. At the moment, this is supposed to be a heavier dependency than NNlib.jl, and
it makes no attempt to separate code across different architectures.

## Changelog

### Updating from v0.2 to v0.3

`groupnorm` with statistics tracking support has been removed.

### Updating from v0.1 to v0.2

Support for `CUDA` has been moved to a weak dependency. If you want to use `CUDA`, you need
to install and load `LuxCUDA` as `using LuxCUDA` or `import LuxCUDA`.
