# LuxCUDA

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/api/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/api/)

[![CI](https://github.com/LuxDL/LuxCUDA.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/LuxCUDA.jl/actions/workflows/CI.yml)
[![Buildkite NVIDIA GPU CI](https://img.shields.io/buildkite/7b7e33f865b82c14011f4e3dda13a7f32b10828d4c186bad41.svg?label=gpu&logo=nvidia)](https://buildkite.com/julialang/luxcuda-dot-jl/)
[![codecov](https://codecov.io/gh/LuxDL/LuxCUDA.jl/branch/main/graph/badge.svg?token=1ZY0A2NPEM)](https://codecov.io/gh/LuxDL/LuxCUDA.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/LuxCUDA)](https://pkgs.genieframework.com?packages=LuxCUDA)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

`LuxCUDA` is meant to be used as a trigger package for all CUDA dependencies in `Lux`.
Users requiring CUDA support should install `LuxCUDA` and load it alongside `Lux`.
