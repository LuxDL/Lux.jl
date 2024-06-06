# LuxDeviceUtils

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://lux.csail.mit.edu/dev/api/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://lux.csail.mit.edu/stable/api/)

[![CI](https://github.com/LuxDL/LuxDeviceUtils.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/LuxDeviceUtils.jl/actions/workflows/CI.yml)
[![Buildkite](https://badge.buildkite.com/b098d6387b2c69bd0ab684293ff66332047b219e1b8f9bb486.svg?branch=main)](https://buildkite.com/julialang/luxdeviceutils-dot-jl)
[![codecov](https://codecov.io/gh/LuxDL/LuxDeviceUtils.jl/branch/main/graph/badge.svg?token=1ZY0A2NPEM)](https://codecov.io/gh/LuxDL/LuxDeviceUtils.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

`LuxDeviceUtils.jl` is a lightweight package defining rules for transferring data across
devices. Most users should directly use [Lux.jl](https://lux.csail.mit.edu/) instead.

Currently we provide support for the following backends:

1. `CUDA.jl` for NVIDIA GPUs.
2. `AMDGPU.jl` for AMD ROCM GPUs.
3. `Metal.jl` for Apple Metal GPUs. **(Experimental)**
4. `oneAPI.jl` for Intel GPUs. **(Experimental)**
