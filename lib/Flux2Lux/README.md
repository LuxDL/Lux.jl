# Flux2Lux

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/lib/Flux2Lux/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/lib/Flux2Lux/)

[![CI](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml)
[![CI Nightly](https://github.com/avik-pal/Lux.jl/actions/workflows/CINightly.yml/badge.svg)](https://github.com/avik-pal/Lux.jl/actions/workflows/CINightly.yml)
[![codecov](https://codecov.io/gh/avik-pal/Lux.jl/branch/main/graph/badge.svg?token=IMqBM1e3hz)](https://codecov.io/gh/avik-pal/Lux.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Flux2Lux)](https://pkgs.genieframework.com?packages=Flux2Lux)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Flux2Lux is a package that allows you to convert Flux.jl models to Lux.jl.

## Difference from `Lux.transform`

`Lux.transform` has been deprecated in favor of `Flux2Lux.jl`. This package is a strict
superset of its predecessor. It provides additional features like `preserve_ps_st` and
`force_transform`. See the documentation of `Flux2Lux.transform` for more details.
