# LuxLib

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/lib/LuxLib/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/lib/LuxLib/)

[![CI](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml)
[![CI Nightly](https://github.com/avik-pal/Lux.jl/actions/workflows/CINightly.yml/badge.svg)](https://github.com/avik-pal/Lux.jl/actions/workflows/CINightly.yml)
[![codecov](https://codecov.io/gh/avik-pal/Lux.jl/branch/main/graph/badge.svg?flag=LuxLib&token=IMqBM1e3hz)](https://codecov.io/gh/avik-pal/Lux.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/LuxLib)](https://pkgs.genieframework.com?packages=LuxLib)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)


Backend for [Lux.jl](http://lux.csail.mit.edu/stable).

## Tutorials

This is a developer-facing project and most users **should not** depend on it directly. As
such, we don't have tutorials for this package. Instead, we recommend you check out the
[Lux tutorials](http://lux.csail.mit.edu/stable/examples/examples/).

## What's the distinction from NNlib.jl?

Think of this package as a temporary location for functionalities that will move into
NNlib.jl. At the moment, this is supposed to be a heavier dependency than NNlib.jl, and
it makes no attempt to separate code across different architectures.
