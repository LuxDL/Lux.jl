# LuxTestUtils.jl

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://lux.csail.mit.edu/dev/api/Testing_Functionality/LuxTestUtils)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://lux.csail.mit.edu/stable/api/Testing_Functionality/LuxTestUtils)

[![CI](https://github.com/LuxDL/LuxTestUtils.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/LuxTestUtils.jl/actions/workflows/CI.yml)
[![Build status](https://img.shields.io/buildkite/e788fcafd7f48b654ded5b39d5ca119ee82f76274d2edb1bc9/main.svg?label=gpu&branch=master)](https://buildkite.com/julialang/luxtestutils-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Utilities for testing [Lux.jl](http://lux.csail.mit.edu/).

## Installation

```julia
] add LuxTestUtils
```

> **Warning**
> This is a testing package. Hence, we don't use features like weak dependencies to reduce
  load times. It is recommended that you exclusively use this package for testing and not
  add a dependency to it in your main package Project.toml.
