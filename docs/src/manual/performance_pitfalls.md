# Performance Pitfalls & How to Catch Them

## Spurious Type-Promotion

## Detecting Type-Instability

## Scalar Indexing on GPU Arrays

When running code on GPUs, it is recommended to
[disallow scalar indexing](https://cuda.juliagpu.org/stable/usage/workflow/#UsageWorkflowScalar).
Note that this is disabled by default except in REPL. You can disable it even in REPL mode
using:

```@example perf-pitfalls-scalar-indexing
using GPUArraysCore
GPUArraysCore.allowscalar(false)
```
