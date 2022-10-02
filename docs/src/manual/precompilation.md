# Controlling Snoop Precompilation

Starting from `v0.4.27`, `Lux` uses
[SnoopPrecompile.jl](https://timholy.github.io/SnoopCompile.jl/dev/snoop_pc/) to precompile
certain use cases of `Lux`. This has significantly reduces Time to First Gradient (TTFG) for
Zygote. However, this increases the precompilation time by a lot.

## Benefits of Precompilation

Before showing how to disable precompilation, let's first make a case for snoop precompile
(after all it is enabled by default).

### With Snoop Precompilation Enabled

```julia-repl
julia> using Boltz, Lux, Zygote

julia> m, ps, st = resnet(:resnet18);

julia> x = randn(Float32, 224, 224, 3, 2);

julia> @time m(x, ps, st);
  3.437890 seconds (4.91 M allocations: 774.738 MiB, 2.67% gc time, 91.46% compilation time)

julia> @time Zygote.gradient(p -> sum(first(m(x, p, st))), ps);
  83.417719 seconds (80.84 M allocations: 5.780 GiB, 19.74% gc time, 98.17% compilation time: 1% of which was recompilation)
```

### With Snoop Precompilation Disabled

```julia-repl
julia> using Boltz, Lux, Zygote

julia> m, ps, st = resnet(:resnet18);

julia> x = randn(Float32, 224, 224, 3, 2);

julia> @time m(x, ps, st);
  5.471836 seconds (16.64 M allocations: 1.316 GiB, 6.75% gc time, 93.26% compilation time)

julia> @time Zygote.gradient(p -> sum(first(m(x, p, st))), ps);
  100.497466 seconds (95.42 M allocations: 6.467 GiB, 2.49% gc time, 99.05% compilation time)
```

## Disabling Precompilation

You can use [Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl) to control
the "amount" of precompilation. If you want to completely disable precompilation, you can
set `LuxSnoopPrecompile` to `false` in your `LocalPreferences.toml` file. This can be done
using the following command:

```julia
using Preferences, UUIDs

Preferences.@set_preferences!(UUID("b2108857-7c20-44ae-9111-449ecde12c47"),
                              "LuxSnoopPrecompile", false)
# Preferences.@set_preferences!(UUID("b2108857-7c20-44ae-9111-449ecde12c47"),
#                               "LuxPrecompileComponentArrays", false)
```

If `LuxSnoopPrecompile` is set to `false`, then `Lux` will not use `SnoopPrecompile.jl`:

```julia-repl
julia> @time_imports using Lux

    119.0 ms  Lux 6.41% compilation time
```

The other option is to just disable compilation of `ComponentArrays.jl` codepaths. This is
desirable if you are not planning to use Lux with any of the SciML Packages. This can be
done by setting `LuxPrecompileComponentArrays` to `false`:

```julia-repl
julia> @time_imports using Lux

    3366.4 ms  Lux 0.22% compilation time
```

If you have both the `LuxSnoopPrecompile` and `LuxPrecompileComponentArrays` set to `true`:

```julia-repl
julia> @time_imports using Lux

    5738.5 ms  Lux 0.13% compilation time
```
