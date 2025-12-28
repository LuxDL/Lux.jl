# [Exporting Lux Models to Jax (via EnzymeJAX & Reactant)](@id exporting_to_stablehlo)

In this manual, we will go over how to export Lux models to StableHLO and use
[EnzymeJAX](https://github.com/EnzymeAD/Enzyme-JAX) to run integrate Lux models with
JAX. We assume that users are familiar with
[Reactant compilation of Lux models](@ref reactant-compilation).

```@example exporting_to_stablehlo
using Lux, Reactant, Random, NPZ

const dev = reactant_device()
```

We simply define a Lux model and generate the stablehlo code using `Reactant.@code_hlo`.

```@example exporting_to_stablehlo
model = Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    FlattenLayer(3),
    Chain(
        Dense(256 => 128, relu),
        Dense(128 => 84, relu),
        Dense(84 => 10)
    )
)
ps, st = Lux.setup(Random.default_rng(), model) |> dev
nothing # hide
```

Generate an example input.

```@example exporting_to_stablehlo
x = randn(Random.default_rng(), Float32, 28, 28, 1, 4) |> dev
nothing # hide
```

Now instead of compiling the model, we will use the
`Reactant.Serialization.export_to_enzymejax` function to export the model.

```@example exporting_to_stablehlo
python_file_path = Reactant.Serialization.export_to_enzymejax(
    model, x, ps, st; function_name="run_lux_model"
)
```

This will generate a python file that can be used to run the model using EnzymeJAX.

```@example exporting_to_stablehlo
println(read(open(python_file_path, "r"), String))
nothing # hide
```
