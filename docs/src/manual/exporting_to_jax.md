# [Exporting Lux Models to Jax (via EnzymeJAX & Reactant)](@id exporting_to_stablehlo)

In this manual, we will go over how to export Lux models to StableHLO and use
[EnzymeJAX](https://github.com/EnzymeAD/Enzyme-JAX) to run integrate Lux models with
JAX. We assume that users are familiar with
[Reactant compilation of Lux models](@ref reactant-compilation).

```@example exporting_to_stablehlo
using Lux, Reactant, Random

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

Now instead of compiling the model, we will use `Reactant.@code_hlo` to generate the
StableHLO code.

```@example exporting_to_stablehlo
hlo_code = @code_hlo model(x, ps, st)
```

Now we just save this into an `mlir` file.

```@example exporting_to_stablehlo
write("exported_lux_model.mlir", string(hlo_code))
nothing # hide
```

Now we define a python script to run the model using EnzymeJAX.

```python
from enzyme_ad.jax import hlo_call

import jax
import jax.numpy as jnp

with open("exported_lux_model.mlir", "r") as file:
    code = file.read()


def run_lux_model(
    x,
    weight1,
    bias1,
    weight3,
    bias3,
    weight6_1,
    bias6_1,
    weight6_2,
    bias6_2,
    weight6_3,
    bias6_3,
):
    return hlo_call(
        x,
        weight1,
        bias1,
        weight3,
        bias3,
        weight6_1,
        bias6_1,
        weight6_2,
        bias6_2,
        weight6_3,
        bias6_3,
        source=code,
    )


# Note that all the inputs must be transposed, i.e. if the julia function has an input of
# shape (28, 28, 1, 4), then the input to the exported function called from python must be
# of shape (4, 1, 28, 28). This is because multi-dimensional arrays in Julia are stored in
# column-major order, while in JAX/Python they are stored in row-major order.

# Input as defined in our exported Lux model
x = jax.random.normal(jax.random.PRNGKey(0), (4, 1, 28, 28))

# Weights and biases corresponding to `ps` and `st` in our exported Lux model
weight1 = jax.random.normal(jax.random.PRNGKey(0), (6, 1, 5, 5))
bias1 = jax.random.normal(jax.random.PRNGKey(0), (6,))
weight3 = jax.random.normal(jax.random.PRNGKey(0), (16, 6, 5, 5))
bias3 = jax.random.normal(jax.random.PRNGKey(0), (16,))
weight6_1 = jax.random.normal(jax.random.PRNGKey(0), (256, 128))
bias6_1 = jax.random.normal(jax.random.PRNGKey(0), (128,))
weight6_2 = jax.random.normal(jax.random.PRNGKey(0), (128, 84))
bias6_2 = jax.random.normal(jax.random.PRNGKey(0), (84,))
weight6_3 = jax.random.normal(jax.random.PRNGKey(0), (84, 10))
bias6_3 = jax.random.normal(jax.random.PRNGKey(0), (10,))

# Run the exported Lux model
print(
    jax.jit(run_lux_model)(
        x,
        weight1,
        bias1,
        weight3,
        bias3,
        weight6_1,
        bias6_1,
        weight6_2,
        bias6_2,
        weight6_3,
        bias6_3,
    )
)
```
