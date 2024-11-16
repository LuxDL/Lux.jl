# Exporting Lux Models to Jax (via EnzymeJAX & Reactant)

!!! danger "Experimental"

    This feature is experimental and is subject to change without notice. Additionally,
    this feature currently requires some manual setup for interacting with Jax, which we are
    working on improving.

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
ps, st = Lux.setup(Random.default_rng(), model) |> dev;
```

Generate an example input.

```@example exporting_to_stablehlo
x = randn(Random.default_rng(), Float32, 28, 28, 1, 4) |> dev;
```

Now instead of compiling the model, we will use `Reactant.@code_hlo` to generate the
StableHLO code.

```@example exporting_to_stablehlo
hlo_code = @code_hlo model(x, ps, st)
```

Now we just save this into an `mlir` file.

```@example exporting_to_stablehlo
open("exported_lux_model.mlir", "w") do io
    write(io, string(hlo_code))
end
```

Now we define a python script to run the model using EnzymeJAX.

```python
from enzyme_ad.jax import primitives

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
    return primitives.ffi_call(
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
        out_shapes=[
            jax.core.ShapedArray([4, 10], jnp.float32),
        ],
        fn="main",
        source=code,
        lang=primitives.LANG_MHLO,
        pipeline_options=primitives.JaXPipeline(""),
    )


# Note that all the inputs must be transposed, i.e. if the Lux model has an input of shape
# (28, 28, 1, 4), then the input to the exported Lux model must be of shape (4, 1, 28, 28)
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
