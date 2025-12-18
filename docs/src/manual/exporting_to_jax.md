# [Exporting Lux Models to Jax (via EnzymeJAX & Reactant)](@id exporting_to_stablehlo)

In this manual, we will go over how to export Lux models to StableHLO and use
[EnzymeJAX](https://github.com/EnzymeAD/Enzyme-JAX) to run integrate Lux models with
JAX. We assume that users are familiar with
[Reactant compilation of Lux models](@ref reactant-compilation).

```@example exporting_to_stablehlo
using Lux, Reactant, Random, NPZ

const dev = reactant_device()
```

We simply define a Lux model and parameters.

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

Now, we can use `Reactant.Serialization.export_to_enzymejax` to generate the necessary files to run the model in JAX. This function will generate a Python script, an MLIR file, and a `.npz` file with the inputs.

```@example exporting_to_stablehlo
lux_model_func(x, ps, st) = model(x, ps, st)
# It's recommended to create a temporary directory for the exported files
output_dir = mktempdir()
println("Exported files will be in: ", output_dir)
py_script_path = Reactant.Serialization.export_to_enzymejax(
    lux_model_func, x, ps, st; function_name="lux_model", output_dir=output_dir)
println("Python script generated at: ", py_script_path)
nothing # hide
```

The generated files will be in the `output_dir`. You can now run the model in Python.

```python
# This is a sample python script. The actual generated script can be run directly.
# Assuming you are in the output directory you can run `python lux_model.py`.
#
# To integrate into your own python code, you can do the following.
# Make sure the output directory is in your python path.
import jax
import numpy as np
from lux_model import run_lux_model
import os
import glob

# Find the inputs file
# The format is {function_name}_{id}_inputs.npz
input_files = glob.glob(os.path.join(".", "lux_model_*_inputs.npz"))
assert len(input_files) == 1, "Expected to find exactly one inputs file"

# Load the inputs from the .npz file
inputs = np.load(input_files[0])
inputs = [inputs[f] for f in inputs.files]

# Run the exported Lux model
result = run_lux_model(*inputs)
print(result)

# The function can be JIT-compiled
jitted_model = jax.jit(run_lux_model)
result_jitted = jitted_model(*inputs)
print(result_jitted)
```
