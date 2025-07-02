"""
    export_as_tf_saved_model(
        model_dir::String,
        model::AbstractLuxLayer,
        x,
        ps,
        st;
        mode=:inference,
        force::Bool=false,
    )

Serializes a Lux model to a TensorFlow SavedModel format.

A SavedModel contains a complete TensorFlow program, including trained parameters (i.e,
tf.Variables) and computation. It does not require the original model building code to run,
which makes it useful for sharing or deploying with [TFLite](https://tensorflow.org/lite),
[TensorFlow.js](https://js.tensorflow.org/),
[TensorFlow Serving](https://www.tensorflow.org/tfx/serving/tutorials/Serving_REST_simple),
or [TensorFlow Hub](https://tensorflow.org/hub). Refer to the
[official documentation](https://www.tensorflow.org/guide/saved_model) for more details.

!!! warning "Load `Reactant.jl` and `PythonCall.jl` before using this function"

    This function requires the `Reactant` and `PythonCall` extensions to be loaded. If you
    haven't done so, please load them before calling this function.

!!! note "All inputs must be on `reactant_device()`"

    The inputs `x`, `ps`, and `st` must be on the device returned by `reactant_device()`. If
    you are using a GPU, ensure that the inputs are on the GPU device.

!!! danger "Running the saved model"

    Currently we don't support saving a dynamically shaped tensor. Hence, for inference the
    input must be the same shape as the one used during export.

!!! warning "Transposed Inputs"

    When providing inputs to the loaded model, ensure that the input tensors are transposed,
    i.e. if the inputs was `[S₁, S₂, ..., Sₙ]` during export, then the input to the loaded
    model should be `[Sₙ, ..., S₂, S₁]`.

## Arguments

  - `model_dir`: The directory where the model will be saved.
  - `model`: The model to be saved.
  - `x`: The input to the model.
  - `ps`: The parameters of the model.
  - `st`: The states of the model.

## Keyword Arguments

  - `mode`: The mode of the model. Can be either `:inference` or `:training`. Defaults to
    `:inference`. If set to `:training`, we will call [`LuxCore.trainmode`](@ref) on the
    model state, else we will call [`LuxCore.testmode`](@ref).
  - `force`: If `true`, the function will overwrite existing files in the specified
    directory. Defaults to `false`. If the directory is not empty and `force` is `false`,
    the function will throw an error.

## Example

Export the model to a TensorFlow SavedModel format.

```julia
using Lux, Reactant, PythonCall, Random

dev = reactant_device()

model = Chain(
    Conv((5, 5), 1 => 6, relu),
    BatchNorm(6),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    BatchNorm(16),
    MaxPool((2, 2)),
    FlattenLayer(3),
    Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)),
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model) |> dev;

x = rand(Float32, 28, 28, 1, 4) |> dev;

Lux.Serialization.export_as_tf_saved_model("/tmp/testing_tf_saved_model", model, x, ps, st)
```

Load the model and run inference on a random input tensor.

```python
import tensorflow as tf
import numpy as np

x_tf = tf.constant(np.random.rand(4, 1, 28, 28), dtype=tf.float32)

restored_model = tf.saved_model.load("/tmp/testing_tf_saved_model")
restored_model.f(x_tf)[0]
```
"""
function export_as_tf_saved_model(
    model_dir::String,
    model::AbstractLuxLayer,
    x,
    ps,
    st;
    mode=:inference,
    force::Bool=false,
)
    @assert mode in (:inference, :training) "mode must be either :inference or :training"

    @assert is_extension_loaded(Val(:Reactant)) "Reactant is not loaded. Please load it \
                                                 before using this function."

    mkpath(model_dir)
    if !isempty(readdir(model_dir))
        if force
            @warn "Directory $(model_dir) is not empty. This function might overwrite \
                   existing files."
        else
            throw(ArgumentError("Directory $(model_dir) is not empty. Pass `force=true` \
                                 to overwrite existing files."))
        end
    end

    export_as_tf_saved_model_internal(
        model_dir,
        model,
        x,
        ps,
        mode === :inference ? LuxCore.testmode(st) : LuxCore.trainmode(st),
    )
    return nothing
end

# defined in the Reactant extension
function export_as_tf_saved_model_internal end
