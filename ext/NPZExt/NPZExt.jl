module NPZExt

import Lux
import NPZ
import Random

abstract type DataLoader end
abstract type PytorchLoader <: DataLoader end

struct PTLoader <: PytorchLoader end
@kwdef struct NPZLoader <: PytorchLoader
    dummy_field::Bool = true
end

# fallback
Lux.load(::DataLoader, args...) = error("DataLoader $(typeof(args[1])) not implemented.")

function Lux.load(N::NPZLoader, filepath::String)

    file = NPZ.npzread(filepath)

    # extract model architecture info from file
    # TODO don't hardcode
    layer_map = [
        "embed_layer"  => :layer_1,
        "layer_1"      => :layer_2,
        "layer_2"      => :layer_4,
        "layer_3"      => :layer_6,
        "output_layer" => :layer_7
    ]

    # construct Lux model architecture from layer map
    # TODO don't hardcode
    model = Chain(
        Dense(13 => 32, leakyrelu),
        Dense(32 => 64, leakyrelu),
        Dropout(0.2),
        Dense(64 => 64, leakyrelu),
        Dropout(0.1),
        Dense(64 => 32, leakyrelu),
        Dense(32 => 1)
    )

    # initialize model parameters, TODO zero initializer?
    parameters, states = Lux.setup(Random.default_rng(), model)

    # load weights from npz file into Lux model parameters
    for (py_name, lux_sym) in layer_map
        lux_layer_params = getproperty(parameters, lux_sym)
        
        lux_layer_params.weight .= Float32.(weights[py_name * ".weight"])
        lux_layer_params.bias   .= Float32.(weights[py_name * ".bias"])
    end

    return model, parameters, states
end

end # module