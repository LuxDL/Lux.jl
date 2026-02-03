module NPZExt

import Lux
import NPZ

abstract type DataLoader end
abstract type PytorchLoader <: DataLoader end


function Lux.load(::PytorchLoader, filepath::String)
    # load and interpret torch model here throw error/warning if format can't be interpreted
    # then return model, parameters, states to be used in Lux
    model = nothing
    parameters = nothing
    states = nothing
    return model, parameters, states
end

end # module