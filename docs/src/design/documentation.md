# Guide to Documentation for Lux.jl

## Documentation for Layers

The first line must be indented by 4 spaces and should contain the possible ways to construct the layer. This should be followed up with a description about what the layer does. If mathematical equations are needed to explain what the layer does, go for it. Often times we fuse parameters to make computation faster, this should be reflected in the equations being used, i.e. equations and the internal code must be consistent. (See [`LSTMCell`](@ref), [`GRUCell`](@ref) for some examples)

!!! note
    There is no need to document how the layers are being called since they **must** adhere to `layer(x, ps, st)`. Any deviation from that and the PR will not be accepted.

Next, we will have certain subsections (though all of them might not be necessary for all layers)

* **Arguments**: This section should be present unless the layer is constructed without any arguments (See [`NoOpLayer`](@ref)). All the arguments and their explicit constraints must be explained.
* **Inputs**: This section should all be present. List out the requirements `x` needs to satisfy.
* **Returns**: What will the layer return? We know the second element will be a state but is that updated in any form or not? 
* **Parameters**: What are the properties of the NamedTuple returned from `initialparameters`? Omit if the layer is parameterless
* **States**: What are the properties of the NamedTuple returned from `initialstates`? Omit if the layer is stateless