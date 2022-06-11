# Contribution Guidelines

## Adding New Functionality/Layers

For Style we try to follow [SciMLStyle](https://github.com/SciML/SciMLStyle). The only reason we don't have a badge yet, is we haven't yet updated the package to followed all the guidelines. Here, I am documenting some additional guidelines we enforce:

### Mutability

See [SciMLStyle](https://github.com/SciML/SciMLStyle#out-of-place-and-immutability-is-preferred-when-sufficient-performant) for reference. This is strictly enforced, i.e. all layers/functions provided as part of the external API must be pure functions, even if they come with a performance penalty.

### Branching -- Generated Functions

Zygote doesn't like branches in code. Like it or not, we are stuck with it for the near future. Even if julia is able to optimize branches away, Zygote will most certainly throw away those optimizations (these can be tested via `Zygote.@code_ir`).

#### Writing efficient non-branching code to make Zygote happy

* Rely on `@generated` functions to remove **most** runtime branching. Certain examples:
  * Layers behaving differently during training and inference -- we know at compile-time whether a layer is being run in training/inference mode via `istraining(st)`.
  * Composite Layers relying on a variable number of internal layers -- Again we know the length of the number of internal layers at compile time. Hence we can manually unroll the loops. See [`Parallel`](@ref), [`Chain`](@ref), etc.
* Pass around `Val` in state. `Flux.jl` sets `training` to be `(:auto, true, false)`. Hence, which branch will be evaluated, will have to be determined at runtime time (*bad*). Instead if we pass `Val(true)`, we will be able to specialize functions directly based on `true`, `false`, etc. ensuring there is no runtime cost for these operations. See [`BatchNorm`](@ref), [`Dropout`](@ref), etc.


## Guide to Documentation for Lux.jl

### Documentation for Layers

The first line must be indented by 4 spaces and should contain the possible ways to construct the layer. This should be followed up with a description about what the layer does. If mathematical equations are needed to explain what the layer does, go for it. Often times we fuse parameters to make computation faster, this should be reflected in the equations being used, i.e. equations and the internal code must be consistent. (See [`LSTMCell`](@ref), [`GRUCell`](@ref) for some examples)

!!! note
    There is no need to document how the layers are being called since they **must** adhere to `layer(x, ps, st)`. Any deviation from that and the PR will not be accepted.

Next, we will have certain subsections (though all of them might not be necessary for all layers)

* **Arguments**: This section should be present unless the layer is constructed without any arguments (See [`NoOpLayer`](@ref)). All the arguments and their explicit constraints must be explained.
  * It is recommended to separate out the Keyword Arguments in their own section
* **Inputs**: This section should always be present. List out the requirements `x` needs to satisfy. (don't write about `ps` and `st` since that is expected by default)
* **Returns**: What will the layer return? We know the second element will be a state but is that updated in any form or not? 
* **Parameters**: What are the properties of the NamedTuple returned from `initialparameters`? Omit if the layer is parameterless
* **States**: What are the properties of the NamedTuple returned from `initialstates`? Omit if the layer is stateless

