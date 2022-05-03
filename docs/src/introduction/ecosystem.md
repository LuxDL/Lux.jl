# Seamless Composability

As you might have noticed we don't do much apart from Neural Networks. All other parts of the DL training/evaluation pipeline should be offloaded to the following frameworks:

* Data Manipulation/Loading -- [Augmentor.jl](https://evizero.github.io/Augmentor.jl/stable/), [DataLoaders.jl](https://lorenzoh.github.io/DataLoaders.jl/docs/dev/), [Images.jl](https://juliaimages.org/stable/)
* Optimisation -- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl), [ParameterSchedulers.jl](https://darsnack.github.io/ParameterSchedulers.jl/dev/README.html)
* Automatic Differentiation -- [Zygote.jl](https://github.com/FluxML/Zygote.jl)
* Parameter Manipulation -- [Functors.jl](https://fluxml.ai/Functors.jl/stable/)
* Model Checkpointing -- [Serialization.jl](https://docs.julialang.org/en/v1/stdlib/Serialization/)
* Activation Functions / Common Neural Network Primitives -- [NNlib.jl](https://fluxml.ai/Flux.jl/stable/models/nnlib/)
* Distributed Training -- [FluxMPI.jl](https://github.com/avik-pal/FluxMPI.jl)
* Training Visualization -- [Wandb.jl](https://github.com/avik-pal/Wandb.jl)
