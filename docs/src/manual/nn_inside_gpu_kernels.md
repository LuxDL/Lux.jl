# Neural Networks Inside GPU Kernels

In this page, we will describe how to embed neural networks inside GPU kernels. We will use
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) to do this,
making it compatible with multiple GPU backends.

!!! warning "Experimental Feature"

    This is a relatively new and experimental feature. Expect edge cases and open issues
    on GitHub if you find any.

!!! note "Inference Only"

    Currently this works only for inference. We will eventually test automatic
    differentiation using Enzyme.jl

!!! danger "Batching"

    In most usecases, this form of batching via embedding the neural network inside a GPU
    kernel is not recommended and will lead to suboptimal performance. Instead, batch the
    input data and let Lux handle the batching internally.

```@example nn_in_gpu_kernels
using Lux, LuxCUDA, Random, Functors
using KernelAbstractions, StaticArrays
```

First thing to remember is that we can't use regular high-level operations inside the
kernels, instead we will use Static Arrays. Leveraging Julia's multiple dispatch Lux will
use specialized operations that are compatible with GPU kernels.

```@example nn_in_gpu_kernels
@kernel function nn_eval_single_batch!(output, model, input, ps, st)
    i = @index(Global, Linear)
    y, st_ = Lux.apply(model, input[i], ps, st)
    output[i] = y
end
```

We define and initialize the neural network as usual, but we need to additionally convert
the Arrays into SArrays.

```@example nn_in_gpu_kernels
nn = Chain(Dense(4, 4, relu), Dense(4, 4))
ps, st = Lux.setup(Xoshiro(123), nn)

to_sarray(x) = SArray{Tuple{size(x)...}}(x)
ps_static = fmap(to_sarray, ps)
st_static = fmap(to_sarray, st)
```

First we will run it on CPU.

!!! warning

    Currently due to a minor bug, we cannot call the Lux models with vector input. As a
    workaround we make them into Matrix with batch size 1.

```@example nn_in_gpu_kernels
input = [@SArray(rand(Float64, 4, 1)) for i in 1:1024]
output = [@SArray(zeros(Float64, 4, 1)) for i in 1:1024] # Allocate the output
```

Now run the model using KernelAbstractions.jl

```@example nn_in_gpu_kernels
backend = KernelAbstractions.get_backend(output)
cpu_kernel! = nn_eval_single_batch!(backend)
cpu_kernel!(output, nn, input, ps_static, st_static; ndrange=length(output))
KernelAbstractions.synchronize(backend)
output
```

Now we will run the same model on GPU.

```@example nn_in_gpu_kernels
gdev = gpu_device()

input_gpu = input |> gdev
output_gpu = [@SArray(zeros(Float64, 4, 1)) for i in 1:1024] |> gdev
```

```@example nn_in_gpu_kernels
backend = KernelAbstractions.get_backend(output_gpu)
gpu_kernel! = nn_eval_single_batch!(backend)
gpu_kernel!(output_gpu, nn, input_gpu, ps_static, st_static; ndrange=length(output_gpu))
KernelAbstractions.synchronize(backend)
output_gpu
```
