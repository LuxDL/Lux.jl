
<a id='Utilities'></a>

# Utilities




<a id='Index'></a>

## Index

- [`Lux.cpu`](#Lux.cpu)
- [`Lux.disable_stacktrace_truncation!`](#Lux.disable_stacktrace_truncation!)
- [`Lux.foldl_init`](#Lux.foldl_init)
- [`Lux.gpu`](#Lux.gpu)
- [`Lux.istraining`](#Lux.istraining)
- [`Lux.multigate`](#Lux.multigate)
- [`Lux.replicate`](#Lux.replicate)


<a id='Device Management / Data Transfer'></a>

## Device Management / Data Transfer

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.cpu' href='#Lux.cpu'>#</a>&nbsp;<b><u>Lux.cpu</u></b> &mdash; <i>Function</i>.



```julia
cpu(x)
```

Transfer `x` to CPU.

::: warning

This function has been deprecated. Use [`cpu_device`](../LuxDeviceUtils/index#LuxDeviceUtils.cpu_device) instead.

:::


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/deprecated.jl#L2-L12' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.gpu' href='#Lux.gpu'>#</a>&nbsp;<b><u>Lux.gpu</u></b> &mdash; <i>Function</i>.



```julia
gpu(x)
```

Transfer `x` to GPU determined by the backend set using [`Lux.gpu_backend!`](../LuxDeviceUtils/index#LuxDeviceUtils.gpu_backend!).

:::warning

This function has been deprecated. Use [`gpu_device`](../LuxDeviceUtils/index#LuxDeviceUtils.gpu_device) instead. Using this function inside performance critical code will cause massive slowdowns due to type inference failure.

:::


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/deprecated.jl#L19-L30' class='documenter-source'>source</a><br>

</div>
<br>

:::warning


For detailed API documentation on Data Transfer check out the [LuxDeviceUtils.jl](../LuxDeviceUtils/)


:::


<a id='Weight Initialization'></a>

## Weight Initialization


:::warning


For API documentation on Initialization check out the [WeightInitializers.jl](../WeightInitializers/)


:::


<a id='Miscellaneous Utilities'></a>

## Miscellaneous Utilities

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.foldl_init' href='#Lux.foldl_init'>#</a>&nbsp;<b><u>Lux.foldl_init</u></b> &mdash; <i>Function</i>.



```julia
foldl_init(op, x)
foldl_init(op, x, init)
```

Exactly same as `foldl(op, x; init)` in the forward pass. But, gives gradients wrt `init` in the backward pass.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/utils.jl#L153-L159' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.istraining' href='#Lux.istraining'>#</a>&nbsp;<b><u>Lux.istraining</u></b> &mdash; <i>Function</i>.



```julia
istraining(::Val{training})
istraining(st::NamedTuple)
```

Returns `true` if `training` is `true` or if `st` contains a `training` field with value `true`. Else returns `false`.

Method undefined if `st.training` is not of type `Val`.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/utils.jl#L11-L19' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.multigate' href='#Lux.multigate'>#</a>&nbsp;<b><u>Lux.multigate</u></b> &mdash; <i>Function</i>.



```julia
multigate(x::AbstractArray, ::Val{N})
```

Split up `x` into `N` equally sized chunks (along dimension `1`).


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/utils.jl#L72-L76' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.replicate' href='#Lux.replicate'>#</a>&nbsp;<b><u>Lux.replicate</u></b> &mdash; <i>Function</i>.



```julia
replicate(rng::AbstractRNG)
replicate(rng::CUDA.RNG)
```

Creates a copy of the `rng` state depending on its type.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/utils.jl#L2-L7' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Truncated Stacktraces'></a>

## Truncated Stacktraces

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.disable_stacktrace_truncation!' href='#Lux.disable_stacktrace_truncation!'>#</a>&nbsp;<b><u>Lux.disable_stacktrace_truncation!</u></b> &mdash; <i>Function</i>.



```julia
disable_stacktrace_truncation!(; disable::Bool=true)
```

An easy way to update `TruncatedStacktraces.VERBOSE` without having to load it manually.

Effectively does `TruncatedStacktraces.VERBOSE[] = disable`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/stacktraces.jl#L1-L7' class='documenter-source'>source</a><br>

</div>
<br>
