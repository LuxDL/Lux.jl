
<a id='Utilities'></a>

# Utilities




<a id='Index'></a>

## Index

- [`Lux.StatefulLuxLayer`](#Lux.StatefulLuxLayer)
- [`Lux.cpu`](#Lux.cpu)
- [`Lux.disable_stacktrace_truncation!`](#Lux.disable_stacktrace_truncation!)
- [`Lux.f16`](#Lux.f16)
- [`Lux.f32`](#Lux.f32)
- [`Lux.f64`](#Lux.f64)
- [`Lux.foldl_init`](#Lux.foldl_init)
- [`Lux.gpu`](#Lux.gpu)
- [`Lux.istraining`](#Lux.istraining)
- [`Lux.multigate`](#Lux.multigate)


<a id='Device-Management-/-Data-Transfer'></a>

## Device Management / Data Transfer

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.cpu' href='#Lux.cpu'>#</a>&nbsp;<b><u>Lux.cpu</u></b> &mdash; <i>Function</i>.



```julia
cpu(x)
```

Transfer `x` to CPU.

::: warning

This function has been deprecated. Use [`cpu_device`](../Accelerator_Support/LuxDeviceUtils#LuxDeviceUtils.cpu_device) instead.

:::


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/111e38cf567b32e7b94bc62f0907ebd41f86905a/src/deprecated.jl#L2-L12' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.gpu' href='#Lux.gpu'>#</a>&nbsp;<b><u>Lux.gpu</u></b> &mdash; <i>Function</i>.



```julia
gpu(x)
```

Transfer `x` to GPU determined by the backend set using [`Lux.gpu_backend!`](../Accelerator_Support/LuxDeviceUtils#LuxDeviceUtils.gpu_backend!).

:::warning

This function has been deprecated. Use [`gpu_device`](../Accelerator_Support/LuxDeviceUtils#LuxDeviceUtils.gpu_device) instead. Using this function inside performance critical code will cause massive slowdowns due to type inference failure.

:::


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/111e38cf567b32e7b94bc62f0907ebd41f86905a/src/deprecated.jl#L19-L30' class='documenter-source'>source</a><br>

</div>
<br>

:::warning


For detailed API documentation on Data Transfer check out the [LuxDeviceUtils.jl](../Building_Blocks/WeightInitializers#WeightInitializers-API)


:::


<a id='Weight-Initialization'></a>

## Weight Initialization


:::warning


For API documentation on Initialization check out the [WeightInitializers.jl](../Building_Blocks/WeightInitializers#WeightInitializers-API)


:::


<a id='Miscellaneous-Utilities'></a>

## Miscellaneous Utilities

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.foldl_init' href='#Lux.foldl_init'>#</a>&nbsp;<b><u>Lux.foldl_init</u></b> &mdash; <i>Function</i>.



```julia
foldl_init(op, x)
foldl_init(op, x, init)
```

Exactly same as `foldl(op, x; init)` in the forward pass. But, gives gradients wrt `init` in the backward pass.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/111e38cf567b32e7b94bc62f0907ebd41f86905a/src/utils.jl#L167-L173' class='documenter-source'>source</a><br>

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


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/111e38cf567b32e7b94bc62f0907ebd41f86905a/src/utils.jl#L2-L10' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.multigate' href='#Lux.multigate'>#</a>&nbsp;<b><u>Lux.multigate</u></b> &mdash; <i>Function</i>.



```julia
multigate(x::AbstractArray, ::Val{N})
```

Split up `x` into `N` equally sized chunks (along dimension `1`).


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/111e38cf567b32e7b94bc62f0907ebd41f86905a/src/utils.jl#L68-L72' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Updating-Floating-Point-Precision'></a>

## Updating Floating Point Precision


By default, Lux uses Float32 for all parameters and states. To update the precision simply pass the parameters / states / arrays into one of the following functions.

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.f16' href='#Lux.f16'>#</a>&nbsp;<b><u>Lux.f16</u></b> &mdash; <i>Function</i>.



```julia
f16(m)
```

Converts the `eltype` of `m` *floating point* values to `Float16`. Recurses into structs marked with `Functors.@functor`.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/111e38cf567b32e7b94bc62f0907ebd41f86905a/src/utils.jl#L205-L210' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.f32' href='#Lux.f32'>#</a>&nbsp;<b><u>Lux.f32</u></b> &mdash; <i>Function</i>.



```julia
f32(m)
```

Converts the `eltype` of `m` *floating point* values to `Float32`. Recurses into structs marked with `Functors.@functor`.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/111e38cf567b32e7b94bc62f0907ebd41f86905a/src/utils.jl#L205-L210' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.f64' href='#Lux.f64'>#</a>&nbsp;<b><u>Lux.f64</u></b> &mdash; <i>Function</i>.



```julia
f64(m)
```

Converts the `eltype` of `m` *floating point* values to `Float64`. Recurses into structs marked with `Functors.@functor`.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/111e38cf567b32e7b94bc62f0907ebd41f86905a/src/utils.jl#L205-L210' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Stateful-Layer'></a>

## Stateful Layer

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.StatefulLuxLayer' href='#Lux.StatefulLuxLayer'>#</a>&nbsp;<b><u>Lux.StatefulLuxLayer</u></b> &mdash; <i>Type</i>.



```julia
StatefulLuxLayer(model, ps, st; st_fixed_type = Val(true))
```

::: warning

This is not a Lux.AbstractExplicitLayer

:::

A convenience wrapper over Lux layers which stores the parameters and states internally. Most users should not be using this version. This comes handy when Lux internally uses the `@compact` to construct models and in SciML codebases where propagating state might involving [`Box`ing](https://github.com/JuliaLang/julia/issues/15276).

For a motivating example, see the Neural ODE tutorial.

**Arguments**

  * `model`: A Lux layer
  * `ps`: The parameters of the layer. This can be set to `nothing`, if the user provides the parameters on function call
  * `st`: The state of the layer

**Keyword Arguments**

  * `st_fixed_type`: If `Val(true)`, then the type of the `state` is fixed, i.e., `typeof(last(model(x, ps, st))) == st`. If this is not the case, then `st_fixed_type` must be set to `Val(false)`. If `st_fixed_type` is set to `Val(false)`, then type stability is not guaranteed.

**Inputs**

  * `x`: The input to the layer
  * `ps`: The parameters of the layer. Optional, defaults to `s.ps`

**Outputs**

  * `y`: The output of the layer


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/111e38cf567b32e7b94bc62f0907ebd41f86905a/src/helpers/stateful.jl#L1-L39' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Truncated-Stacktraces'></a>

## Truncated Stacktraces

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.disable_stacktrace_truncation!' href='#Lux.disable_stacktrace_truncation!'>#</a>&nbsp;<b><u>Lux.disable_stacktrace_truncation!</u></b> &mdash; <i>Function</i>.



```julia
disable_stacktrace_truncation!(; disable::Bool=true)
```

An easy way to update `TruncatedStacktraces.VERBOSE` without having to load it manually.

Effectively does `TruncatedStacktraces.VERBOSE[] = disable`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/111e38cf567b32e7b94bc62f0907ebd41f86905a/src/stacktraces.jl#L1-L7' class='documenter-source'>source</a><br>

</div>
<br>
