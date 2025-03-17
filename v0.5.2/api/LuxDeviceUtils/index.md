
<a id='LuxDeviceUtils'></a>

# LuxDeviceUtils


`LuxDeviceUtils.jl` is a lightweight package defining rules for transferring data across devices. Most users should directly use Lux.jl instead.




<a id='Index'></a>

## Index

- [`LuxDeviceUtils.cpu_device`](#LuxDeviceUtils.cpu_device)
- [`LuxDeviceUtils.gpu_backend!`](#LuxDeviceUtils.gpu_backend!)
- [`LuxDeviceUtils.gpu_device`](#LuxDeviceUtils.gpu_device)
- [`LuxDeviceUtils.reset_gpu_device!`](#LuxDeviceUtils.reset_gpu_device!)
- [`LuxDeviceUtils.supported_gpu_backends`](#LuxDeviceUtils.supported_gpu_backends)


<a id='Preferences'></a>

## Preferences

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxDeviceUtils.gpu_backend!' href='#LuxDeviceUtils.gpu_backend!'>#</a>&nbsp;<b><u>LuxDeviceUtils.gpu_backend!</u></b> &mdash; <i>Function</i>.



```julia
gpu_backend!() = gpu_backend!("")
gpu_backend!(backend) = gpu_backend!(string(backend))
gpu_backend!(backend::AbstractLuxGPUDevice)
gpu_backend!(backend::String)
```

Creates a `LocalPreferences.toml` file with the desired GPU backend.

If `backend == ""`, then the `gpu_backend` preference is deleted. Otherwise, `backend` is validated to be one of the possible backends and the preference is set to `backend`.

If a new backend is successfully set, then the Julia session must be restarted for the change to take effect.

</div>
<br>

<a id='Data Transfer'></a>

## Data Transfer

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxDeviceUtils.cpu_device' href='#LuxDeviceUtils.cpu_device'>#</a>&nbsp;<b><u>LuxDeviceUtils.cpu_device</u></b> &mdash; <i>Function</i>.



```julia
cpu_device() -> LuxCPUDevice()
```

Return a `LuxCPUDevice` object which can be used to transfer data to CPU.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxDeviceUtils.gpu_device' href='#LuxDeviceUtils.gpu_device'>#</a>&nbsp;<b><u>LuxDeviceUtils.gpu_device</u></b> &mdash; <i>Function</i>.



```julia
gpu_device(; force_gpu_usage::Bool=false) -> AbstractLuxDevice()
```

Selects GPU device based on the following criteria:

1. If `gpu_backend` preference is set and the backend is functional on the system, then that device is selected.
2. Otherwise, an automatic selection algorithm is used. We go over possible device backends in the order specified by `supported_gpu_backends()` and select the first functional backend.
3. If no GPU device is functional and  `force_gpu_usage` is `false`, then `cpu_device()` is invoked.
4. If nothing works, an error is thrown.

</div>
<br>

<a id='Miscellaneous'></a>

## Miscellaneous

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxDeviceUtils.reset_gpu_device!' href='#LuxDeviceUtils.reset_gpu_device!'>#</a>&nbsp;<b><u>LuxDeviceUtils.reset_gpu_device!</u></b> &mdash; <i>Function</i>.



```julia
reset_gpu_device!()
```

Resets the selected GPU device. This is useful when automatic GPU selection needs to be run again.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxDeviceUtils.supported_gpu_backends' href='#LuxDeviceUtils.supported_gpu_backends'>#</a>&nbsp;<b><u>LuxDeviceUtils.supported_gpu_backends</u></b> &mdash; <i>Function</i>.



```julia
supported_gpu_backends() -> Tuple{String, ...}
```

Return a tuple of supported GPU backends.

!!! warning
    This is not the list of functional backends on the system, but rather backends which `Lux.jl` supports.


!!! warning
    `Metal.jl` support is **extremely** experimental and most things are not expected to work.


</div>
<br>
