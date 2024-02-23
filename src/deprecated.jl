# Deprecations for v0.6
"""
    cpu(x)

Transfer `x` to CPU.

!!! warning

    This function has been deprecated. Use [`cpu_device`](@ref) instead.
"""
function cpu(x)
    Base.depwarn("`cpu` has been deprecated and will be removed in v0.6. Use \
        `cpu_device` instead.", :cpu)
    return (cpu_device())(x)
end

"""
    gpu(x)

Transfer `x` to GPU determined by the backend set using [`Lux.gpu_backend!`](@ref).

!!! warning

    This function has been deprecated. Use [`gpu_device`](@ref) instead. Using this function
    inside performance critical code will cause massive slowdowns due to type inference
    failure.
"""
function gpu(x)
    @warn "Using `gpu` inside performance critical code will cause massive slowdowns due \
        to type inference failure. Please update your code to use `gpu_device` \
        API." maxlog=1

    Base.depwarn("`gpu` has been deprecated and will be removed in v0.6. Use \
        `gpu_device` instead.", :gpu)
    return (gpu_device())(x)
end
