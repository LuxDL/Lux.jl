# Deprecations for v0.6
"""
    cpu(x)

Transfer `x` to CPU.

!!! danger

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

!!! danger

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

"""
    disable_stacktrace_truncation!(; disable::Bool=true)

An easy way to update `TruncatedStacktraces.VERBOSE` without having to load it manually.

Effectively does `TruncatedStacktraces.VERBOSE[] = disable`

!!! danger

    This function is now deprecated and will be removed in v0.6.
"""
function disable_stacktrace_truncation!(; disable::Bool=true)
    Base.depwarn("`disable_stacktrace_truncation!` is not needed anymore, as \
        stacktraces are truncated by default. This function is now deprecated and will be \
        removed in v0.6.",
        :disable_stacktrace_truncation)
    return
end
