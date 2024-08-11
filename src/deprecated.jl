# Deprecations for v1
"""
    cpu(x)

Transfer `x` to CPU.

!!! danger "Deprecation Notice"

    This function has been deprecated. Use [`cpu_device`](@ref) instead.
"""
function cpu end

@deprecate cpu(x) (MLDataDevices.cpu_device())(x)

"""
    gpu(x)

Transfer `x` to GPU determined by the backend set using [`Lux.gpu_backend!`](@ref).

!!! danger "Deprecation Notice"

    This function has been deprecated. Use [`gpu_device`](@ref) instead. Using this function
    inside performance critical code will cause massive slowdowns due to type inference
    failure.
"""
function gpu end

@deprecate gpu(x) (MLDataDevices.gpu_device())(x)

"""
    disable_stacktrace_truncation!(; disable::Bool=true)

An easy way to update `TruncatedStacktraces.VERBOSE` without having to load it manually.

Effectively does `TruncatedStacktraces.VERBOSE[] = disable`

!!! danger "Deprecation Notice"

    This function is now deprecated and will be removed in v1.
"""
function disable_stacktrace_truncation!(; disable::Bool=true)
    Base.depwarn(
        "`disable_stacktrace_truncation!` is not needed anymore, as stacktraces are \
        truncated by default. This function is now deprecated and will be removed in v1.",
        :disable_stacktrace_truncation)
    return
end

# Other deprecated functions
@deprecate xlogx(x::Number) LuxOps.xlogx(x)
@deprecate xlogy(x::Number, y::Number) LuxOps.xlogy(x, y)
@deprecate foldl_init(args...) LuxOps.foldl_init(args...)
@deprecate istraining(args...) LuxOps.istraining(args...)

# While the ones below aren't public, we ended up using them at quite a few places
@deprecate _getproperty(args...) LuxOps.getproperty(args...)
@deprecate _eachslice(args...) LuxOps.eachslice(args...)
