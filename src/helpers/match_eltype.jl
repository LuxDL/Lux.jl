"""
    match_eltype(layer, ps, st, args...)

Helper function to "maybe" (see below) match the element type of `args...` with the element
type of the layer's parameters and states. This is useful for debugging purposes, to track
down accidental type-promotions inside Lux layers.

# Extended Help

## Controlling the Behavior via Preferences

Behavior of this function is controlled via the  `eltype_mismatch_handling` preference. The
following options are supported:

  - `"none"`: This is the default behavior. In this case, this function is a no-op, i.e., it
    simply returns `args...`.
  - `"warn"`: This option will issue a warning if the element type of `args...` does not
    match the element type of the layer's parameters and states. The warning will contain
    information about the layer and the element type mismatch.
  - `"convert"`: This option is same as `"warn"`, but it will also convert the element type
    of `args...` to match the element type of the layer's parameters and states (for the
    cases listed below).
  - `"error"`: Same as `"warn"`, but instead of issuing a warning, it will throw an error.

!!! warning

    We print the warning for type-mismatch only once.

## Element Type Conversions

For `"convert"` only the following conversions are done:

| Element Type of parameters/states | Element Type of `args...` | Converted to |
|:--------------------------------- |:------------------------- |:------------ |
| `Float64`                         | `Integer`                 | `Float64`    |
| `Float32`                         | `Float64`                 | `Float32`    |
| `Float32`                         | `Integer`                 | `Float32`    |
| `Float16`                         | `Float64`                 | `Float16`    |
| `Float16`                         | `Float32`                 | `Float16`    |
| `Float16`                         | `Integer`                 | `Float16`    |
"""
function match_eltype end

@static if ELTYPE_MISMATCH_HANDLING == "none" # Just return the input
    @inline match_eltype(layer, ps, st, x) = x
    @inline function match_eltype(layer, ps, st, x, args...)
        return (x, args...)
    end
else
    @inline function match_eltype(layer, ps, st, x)
        fn = let elType = recursive_eltype((ps, st), Val(true)), layer = layer
            arr -> match_eltype(layer, elType, __eltype(arr), arr)
        end
        return recursive_map(fn, x)
    end
    @inline function match_eltype(layer, ps, st, x, args...)
        fn = let elType = recursive_eltype((ps, st), Val(true)), layer = layer
            arr -> match_eltype(layer, elType, __eltype(arr), arr)
        end
        return (recursive_map(fn, x), recursive_map(fn, args)...)
    end
end

for (T1, T2) in ((:Float64, :Integer), (:Float32, :Float64), (:Float32, :Integer),
    (:Float16, :Float64), (:Float16, :Float32), (:Float16, :Integer))
    warn_msg = "Layer with `$(T1)` parameters and states received `$(T2)` input."
    convert_msg = "Layer with `$(T1)` parameters and states received `$(T2)` input. \
                   Converting to `$(T1)`."
    error_msg = "Layer with `$(T1)` parameters and states received `$(T2)` input. This is \
                 not allowed because the preference `eltype_mismatch_handling` is set to \
                 `\"error\"`. To debug further, use `Lux.Experimental.@debug_mode`."
    @eval function match_eltype(layer, t1::Type{<:$(T1)}, ::Type{<:$(T2)}, x::AbstractArray)
        @static if ELTYPE_MISMATCH_HANDLING == "warn"
            __warn_mismatch(layer, x, $(warn_msg))
            return x
        elseif ELTYPE_MISMATCH_HANDLING == "convert"
            __warn_mismatch(layer, x, $(convert_msg))
            return __convert_eltype($(T1), x)
        elseif ELTYPE_MISMATCH_HANDLING == "error"
            throw(EltypeMismatchException($(error_msg)))
        end
    end
end

@inline match_eltype(_, ::Type, ::Type, x::AbstractArray) = x

@inline __convert_eltype(::Type{T}, x::AbstractArray) where {T} = broadcast(T, x)

function __warn_mismatch(layer, x, warn_msg)
    @warn warn_msg layer summary(x) maxlog=1
end
