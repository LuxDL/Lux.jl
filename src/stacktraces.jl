"""
    disable_stacktrace_truncation!(; disable::Bool=true)

An easy way to update `TruncatedStacktraces.VERBOSE` without having to load it manually.

Effectively does `TruncatedStacktraces.VERBOSE[] = disable`
"""
function disable_stacktrace_truncation!(; disable::Bool=true)
    @static if VERSION ≥ v"1.10-"
        @warn "`disable_stacktrace_truncation!` is not needed anymore, as stacktraces are truncated by default." maxlog=1
    else
        return TruncatedStacktraces.VERBOSE[] = disable
    end
end

# NamedTuple / Tuples -- Lux uses them quite frequenty (states) making the error messages
# too verbose
@static if VERSION ≤ v"1.9" && !TruncatedStacktraces.DISABLE
    function Base.show(io::IO, t::Type{<:Tuple})
        if (TruncatedStacktraces.VERBOSE[] ||
            !hasfield(t, :parameters) ||
            length(t.parameters) == 0)
            invoke(show, Tuple{IO, Type}, io, t)
        else
            try
                fields = t.parameters
                fields_truncated = length(fields) > 2 ? "$(fields[1]),$(fields[2]),…" :
                                   (length(fields) == 2 ? "$(fields[1]),$(fields[2])" :
                                    (length(fields) == 1 ? "$(fields[1])" : ""))
                print(io, "Tuple{$fields_truncated}")
            catch
                invoke(show, Tuple{IO, Type}, io, t)
            end
        end
    end

    function Base.show(io::IO, t::Type{<:NamedTuple})
        if (TruncatedStacktraces.VERBOSE[] ||
            !hasfield(t, :parameters) ||
            length(t.parameters) == 0)
            invoke(show, Tuple{IO, Type}, io, t)
        else
            try
                fields = first(t.parameters)
                fields_truncated = length(fields) > 2 ? "$(fields[1]),$(fields[2]),…" :
                                   (length(fields) == 2 ? "$(fields[1]),$(fields[2])" :
                                    (length(fields) == 1 ? "$(fields[1])" : ""))
                print(io, "NamedTuple{($fields_truncated),…}")
            catch
                invoke(show, Tuple{IO, Type}, io, t)
            end
        end
    end
end

# Lux Layers
## layers/basic.jl
@truncate_stacktrace WrappedFunction

@truncate_stacktrace Dense 1

@truncate_stacktrace Scale 1

@truncate_stacktrace Bilinear 1

@truncate_stacktrace Embedding

## layers/containers.jl

# All the containers will just use truncated stacktraces of the different
# base layers

## layers/conv.jl
@truncate_stacktrace Conv 1 2

@truncate_stacktrace Upsample 1

@truncate_stacktrace CrossCor 1 2

@truncate_stacktrace ConvTranspose 1 2

## layers/dropout.jl
@truncate_stacktrace Dropout

@truncate_stacktrace VariationalHiddenDropout

## layers/normalize.jl
@truncate_stacktrace BatchNorm

@truncate_stacktrace GroupNorm

@truncate_stacktrace LayerNorm

@truncate_stacktrace InstanceNorm

@truncate_stacktrace WeightNorm 2

## layers/recurrent.jl
@truncate_stacktrace RNNCell 1

@truncate_stacktrace LSTMCell 1

@truncate_stacktrace GRUCell 1
