"""
    disable_stacktrace_truncation!(; disable::Bool=true)

An easy way to update `TruncatedStacktraces.VERBOSE` without having to load it manually.

Effectively does `TruncatedStacktraces.VERBOSE[] = disable`
"""
function disable_stacktrace_truncation!(; disable::Bool=true)
    return TruncatedStacktraces.VERBOSE[] = disable
end

# NamedTuple -- Lux uses them quite frequenty (states) making the error messages too verbose
function Base.show(io::IO, ::Type{<:NamedTuple{fields, fTypes}}) where {fields, fTypes}
    if TruncatedStacktraces.VERBOSE[]
        print(io, "NamedTuple{$fields, $fTypes}")
    else
        fields_truncated = if length(fields) > 2
            "($(fields[1]), $(fields[2]), ...)"
        else
            fields
        end
        print(io, "NamedTuple{$fields_truncated, ...}")
    end
end

# Automatically generate TruncatedStacktraces functions
# NOT PART OF API: CAN BE REMOVED WITHOUT IT BEING CONSIDERED BREAKING
macro __truncate_stacktrace(f, short_display)
    quote
        __truncate_stacktrace($(esc(f)), $(esc(short_display)))
    end
end

__maximum(x, ::Int) = maximum(x)
__maximum(::Tuple{}, t::Int) = t
__minimum(x, ::Int) = minimum(x)
__minimum(::Tuple{}, ::Int) = 1

function __truncate_stacktrace(l, short_display::NTuple{N, Int}) where {N}
    pcount = __get_parameter_count(l)
    @assert __maximum(short_display, pcount) <= pcount && __minimum(short_display, 1) >= 1

    name = :(Base.show)
    whereparams = ntuple(_ -> gensym(), pcount)
    args = Any[:(io::IO), :(t::Type{$l{$(whereparams...)}})]
    kwargs = []

    delim = ", "
    long_string = "$l{$(join(whereparams, delim)...)}"
    short_string = "$l{$(join([whereparams[i] for i in short_display], delim)...), ...}"

    body = quote
        if TruncatedStacktraces.VERBOSE[]
            print(io, string($l) * "{" * join([$(whereparams...)], ", ") * ", ...}")
        else
            print(io,
                  string($l) *
                  "{" *
                  join([$(whereparams[[short_display...]]...)], ", ") *
                  (length($short_display) == 0 ? "" : ", ") *
                  "...}")
        end
    end

    fdef = Dict(:name => name, :args => args, :kwargs => kwargs, :body => body,
                :whereparams => whereparams)

    return eval(MacroTools.combinedef(fdef))
end

function __get_parameter_count(T::Union{DataType, UnionAll})
    return length(Base.unwrap_unionall(T).parameters)
end

# Lux Layers
## layers/basic.jl
@__truncate_stacktrace WrappedFunction ()

@__truncate_stacktrace Dense (1,)

@__truncate_stacktrace Scale (1,)

@__truncate_stacktrace Bilinear (1,)

@__truncate_stacktrace Embedding ()

## layers/containers.jl

# All the containers will just use truncated stacktraces of the different
# base layers

## layers/conv.jl
@__truncate_stacktrace Conv (1, 2)

@__truncate_stacktrace Upsample (1,)

@__truncate_stacktrace CrossCor (1, 2)

@__truncate_stacktrace ConvTranspose (1, 2)

## layers/dropout.jl
@__truncate_stacktrace Dropout ()

@__truncate_stacktrace VariationalHiddenDropout ()

## layers/normalize.jl
@__truncate_stacktrace BatchNorm ()

@__truncate_stacktrace GroupNorm ()

@__truncate_stacktrace LayerNorm ()

@__truncate_stacktrace InstanceNorm ()

@__truncate_stacktrace WeightNorm (2,)

## layers/recurrent.jl
@__truncate_stacktrace RNNCell (1,)

@__truncate_stacktrace LSTMCell (1,)

@__truncate_stacktrace GRUCell (1,)
