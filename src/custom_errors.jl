abstract type AbstractLuxException <: Exception end

function Base.showerror(io::IO, e::Err) where {Err<:AbstractLuxException}
    return print(io, string(nameof(Err)) * "(" * (hasfield(Err, :msg) ? e.msg : "") * ")")
end

struct LuxCompactModelParsingException <: AbstractLuxException
    msg::String
end

struct FluxModelConversionException <: AbstractLuxException
    msg::String
end

struct SimpleChainsModelConversionException <: AbstractLuxException
    msg::String
end

function SimpleChainsModelConversionException(layer::AbstractLuxLayer)
    return SimpleChainsModelConversionException("Conversion to SimpleChains not supported \
                                                 for $(typeof(layer))")
end

struct EltypeMismatchException <: AbstractLuxException
    msg::String
end
