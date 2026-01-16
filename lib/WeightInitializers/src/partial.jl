module PartialFunction

using ConcreteStructs: @concrete
using Random: AbstractRNG

@concrete struct Partial{T} <: Function
    f <: Function
    rng <: Union{Nothing,AbstractRNG}
    kwargs
end

function Base.show(io::IO, ::MIME"text/plain", f::Partial{T}) where {T}
    print(io, "$(f.f)(")
    if f.rng !== nothing
        print(io, "$(nameof(typeof(f.rng)))(...), ")
    else
        print(io, "rng, ")
    end
    if T === Nothing
        print(io, "::Type{T}, ")
    else
        T !== Missing ? print(io, "$(T), ") : nothing
    end
    print(io, "dims...")
    kwargs_str = String[]
    for (k, v) in pairs(f.kwargs)
        push!(kwargs_str, "$(k)=$(v)")
    end
    length(kwargs_str) > 0 && print(io, "; ", join(kwargs_str, ", "))
    return print(io, ")")
end

function (f::Partial{<:Union{Nothing,Missing}})(args...; kwargs...)
    f.rng === nothing && return f.f(args...; f.kwargs..., kwargs...)
    return f.f(f.rng, args...; f.kwargs..., kwargs...)
end
function (f::Partial{<:Union{Nothing,Missing}})(rng::AbstractRNG, args...; kwargs...)
    @assert f.rng === nothing
    return f.f(rng, args...; f.kwargs..., kwargs...)
end
function (f::Partial{T})(args...; kwargs...) where {T<:Number}
    f.rng === nothing && return f.f(T, args...; f.kwargs..., kwargs...)
    return f.f(f.rng, T, args...; f.kwargs..., kwargs...)
end
function (f::Partial{T})(rng::AbstractRNG, args...; kwargs...) where {T<:Number}
    @assert f.rng === nothing
    return f.f(rng, T, args...; f.kwargs..., kwargs...)
end

end
