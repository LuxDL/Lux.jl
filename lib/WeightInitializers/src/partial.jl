@concrete struct PartialWeightInitializationFunction{T} <: Function
    f <: Function
    rng <: Union{Nothing, AbstractRNG}
    kwargs
end

function Base.show(
        io::IO, ::MIME"text/plain", f::PartialWeightInitializationFunction{T}) where {T}
    print(io, "$(f.f)(")
    f.rng !== nothing ? print(io, "$(f.rng), ") : print(io, "rng, ")
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
    print(io, ")")
end

function (f::PartialWeightInitializationFunction{<:Union{Nothing, Missing}})(
        args...; kwargs...)
    f.rng === nothing && return f.f(args...; f.kwargs..., kwargs...)
    return f.f(f.rng, args...; f.kwargs..., kwargs...)
end
function (f::PartialWeightInitializationFunction{T})(args...; kwargs...) where {T <: Number}
    f.rng === nothing && return f.f(T, args...; f.kwargs..., kwargs...)
    return f.f(f.rng, T, args...; f.kwargs..., kwargs...)
end
