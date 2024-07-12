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

# ::Type{T} is already specified
function (f::PartialWeightInitializationFunction{T, F, <:AbstractRNG})(
        dims::Integer...; kwargs...) where {T <: Number, F}
    return f.f(f.rng, T, dims...; f.kwargs..., kwargs...)
end
function (f::PartialWeightInitializationFunction{T, F, Nothing})(
        rng::AbstractRNG; kwargs...) where {T <: Number, F}
    return PartialWeightInitializationFunction{T}(f.f, rng, (; f.kwargs..., kwargs...))
end
function (f::PartialWeightInitializationFunction{T, F, Nothing})(
        rng::AbstractRNG, dims::Integer...; kwargs...) where {T <: Number, F}
    return f.f(rng, T, dims...; f.kwargs..., kwargs...)
end

# ::Type{T} is not needed
function (f::PartialWeightInitializationFunction{Missing, F, <:AbstractRNG})(
        dims::Integer...; kwargs...) where {F}
    return f.f(f.rng, dims...; f.kwargs..., kwargs...)
end
function (f::PartialWeightInitializationFunction{Missing, F, Nothing})(
        rng::AbstractRNG; kwargs...) where {F}
    return PartialWeightInitializationFunction{Missing}(
        f.f, rng, (; f.kwargs..., kwargs...))
end
function (f::PartialWeightInitializationFunction{Missing, F, Nothing})(
        rng::AbstractRNG, dims::Integer...; kwargs...) where {F}
    return f.f(rng, dims...; f.kwargs..., kwargs...)
end

# ::Type{T} is not specified
function (f::PartialWeightInitializationFunction{Nothing, F, Union{<:AbstractRNG, Nothing}})(
        ::Type{T}; kwargs...) where {T <: Number, F}
    return PartialWeightInitializationFunction{T}(f.f, f.rng, (; f.kwargs..., kwargs...))
end
function (f::PartialWeightInitializationFunction{Nothing, F, <:AbstractRNG})(
        ::Type{T}, dims::Integer...; kwargs...) where {T <: Number, F}
    return f.f(f.rng, T, dims...; f.kwargs..., kwargs...)
end
function (f::PartialWeightInitializationFunction{Nothing, F, Nothing})(
        rng::AbstractRNG, ::Type{T}; kwargs...) where {T <: Number, F}
    return PartialWeightInitializationFunction{T}(f.f, rng, (; f.kwargs..., kwargs...))
end
function (f::PartialWeightInitializationFunction{Nothing, F, Nothing})(
        rng::AbstractRNG, ::Type{T}, dims::Integer...; kwargs...) where {T <: Number, F}
    return f.f(rng, T, dims...; f.kwargs..., kwargs...)
end
