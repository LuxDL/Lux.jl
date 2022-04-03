zeros32(rng::AbstractRNG, args...; kwargs...) = zeros32(args...; kwargs...)
ones32(rng::AbstractRNG, args...; kwargs...) = ones32(args...; kwargs...)
Base.zeros(rng::AbstractRNG, args...; kwargs...) = zeros(args...; kwargs...)
Base.ones(rng::AbstractRNG, args...; kwargs...) = ones(args...; kwargs...)

function var!(y, x; kwargs...)
    mean!(y, x)
    return var!(y, x, y; kwargs...)
end

function var!(y1, y2, x, μ; corrected::Bool = true)
    m = (length(x) ÷ length(y1)) - corrected
    @. y2 = abs2(x - μ) / m
    mean!(y1, x)
    return y1
end

istraining() = false