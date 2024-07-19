# Currently these don't do anything. But once we add LoopVectorization.jl and
# VectorizedStatistics.jl, we can will specialize the CPU dispatches to use them.

fast_sum(x::AbstractArray; dims=:) = fast_sum(get_device_type(x), x; dims)
fast_sum(::Type{T}, x::AbstractArray; dims=:) where {T} = sum(x; dims)

fast_mean(x::AbstractArray; dims=:) = fast_mean(get_device_type(x), x; dims)
fast_mean(::Type{T}, x::AbstractArray; dims=:) where {T} = mean(x; dims)

fast_var(x::AbstractArray; kwargs...) = fast_var(get_device_type(x), x; kwargs...)
function fast_var(
        ::Type{T}, x::AbstractArray; mean=nothing, dims=:, corrected=true) where {T}
    return var(x; mean, dims, corrected)
end
