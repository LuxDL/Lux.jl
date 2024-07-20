# Currently these don't do anything. But once we add LoopVectorization.jl and
# VectorizedStatistics.jl, we can will specialize the CPU dispatches to use them.
fast_mean(x::AbstractArray; dims=:) = fast_mean(internal_operation_mode(x), x; dims)
fast_mean(opmode, x::AbstractArray; dims=:) = mean(x; dims)

fast_var(x::AbstractArray; kwargs...) = fast_var(internal_operation_mode(x), x; kwargs...)
function fast_var(opmode, x::AbstractArray; mean=nothing, dims=:, corrected=true)
    return var(x; mean, dims, corrected)
end
