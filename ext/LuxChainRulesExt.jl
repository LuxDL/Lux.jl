module LuxChainRulesExt

using ChainRules: ChainRules

# https://github.com/FluxML/Zygote.jl/pull/1328 broke the RNNs completely. Putting an
# emergency patch here
function ChainRules._setindex_zero(
        x::Vector{<:AbstractArray{T}}, dy, inds::Integer...) where {T <: Number}
    return [fill!(similar(xᵢ), 0) for xᵢ in x]
end

function ChainRules.∇getindex!(
        dx::Vector{<:AbstractArray{T}}, dy, inds::Integer...) where {T <: Number}
    dx[inds...] .+= dy
    return dx
end

end
