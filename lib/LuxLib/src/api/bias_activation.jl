function bias_activation(σ::F, x::AbstractArray, bias::Optional{<:AbstractVector}) where {F}
    _bias_act_check(x, bias)
    return __bias_activation_impl(σ, x, bias)
end

function bias_activation!!(
        σ::F, x::AbstractArray, bias::Optional{<:AbstractVector}) where {F}
    _bias_act_check(x, bias)
    return __bias_activation_impl!!(σ, x, bias)
end

_bias_act_check(x, b) = nothing
function _bias_act_check(x::AbstractArray{<:Number, N}, bias::AbstractVector) where {N}
    if N == 1
        @assert length(bias) == length(x)
    else
        @assert length(bias) == size(x, N - 1)
    end
end

CRC.@non_differentiable _bias_act_check(::Any, ::Any)
EnzymeRules.inactive_noinl(::typeof(_bias_act_check), ::Any, ::Any) = nothing
