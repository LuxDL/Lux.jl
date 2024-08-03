"""
    bias_activation(σ, x, bias)

Applies the activation function `σ` elementwise to the result of broadcasted addition of `x`
and `bias` along the penultimate dimension. A vector `x` is treated as a matrix with a
single last dimension.

## Arguments

  - `σ`: Activation function
  - `x`: Input to be transformed
  - `bias`: Bias to be added. Can be `nothing`.

See also [`bias_activation!!`](@ref), [`fast_activation!!`](@ref).
"""
function bias_activation(σ::F, x::AbstractArray, bias::Optional{<:AbstractVector}) where {F}
    _bias_act_check(x, bias)
    return _bias_activation_impl(select_fastest_activation(σ, x, bias),
        attempt_fast_implementation((x, bias)), x, bias)
end

for (fast_mode, fop) in (
    (True, :__bias_activation_impl), (False, :__generic_bias_activation))
    @eval function _bias_activation_impl(σ::F, ::$(fast_mode), x::AbstractArray,
            bias::Optional{<:AbstractVector}) where {F}
        return $(fop)(σ, x, bias)
    end
end

"""
    bias_activation!!(σ, x, bias)

Same as [`bias_activation`](@ref) but might update `x` in-place if possible. Users should
not rely on `x` being mutated, it is recommended to use it like
`y = bias_activation!!(σ, x, bias)`. If `x` is updated in-place, `y` aliases `x`.

See also [`bias_activation`](@ref), [`fast_activation!!`](@ref).
"""
function bias_activation!!(
        σ::F, x::AbstractArray, bias::Optional{<:AbstractVector}) where {F}
    _bias_act_check(x, bias)
    return _bias_activation_impl!!(select_fastest_activation(σ, x, bias),
        attempt_fast_implementation((x, bias)), x, bias)
end

for (fast_mode, fop) in (
    (True, :__bias_activation_impl!!), (False, :__generic_bias_activation))
    @eval function _bias_activation_impl!!(σ::F, ::$(fast_mode), x::AbstractArray,
            bias::Optional{<:AbstractVector}) where {F}
        return $(fop)(σ, x, bias)
    end
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
