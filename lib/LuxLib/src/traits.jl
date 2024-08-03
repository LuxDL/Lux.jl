# Immutable Array or Dual Numbers
is_mutable_array(x::T) where {T <: AbstractArray} = static(can_setindex(T))
is_mutable_array(::Nothing) = True()

is_dual_array(x) = False()
is_dual_array(::AbstractArray{<:ForwardDiff.Dual}) = True()

# Current Checks. If any of these are false, we fallback to the generic implementation.
#   - Is Mutable
#   - Doesn't Has Dual Numbers
attempt_fast_implementation(x) = attempt_fast_implementation((x,))
function attempt_fast_implementation(xs::Tuple)
    return unrolled_all(is_mutable_array, xs) & unrolled_all(!is_dual_array, xs)
end

CRC.@non_differentiable attempt_fast_implementation(::Any...)
EnzymeRules.inactive_noinl(::typeof(attempt_fast_implementation), ::Any...) = nothing
