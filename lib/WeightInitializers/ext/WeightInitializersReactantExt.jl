module WeightInitializersReactantExt

using Random: AbstractRNG
using Reactant: Reactant, TracedUtils, TracedRNG, ConcreteRNG, TracedRArray,
    @reactant_overlay
using WeightInitializers: DeviceAgnostic

# random numbers are automatically handled
for op in (:zeros, :ones)
    @eval begin
        function DeviceAgnostic.$(op)(
                ::ConcreteRNG, ::Type{T}, dims::Integer...
            ) where {T <: Number}
            return Reactant.to_rarray($(op)(T, dims...))
        end

        function DeviceAgnostic.$(op)(
                ::TracedRNG, ::Type{T}, dims::Integer...
            ) where {T <: Number}
            return TracedUtils.promote_to(TracedRArray{T, length(dims)}, $(op)(T, dims...))
        end

        @reactant_overlay @noinline function DeviceAgnostic.$(op)(
                ::AbstractRNG, ::Type{T}, dims::Integer...
            ) where {T}
            return TracedUtils.promote_to(TracedRArray{T, length(dims)}, $(op)(T, dims...))
        end
    end
end

end
