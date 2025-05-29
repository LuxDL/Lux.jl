module WeightInitializersReactantExt

using Random: AbstractRNG
using Reactant: Reactant, TracedUtils, ReactantRNG, TracedRArray, @reactant_overlay
using WeightInitializers: DeviceAgnostic

# random numbers are automatically handled
for op in (:zeros, :ones)
    @eval begin
        function DeviceAgnostic.$(op)(
            ::ReactantRNG, ::Type{T}, dims::Integer...
        ) where {T<:Number}
            if Reactant.within_compile()
                return TracedUtils.promote_to(
                    TracedRArray{T,length(dims)}, $(op)(T, dims...)
                )
            else
                return Reactant.to_rarray($(op)(T, dims...))
            end
        end

        @reactant_overlay @noinline function DeviceAgnostic.$(op)(
            ::AbstractRNG, ::Type{T}, dims::Integer...
        ) where {T}
            return TracedUtils.promote_to(TracedRArray{T,length(dims)}, $(op)(T, dims...))
        end
    end
end

end
