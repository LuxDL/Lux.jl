function __match_eltype end

@static if ELTYPE_MISMATCH_HANDLING == "none" # Just return the input
    @inline __match_eltype(layer, ps, st, x) = x
    @inline __match_eltype(layer, ps, st, x, args...) = (x, args...)
else
    @inline function __match_eltype(layer, ps, st, x)
        fn = let elType = recursive_eltype((ps, st), Val(true)), layer = layer
            arr -> __match_eltype(layer, elType, __eltype(arr), arr)
        end
        return recursive_map(fn, x)
    end
    @inline function __match_eltype(layer, ps, st, x, args...)
        fn = let elType = recursive_eltype((ps, st), Val(true)), layer = layer
            arr -> __match_eltype(layer, elType, __eltype(arr), arr)
        end
        return (recursive_map(fn, x), recursive_map(fn, args)...)
    end
end

@inline function __match_eltype(_, ::Type{Bool}, ::Type{T},
        x::AbstractArray{<:Union{AbstractFloat, Integer}}) where {T}
    return x
end

@inline function __match_eltype(_, ::Type{T}, ::Type{T},
        x::AbstractArray{<:Union{AbstractFloat, Integer}}) where {T}
    return x
end

@inline function __match_eltype(layer, ::Type{T1}, ::Type{T2},
        x::AbstractArray{<:Union{AbstractFloat, Integer}}) where {T1, T2}
    @static if ELTYPE_MISMATCH_HANDLING == "warn"
        @warn "Layer with $(T1) parameters and states received \
               $(T2) input." layer summary(x) maxlog=1
        return x
    elseif ELTYPE_MISMATCH_HANDLING == "convert"
        @warn "Layer with $(T1) parameters and states received $(T2) \
               input. Converting to $(T1)." layer summary(x) maxlog=1
        return convert(AbstractArray{T1}, x)
    elseif ELTYPE_MISMATCH_HANDLING == "error"
        throw(EltypeMismatchException("Layer $(layer) with $(T1) parameters and states \
                                       received $(T2) input. This is not allowed because \
                                       the preference `eltype_mismatch_handling` is set \
                                       to `error`. To debug further, use \
                                       `Lux.Experimental.@debug_mode`."))
    end
end

# Let some of the weird types pass through
@inline __match_eltype(_, ::Type, ::Type, x::AbstractArray) = x
