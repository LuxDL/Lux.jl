struct LuxEltypeAdaptor{T} end

(l::LuxEltypeAdaptor)(x) = fmap(adapt(l), x)
function (l::LuxEltypeAdaptor)(x::AbstractArray{T}) where {T}
    return isbitstype(T) ? adapt(l, x) : map(adapt(l), x)
end

function Adapt.adapt_storage(
    ::LuxEltypeAdaptor{T}, x::AbstractArray{<:AbstractFloat}
) where {T<:AbstractFloat}
    return convert(AbstractArray{T}, x)
end

function Adapt.adapt_storage(
    ::LuxEltypeAdaptor{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}
) where {T<:AbstractFloat}
    return convert(AbstractArray{Complex{T}}, x)
end

for (fname, ftype) in zip((:f16, :f32, :f64), (Float16, Float32, Float64))
    @eval begin
        """
            $($fname)(m)

        Converts the `eltype` of `m` *floating point* values to `$($ftype)`.
        To avoid recursion into structs mark them with `Functors.@leaf`.
        """
        $(fname)(m) = (LuxEltypeAdaptor{$(ftype)}())(m)
    end
end

@static if isdefined(Core, :BFloat16)
    bf16_docs = """
        bf16(m)

    Converts the `eltype` of `m` *floating point* values to `BFloat16`.
    To avoid recursion into structs mark them with `Functors.@leaf`.

    !!! warning

        `BFloat16s.jl` needs to be loaded before using this function.

    !!! tip "Support for `BFloat16`"

        Most Lux operations aren't optimized for `BFloat16` yet. Instead this is meant to be
        used together with `Reactant.@compile`.
    """

    bf16(m) = (LuxEltypeAdaptor{Core.BFloat16}())(m)
else
    bf16_docs = """
        bf16(m)

    !!! danger "Not Supported"

        Current Julia version does not support `BFloat16`. Use julia 1.11 or newer.
    """

    bf16(_) = error("`bf16` is not supported on Julia versions 1.11+")
end

@doc (bf16_docs) bf16

# Common incorrect usage
for f in (f16, f32, f64, bf16)
    warn_msg = "$(f) is not meant to be broadcasted like `$(f).(x)` or `x .|> $(f)`, \
                and this might give unexpected results and could lead to crashes. Directly \
                use `$(f)` as `$(f)(x)` or `x |> $(f)` instead."
    @eval begin
        function Base.Broadcast.broadcasted(::typeof($(f)), arg1)
            @warn $(warn_msg)
            arg1′ = Broadcast.broadcastable(arg1)
            return Broadcast.broadcasted(Broadcast.combine_styles(arg1′), $(f), arg1′)
        end

        function Base.Broadcast.broadcasted(::typeof(|>), arg1, ::typeof($(f)))
            @warn $(warn_msg)
            arg1′ = Broadcast.broadcastable(arg1)
            return Broadcast.broadcasted(Broadcast.combine_styles(arg1′), $(f), arg1′)
        end
    end
end
