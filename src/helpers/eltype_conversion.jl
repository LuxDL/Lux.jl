struct LuxEltypeAdaptor{T} end

(l::LuxEltypeAdaptor)(x) = fmap(adapt(l), x)
function (l::LuxEltypeAdaptor)(x::AbstractArray{T}) where {T}
    return isbitstype(T) ? adapt(l, x) : map(adapt(l), x)
end

function Adapt.adapt_storage(
        ::LuxEltypeAdaptor{T}, x::AbstractArray{<:AbstractFloat}) where {T <: AbstractFloat}
    return convert(AbstractArray{T}, x)
end

function Adapt.adapt_storage(::LuxEltypeAdaptor{T},
        x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T <: AbstractFloat}
    return convert(AbstractArray{Complex{T}}, x)
end

for (fname, ftype) in zip((:f16, :f32, :f64), (Float16, Float32, Float64))
    @eval begin
        """
            $($fname)(m)

        Converts the `eltype` of `m` *floating point* values to `$($ftype)`.
        Recurses into structs marked with `Functors.@functor`.
        """
        $(fname)(m) = (LuxEltypeAdaptor{$ftype}())(m)
    end
end

# Common incorrect usage
for f in (f16, f32, f64)
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
