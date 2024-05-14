@inline function Lux.__batched_jacobian(f::F, backend::AutoZygote, x, p) where {F}
    return Lux.__batched_jacobian_impl(Base.Fix2(f, p), backend, x)
end

@inline function Lux.__batched_jacobian(f::F, backend::AutoZygote, x) where {F}
    return Lux.__batched_jacobian_impl(f, backend, x)
end

function Lux.__batched_jacobian_impl(f::F, ::AutoZygote, x) where {F}
    

    # x_size = size(x)
    # __f = @closure x -> f(reshape(x, x_size))
    # tag = backend.tag === nothing ? ForwardDiff.Tag(__f, eltype(x)) : backend.tag
    # chunksize = (CK === nothing || CK ≤ 0) ?
    #             ForwardDiff.Chunk{min(prod(size(x)[1:(end - 1)]), 8)}() :
    #             ForwardDiff.Chunk{CK}()
    # return __batched_forwarddiff_jacobian(
    #     __f, reshape(x, :, size(x, ndims(x))), typeof(tag), chunksize)
    error("`batched_jacobian` is not supported for `AutoZygote`. Please use `Zygote.jl` instead.")
end
