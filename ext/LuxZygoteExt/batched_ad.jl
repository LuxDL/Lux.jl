@inline function Lux.__batched_jacobian(f::F, backend::AutoZygote, x) where {F}
    return Lux.__batched_jacobian_impl(f, backend, x)
end

function Lux.__batched_jacobian_impl(f::F, ::AutoZygote, x) where {F}
    # It's not safe to run `pb_f` multiple times. We run the first time to be able to
    # construct the Jacobian
    y, pb_f = Zygote.pullback(f, x)

    @argcheck y isa AbstractArray MethodError
    if ndims(y) ≤ 1 || size(y, ndims(y)) != size(x, ndims(x))
        throw(AssertionError("`batched_jacobian` only supports batched outputs \
                              (ndims(y) > 1) && size(y, ndims(y)) == size(x, ndims(x))."))
    end

    J = similar(x, promote_type(eltype(y), eltype(x)), prod(size(y)[1:(end - 1)]),
        prod(size(x)[1:(end - 1)]), size(x, ndims(x)))
    __fill_chunked_jacobian!(J, 1, f, pb_f, y, x)

    if get_device_type(x) <: LuxCPUDevice  # Use threads
        tasks = map(2:size(J, 1)) do i
            Threads.@spawn begin
                yᵢ, pb_fᵢ = Zygote.pullback(f, x)
                __fill_chunked_jacobian!(J, i, f, pb_fᵢ, yᵢ, x)
            end
        end
        map(fetch, tasks)
    else # Threading has some issues with cuDNN. we are being safe and not using threads
        map(2:size(J, 1)) do i
            yᵢ, pb_fᵢ = Zygote.pullback(f, x)
            __fill_chunked_jacobian!(J, i, f, pb_fᵢ, yᵢ, x)
        end
    end

    return J
end

@inbounds function __fill_chunked_jacobian!(J, i::Int, ::F, pb_f::PBF, y, x) where {F, PBF}
    v = reshape(similar(y), :, size(y, ndims(y)))
    fill!(v, zero(eltype(y)))
    v[i, :] .= one(eltype(y))

    Jᵢ = only(pb_f(reshape(v, size(y))))
    J[i, :, :] .= reshape(Jᵢ, :, size(x, ndims(x)))

    return
end
