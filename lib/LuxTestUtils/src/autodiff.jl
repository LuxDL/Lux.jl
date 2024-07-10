# We are not using DifferentiationInterface because we need to support multiple arguments
function gradient(f::F, ::AutoZygote, args...) where {F}
    grads = Zygote.gradient(f, args...)
    return map(x -> x === nothing ? CRC.ZeroTangent() : x, grads)
end

function gradient(f::F, ::AutoFiniteDiff, args...) where {F}
    gs = Vector{Any}(undef, length(args))
    for i in 1:length(args)
        _f, x = partial_function(f, i, args...)
        if x isa AbstractArray
            gs[i] = FD.finite_difference_gradient(_f, x)
        elseif x isa Number
            gs[i] = FD.finite_difference_derivative(_f, x)
        elseif x isa NamedTuple
            __f, x_flat = flatten_gradient_computable(_f, x)
            gs[i] = x_flat === nothing ? CRC.NoTangent() :
                    NamedTuple(FD.finite_difference_gradient(__f, x_flat))
        else
            gs[i] = CRC.NoTangent()
        end
    end
    return Tuple(gs)
end
