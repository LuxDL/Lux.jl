# We are not using DifferentiationInterface because we need to support multiple arguments
# Zygote.jl
function gradient(f::F, ::AutoZygote, args...) where {F}
    return map((xᵢ, dxᵢ) -> dxᵢ === nothing || xᵢ isa Number ? CRC.ZeroTangent() : dxᵢ,
        args, Zygote.gradient(f, args...))
end

# FiniteDiff.jl
function gradient(f::F, ::AutoFiniteDiff, args...) where {F}
    gs = Vector{Any}(undef, length(args))
    for i in 1:length(args)
        _f, x = partial_function(f, i, args...)
        if x isa AbstractArray
            gs[i] = FD.finite_difference_gradient(_f, x)
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

# Enzyme.jl
function gradient(f::F, ::AutoEnzyme{Nothing}, args...) where {F}
    return gradient(f, AutoEnzyme(Enzyme.Reverse), args...)
end

function gradient(f::F, ad::AutoEnzyme{<:Enzyme.ReverseMode}, args...) where {F}
    args_activity = map(args) do x
        x isa Number && return Enzyme.Active(x)
        needs_gradient(x) && return Enzyme.Duplicated(x, Enzyme.make_zero(x))
        return Enzyme.Const(x)
    end
    res = Enzyme.autodiff(ad.mode, f, Enzyme.Active, args_activity...)
    counter = 1
    return Tuple(map(enumerate(args)) do (i, x)
        if x isa Number
            counter += 1
            return res[counter - 1]
        end
        needs_gradient(x) && return args_activity[i].dval
        return CRC.NoTangent()
    end)
end

function gradient(f::F, ::AutoEnzyme{<:Enzyme.ForwardMode}, args...) where {F}
    return error("AutoEnzyme{ForwardMode} is not supported yet.")
end
