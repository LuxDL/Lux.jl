# Non Differentiable Functions
CRC.@non_differentiable replicate(::Any)
CRC.@non_differentiable compute_adaptive_pooling_dims(::Any, ::Any)
CRC.@non_differentiable istraining(::Any)
CRC.@non_differentiable _get_norm_except_dims(::Any, ::Any)
CRC.@non_differentiable _affine(::Any)
CRC.@non_differentiable _track_stats(::Any)
CRC.@non_differentiable _conv_transpose_dims(::Any...)
CRC.@non_differentiable _calc_padding(::Any...)
CRC.@non_differentiable Base.printstyled(::Any...)
## Type Piracy: Needs upstreaming
## This is needed for fixing NamedTuple nested differentiation
CRC.@non_differentiable fieldcount(::Any)

# Utilities
function CRC.rrule(::typeof(merge), nt1::NamedTuple{F1}, nt2::NamedTuple{F2}) where {F1, F2}
    y = merge(nt1, nt2)
    function ∇merge(dy)
        dnt1 = NamedTuple((f1 => (f1 in F2 ? NoTangent() : getproperty(dy, f1))
        for f1 in F1))
        dnt2 = NamedTuple((f2 => getproperty(dy, f2) for f2 in F2))
        return (NoTangent(), dnt1, dnt2)
    end
    ∇merge(::Union{NoTangent, ZeroTangent}) = (NoTangent(), NoTangent(), NoTangent())
    return y, ∇merge
end

function CRC.rrule(::typeof(_eachslice), x, d::Val)
    return _eachslice(x, d), @closure(Δ->(NoTangent(), ∇_eachslice(Δ, x, d), NoTangent()))
end

# RNN Helpers
## Taken from https://github.com/FluxML/Flux.jl/blob/1f82da4bfa051c809f7f3ce7dd7aeb43be515b14/src/layers/recurrent.jl#L9
function CRC.rrule(::typeof(multigate), x::AbstractArray, c::Val{N}) where {N}
    function ∇multigate(dy)
        dx = map!(zero, similar(x, float(eltype(x)), axes(x)), x)
        foreach(multigate(dx, c), dy) do dxᵢ, dyᵢ
            dyᵢ isa AbstractZero && return
            @. dxᵢ += dyᵢ
        end
        return (NoTangent(), dx, NoTangent())
    end
    return multigate(x, c), ∇multigate
end

# foldl_init
function CRC.rrule(cfg::RuleConfig{>:HasReverseMode},
        ::typeof(foldl_init), op::G, x::Tuple, init) where {G}
    x_arr = [x...]
    y, ∇foldl_init_internal = CRC.rrule_via_ad(cfg, foldl_init, op, x_arr, init)
    ∇foldl_init = @closure Δ -> begin
        ∂foldl_init, ∂op, ∂x, ∂init = ∇foldl_init_internal(Δ)
        ∂x = Tuple(∂x)
        return ∂foldl_init, ∂op, ∂x, ∂init
    end
    return y, ∇foldl_init
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(foldl_init),
        op::G, x::AbstractArray, init) where {G}
    list, start = x, init
    accum_func = @closure (a, b) -> CRC.rrule_via_ad(cfg, op, first(a), b)
    accum_func_inner = @closure (x1, x2) -> begin
        (_d1, dc, _d3) = x1
        (_val, back) = x2
        return back(dc)
    end
    hobbits = Vector{Any}(undef, length(list))  # Unfornately Zygote needs this for CUDA
    accumulate!(accum_func, hobbits, list; init=(start, nothing))
    y = first(last(hobbits))
    ax = axes(x)
    project = ProjectTo.(x)
    ∇foldl_init = Δ -> begin
        trio = accumulate(accum_func_inner, reverse(hobbits); init=(0, Δ, 0))
        ∂op = sum(first, trio)
        ∂x = reshape(map(last, reverse(trio)), ax)
        return (NoTangent(), ∂op,
            [proj(∂xᵢ) for (proj, ∂xᵢ) in zip(project, ∂x)], last(trio)[2])
    end
    return y, ∇foldl_init
end

# getproperty rrule for AbstractExplicitLayer. needed for type stability of Zygote
# gradients
function CRC.rrule(::typeof(getproperty), m::AbstractExplicitLayer, name::Symbol)
    res = getproperty(m, name)
    ∇getproperty = Δ -> ntuple(Returns(NoTangent()), 3)
    return res, ∇getproperty
end
