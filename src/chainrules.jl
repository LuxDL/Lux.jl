# Non Differentiable Functions
CRC.@non_differentiable replicate(::Any)
CRC.@non_differentiable compute_adaptive_pooling_dims(::Any, ::Any)
CRC.@non_differentiable glorot_normal(::Any...)
CRC.@non_differentiable glorot_uniform(::Any...)
CRC.@non_differentiable kaiming_normal(::Any...)
CRC.@non_differentiable kaiming_uniform(::Any...)
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
    function ∇merge(dy::Union{NoTangent, ZeroTangent})
        return (NoTangent(), NoTangent(), NoTangent())
    end
    return y, ∇merge
end

function CRC.rrule(::typeof(_eachslice), x, d::Val)
    return _eachslice(x, d), Δ -> (NoTangent(), ∇_eachslice(Δ, x, d), NoTangent())
end

# RNN Helpers
## Taken from https://github.com/FluxML/Flux.jl/blob/1f82da4bfa051c809f7f3ce7dd7aeb43be515b14/src/layers/recurrent.jl#L9
function CRC.rrule(::typeof(multigate), x::AbstractArray, c::Val{N}) where {N}
    function multigate_pullback(dy)
        dx = map!(zero, similar(x, float(eltype(x)), axes(x)), x)
        foreach(multigate(dx, c), dy) do dxᵢ, dyᵢ
            dyᵢ isa AbstractZero && return
            @. dxᵢ += dyᵢ
        end
        return (NoTangent(), dx, NoTangent())
    end
    return multigate(x, c), multigate_pullback
end

# foldl_init
function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(foldl_init), op::G, x::Tuple,
        init) where {G}
    x_arr = [x...]
    y, ∇foldl_init_internal = CRC.rrule_via_ad(cfg, foldl_init, op, x_arr, init)
    function ∇foldl_init(Δ)
        ∂foldl_init, ∂op, ∂x, ∂init = ∇foldl_init_internal(Δ)
        ∂x = Tuple(∂x)
        return ∂foldl_init, ∂op, ∂x, ∂init
    end
    return y, ∇foldl_init
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(foldl_init), op::G,
        x::AbstractArray, init) where {G}
    list, start = x, init
    hobbits = Vector{Any}(undef, length(list))  # Unfornately Zygote needs this
    accumulate!(hobbits, list; init=(start, nothing)) do (a, _), b
        return CRC.rrule_via_ad(cfg, op, a, b)
    end
    y = first(last(hobbits))
    ax = axes(x)
    project = ProjectTo(x)
    function ∇foldl_init(Δ)
        trio = accumulate(reverse(hobbits); init=(0, Δ, 0)) do (_, dc, _), (_, back)
            return back(dc)
        end
        ∂op = sum(first, trio)
        ∂x = map(last, reverse(trio))
        return NoTangent(), ∂op, project(reshape(∂x, ax)), trio[end][2]
    end
    return y, ∇foldl_init
end
