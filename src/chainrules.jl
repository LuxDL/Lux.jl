# Non Differentiable Functions
CRC.@non_differentiable replicate(::Any)
CRC.@non_differentiable compute_adaptive_pooling_dims(::Any, ::Any)
CRC.@non_differentiable glorot_normal(::Any...)
CRC.@non_differentiable glorot_uniform(::Any...)
CRC.@non_differentiable kaiming_normal(::Any...)
CRC.@non_differentiable kaiming_uniform(::Any...)
CRC.@non_differentiable check_use_cuda()
CRC.@non_differentiable istraining(::Any)
CRC.@non_differentiable _get_norm_except_dims(::Any, ::Any)
CRC.@non_differentiable _affine(::Any)
CRC.@non_differentiable _track_stats(::Any)
CRC.@non_differentiable _conv_transpose_dims(::Any...)
CRC.@non_differentiable _calc_padding(::Any...)

# Utilities
function CRC.rrule(::typeof(merge), nt1::NamedTuple{F1}, nt2::NamedTuple{F2}) where {F1, F2}
    y = merge(nt1, nt2)
    function merge_pullback(dy)
        dnt1 = NamedTuple((f1 => (f1 in F2 ? NoTangent() : getproperty(dy, f1))
                           for f1 in F1))
        dnt2 = NamedTuple((f2 => getproperty(dy, f2) for f2 in F2))
        return (NoTangent(), dnt1, dnt2)
    end
    function merge_pullback(dy::Union{NoTangent, ZeroTangent})
        return (NoTangent(), NoTangent(), NoTangent())
    end
    return y, merge_pullback
end

function CRC.rrule(::typeof(vec), x::AbstractMatrix)
    y = vec(x)
    vec_pullback(dy) = NoTangent(), reshape(dy, size(x))
    return y, vec_pullback
end

function CRC.rrule(::typeof(collect), v::Vector)
    y = collect(v)
    function collect_pullback(dy)
        return NoTangent(), dy
    end
    return y, collect_pullback
end

function CRC.rrule(::typeof(copy), x)
    copy_pullback(dy) = (NoTangent(), dy)
    return copy(x), copy_pullback
end

function CRC.rrule(::typeof(_eachslice), x, d)
    return _eachslice(x, d), Δ -> (NoTangent(), ∇_eachslice(Δ, x, d), NoTangent())
end

# Adapt Interface
function CRC.rrule(::Type{Array}, x::CUDA.CuArray)
    return Array(x), d -> (NoTangent(), CUDA.cu(d))
end

function CRC.rrule(::typeof(adapt_storage), to::LuxCPUAdaptor, x::CUDA.AbstractGPUArray)
    return adapt_storage(to, x),
           d -> (NoTangent(), NoTangent(), adapt_storage(LuxCUDAAdaptor(), d))
end

function CRC.rrule(::typeof(adapt_storage), to::LuxCUDAAdaptor, x::Array)
    return adapt_storage(to, x),
           d -> (NoTangent(), NoTangent(), adapt_storage(LuxCPUAdaptor(), d))
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
        return (NoTangent(), dx, NoTangent(), NoTangent())
    end
    return multigate(x, c), multigate_pullback
end

# layers/recurrent.jl
function CRC.rrule(::typeof(_generate_init_recurrence), out, carry, state)
    result = _generate_init_recurrence(out, carry, state)
    return result, Δ -> (NoTangent(), ∇_generate_init_recurrence(Δ)...)
end
