module MooncakeExt

using ADTypes: AutoMooncake
using Mooncake: Mooncake
using LuxTestUtils: LuxTestUtils

function __init__()
    @static if isempty(VERSION.prerelease)
        try
            Mooncake.prepare_gradient_cache(Base.Fix1(sum, abs2), ones(Float32, 10))
            LuxTestUtils.MOONCAKE_TESTING_ENABLED[] = true
        catch err
            @error "`Mooncake.jl` did not successfully differentiate a simple function or \
                    failed to load on $(VERSION). All Mooncake tests will be \
                    skipped." maxlog = 1 err = err
            LuxTestUtils.MOONCAKE_TESTING_ENABLED[] = false
        end
    end
end

function LuxTestUtils.gradient(f::F, ::AutoMooncake, args...) where {F}
    return LuxTestUtils.gradient(f, mooncake_gradient_function, args...)
end

"""
    mooncake_gradient_function(f, x)

Compute gradient using Mooncake.jl's value_and_gradient!! function.
Returns only the gradient for args x.
"""
# Mooncake's friendly_tangents falls back to a raw Tangent for wrappers it has no
# dedicated support for (Adjoint, Diagonal, SubArray, ...). Drop the non-differentiable
# fields, then unwrap to the single value left, or a plain NamedTuple if several are
# left, matching what other backends return. Also recurses into Tuples and into
# Arrays of Tangents; plain numeric/GPU arrays pass through untouched.
_unwrap_mooncake_tangent(t) = t
function _unwrap_mooncake_tangent(t::Union{Mooncake.Tangent,Mooncake.MutableTangent})
    kv = [(k, v) for (k, v) in pairs(t.fields) if !(v isa Mooncake.NoTangent)]
    isempty(kv) && return Mooncake.NoTangent()
    length(kv) == 1 && return _unwrap_mooncake_tangent(kv[1][2])
    return NamedTuple(k => _unwrap_mooncake_tangent(v) for (k, v) in kv)
end
_unwrap_mooncake_tangent(t::Tuple) = map(_unwrap_mooncake_tangent, t)
function _unwrap_mooncake_tangent(
    t::AbstractArray{<:Union{Mooncake.Tangent,Mooncake.MutableTangent}}
)
    return map(_unwrap_mooncake_tangent, t)
end

function mooncake_gradient_function(f, x)
    # Enable friendly_tangents for testing.
    cache = Mooncake.prepare_gradient_cache(
        f, x; config=Mooncake.Config(; friendly_tangents=true)
    )
    y, tangents = Mooncake.value_and_gradient!!(cache, f, x)
    tangent_func, tangent_args = tangents
    return _unwrap_mooncake_tangent(tangent_args)
end

end
