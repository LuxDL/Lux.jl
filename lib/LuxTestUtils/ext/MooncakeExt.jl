module MooncakeExt

using ADTypes: AutoMooncake
using Mooncake: Mooncake
using LuxTestUtils: LuxTestUtils

function __init__()
    @static if isempty(VERSION.prerelease)
        try
            # FIXME: Mooncake is currently wreaking havoc in the Lux repo test,
            # dropping testing for now
            # Mooncake.prepare_gradient_cache(Base.Fix1(sum, abs2), ones(Float32, 10))
            # LuxTestUtils.MOONCAKE_TESTING_ENABLED[] = true
            LuxTestUtils.MOONCAKE_TESTING_ENABLED[] = false
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
function mooncake_gradient_function(f, x)
    # Enable friendly_tangents for testing.
    cache = Mooncake.prepare_gradient_cache(
        f, x; config=Mooncake.Config(; friendly_tangents=true)
    )
    y, tangents = Mooncake.value_and_gradient!!(cache, f, x)
    tangent_func, tangent_args = tangents
    return tangent_args
end

end
