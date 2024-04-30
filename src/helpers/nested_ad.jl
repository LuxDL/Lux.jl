# Needed for nice nested AD
function __forwarddiff_jvp end

function __partials end  # DON'T REMOVE THIS (DEQs.jl is using it)

GRADIENT_CONVERTIBLE_FUNCTIONS = [ComposedFunction{<:Any, <:StatefulLuxLayer},
    ComposedFunction{<:StatefulLuxLayer}, StatefulLuxLayer]

@inline function __rewrite_ad_call_for_inputs(f::F) where {F}
    f isa ComposedFunction{<:Any, <:StatefulLuxLayer} && return f, f.inner.ps
    f isa ComposedFunction{<:StatefulLuxLayer} &&
        return @closure((x, ps)->f.outer(f.inner(x), ps)), f.inner.ps
    f isa StatefulLuxLayer && return f, f.ps
    return error("Unknown function type: $(typeof(f))")
end

# Essentially computes the gradient of `f(x, y)` wrt x using the function `grad_fn`
# To compute the gradient of `f(x, y)` wrt y, just reorder the arguments with a wrapper
# over `f`
@inline function __internal_ad_gradient_call(grad_fn::G, f::F, x, y) where {G, F}
    return grad_fn(@closure(x->f(x, y)), x)
end
@inline function __internal_ad_gradient_call_no_custom_rrule(
        grad_fn::G, f::F, x, y) where {G, F}
    return grad_fn(@closure(x->f(x, y)), x) # Don' call `__internal_ad_gradient_call`
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__internal_ad_gradient_call), grad_fn::G, f::F, x, y) where {G, F}
    # Check if we can use the faster implementation
    if !Lux._is_extension_loaded(Val(:ForwardDiff)) || DISABLE_AUTOMATIC_NESTED_AD_SWITCH
        if !DISABLE_AUTOMATIC_NESTED_AD_SWITCH
            @warn "Load ForwardDiff.jl for better nested AD handling." maxlog=1
        end
        # Use the AD itself for whatever reason
        return CRC.rrule_via_ad(
            cfg, __internal_ad_gradient_call_no_custom_rrule, grad_fn, f, x, y)
    end

    res = __internal_ad_gradient_call(grad_fn, f, x, y)
    ∇internal_gradient_capture = @closure Δ_ -> begin
        (Δ_ isa CRC.NoTangent || Δ_ isa CRC.ZeroTangent) &&
            return ntuple(Returns(CRC.NoTangent()), 5)

        Δ = CRC.backing(CRC.unthunk(Δ_))
        Δ isa Tuple && (Δ = only(Δ))  # For Zygote and such which return a tuple
        ∂x, ∂y = Lux.__forwarddiff_jvp(@closure((x, y)->grad_fn(f, x, y)), x, Δ, y)
        return CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y
    end

    return res, ∇internal_gradient_capture
end
