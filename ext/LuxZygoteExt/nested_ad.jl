function Lux.__vector_jacobian_product_impl(f::F, ::AutoZygote, x, u) where {F}
    _, pb_f = Zygote.pullback(f, x)
    return only(pb_f(u))
end

# Nested AD Handling
for fType in Lux.AD_CONVERTIBLE_FUNCTIONS
    @eval begin
        @inline function Zygote.gradient(f::$fType, x)
            f_internal, y = Lux.__rewrite_ad_call(f)
            return Lux.__internal_ad_gradient_call(Zygote.gradient, f_internal, x, y)
        end

        @inline function Zygote.jacobian(f::$fType, x::AbstractArray)
            f_internal, y = Lux.__rewrite_ad_call(f)
            return Lux.__internal_ad_jacobian_call(
                Zygote.jacobian, Zygote.gradient, f_internal, x, y)
        end

        @inline function Lux.__vector_jacobian_product_impl(f::$fType, ::AutoZygote, x, u)
            f_internal, y = Lux.__rewrite_ad_call(f)
            return Lux.__internal_ad_pullback_call(Zygote.pullback, f_internal, x, y, u)
        end
    end
end

# Handle Weird Zygote shit
@static if isdefined(Zygote, :ForwardDiff)
    using Zygote: ForwardDiff

    # Forward to a function that doesn't have this _pullback defined so that it triggers the
    # rrule
    for fType in Lux.AD_CONVERTIBLE_FUNCTIONS,
        (fdiff_func, cfg_func) in (
            (:jacobian, :JacobianConfig), (:gradient, :GradientConfig))

        @eval function Zygote._pullback(
                cx::Zygote.AContext, ::typeof(ForwardDiff.$fdiff_func),
                f::$fType, x::AbstractArray)
            return Zygote._pullback(
                cx, ForwardDiff.$fdiff_func, f, x, ForwardDiff.$(cfg_func)(f, x), Val(true))
        end
    end
end
