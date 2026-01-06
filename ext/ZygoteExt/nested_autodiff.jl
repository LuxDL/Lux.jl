function Lux.AutoDiffInternalImpl.vector_jacobian_product_impl(
    f::F, ::AutoZygote, x, u
) where {F}
    _, pb_f = Zygote.pullback(f, x)
    return only(pb_f(u))
end

# Nested AD Handling
for fType in Lux.AutoDiffInternalImpl.AD_CONVERTIBLE_FUNCTIONS
    @eval begin
        function Zygote.gradient(f::$fType, x)
            f̂, y = Lux.AutoDiffInternalImpl.rewrite_autodiff_call(f)
            return Lux.AutoDiffInternalImpl.autodiff_gradient(Zygote.gradient, f̂, x, y)
        end

        function Zygote.jacobian(f::$fType, x::AbstractArray)
            f̂, y = Lux.AutoDiffInternalImpl.rewrite_autodiff_call(f)
            return Lux.AutoDiffInternalImpl.autodiff_jacobian(
                Zygote.jacobian, Zygote.gradient, f̂, x, y
            )
        end

        function Lux.AutoDiffInternalImpl.vector_jacobian_product_impl(
            f::$fType, ::AutoZygote, x, u
        )
            f̂, y = Lux.AutoDiffInternalImpl.rewrite_autodiff_call(f)
            return Lux.AutoDiffInternalImpl.autodiff_pullback(Zygote.pullback, f̂, x, y, u)
        end
    end
end

# Handle Weird Zygote shit
# Forward to a function that doesn't have this _pullback defined so that it triggers the
# rrule
for fType in Lux.AutoDiffInternalImpl.AD_CONVERTIBLE_FUNCTIONS,
    (fdiff_func, cfg_func) in ((:jacobian, :JacobianConfig), (:gradient, :GradientConfig))

    @eval function Zygote._pullback(
        cx::Zygote.AContext, ::typeof(ForwardDiff.$fdiff_func), f::$fType, x::AbstractArray
    )
        return Zygote._pullback(
            cx, ForwardDiff.$fdiff_func, f, x, ForwardDiff.$(cfg_func)(f, x), Val(true)
        )
    end
end
