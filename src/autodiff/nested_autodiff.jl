## Written like this to avoid dynamic dispatch from Zygote
# Input Gradient / Jacobian
function rewrite_autodiff_call(f::ComposedFunction{F,<:StatefulLuxLayer}) where {F}
    return (f, f.inner.ps)
end
function rewrite_autodiff_call(f::ComposedFunction{<:StatefulLuxLayer,F}) where {F}
    return (@closure((x, ps) -> f.outer(f.inner(x), ps)), f.outer.ps)
end
rewrite_autodiff_call(f::StatefulLuxLayer) = f, f.ps

# Parameter Gradient / Jacobian
function rewrite_autodiff_call(
    f::ComposedFunction{F,<:Base.Fix1{<:StatefulLuxLayer}}
) where {F}
    return @closure((ps, x) -> f.outer(f.inner.f(x, ps))), f.inner.x
end
function rewrite_autodiff_call(
    f::ComposedFunction{<:Base.Fix1{<:StatefulLuxLayer},F}
) where {F}
    return @closure((ps, x) -> f.outer.f(x, f.inner(ps))), f.outer.x
end
function rewrite_autodiff_call(f::Base.Fix1{<:StatefulLuxLayer})
    return @closure((ps, x) -> f.f(x, ps)), f.x
end

## Break ambiguity
for op in [
    ComposedFunction{<:StatefulLuxLayer,<:StatefulLuxLayer},
    ComposedFunction{<:Base.Fix1{<:StatefulLuxLayer},<:StatefulLuxLayer},
    ComposedFunction{<:StatefulLuxLayer,<:Base.Fix1{<:StatefulLuxLayer}},
    ComposedFunction{<:Base.Fix1{<:StatefulLuxLayer},<:Base.Fix1{<:StatefulLuxLayer}},
]
    @eval function rewrite_autodiff_call(::$op)
        return error(
            "Cannot rewrite ComposedFunction with StatefulLuxLayer as inner and outer \
             layers"
        )
    end
end

# gradient(...) inside an AD call
for fname in (:autodiff_gradient, :autodiff_gradient_no_custom_rrule)
    @eval $fname(grad_fn::G, f::F, x, y) where {G,F} = grad_fn(Base.Fix2(f, y), x)
end

function CRC.rrule(
    cfg::RuleConfig{>:HasReverseMode}, ::typeof(autodiff_gradient), grad_fn::G, f::F, x, y
) where {G,F}
    @static if !AUTOMATIC_NESTED_AD_SWITCHING
        return CRC.rrule_via_ad(cfg, autodiff_gradient_no_custom_rrule, grad_fn, f, x, y)
    end

    res = autodiff_gradient(grad_fn, f, x, y)
    ∇autodiff_gradient = @closure Δ -> begin
        (Δ isa NoTangent || Δ isa ZeroTangent) && return ntuple(Returns(NoTangent()), 5)

        # For Zygote and such which return a tuple
        (res isa Tuple || Δ isa Tuple) && (Δ = only(Δ))
        Δ = Utils.recursive_unthunk(Δ)
        ∂x, ∂y = forwarddiff_jvp(@closure((x, y) -> grad_fn(f, x, y)), x, Δ, y)
        return NoTangent(), NoTangent(), NoTangent(), ∂x, ∂y
    end

    return res, ∇autodiff_gradient
end

# pullback(...) inside an AD call
for fname in (:autodiff_pullback, :autodiff_pullback_no_custom_rrule)
    @eval function $fname(pullback_fn::P, f::F, x, y, u) where {P,F}
        return only(last(pullback_fn(Base.Fix2(f, y), x))(u))
    end
end

function CRC.rrule(
    cfg::RuleConfig{>:HasReverseMode}, ::typeof(autodiff_pullback), pb_f::P, f::F, x, y, u
) where {P,F}
    @static if !AUTOMATIC_NESTED_AD_SWITCHING
        return CRC.rrule_via_ad(cfg, autodiff_pullback_no_custom_rrule, pb_f, f, x, y, u)
    end

    res = autodiff_pullback(pb_f, f, x, y, u)
    ∇autodiff_pullback = let pb_f = pb_f, f = f, x = x, y = y, u = u, res = res
        Δ -> begin
            (Δ isa NoTangent || Δ isa ZeroTangent) && return ntuple(Returns(NoTangent()), 6)

            # For Zygote and such which return a tuple
            (res isa Tuple || Δ isa Tuple) && (Δ = only(Δ))
            Δ = Utils.recursive_unthunk(Δ)
            ∂x, ∂y = forwarddiff_jvp(x, Δ, y) do x_dual, y_
                return last(pb_f(f, x_dual, y_))(u)
            end
            return (NoTangent(), NoTangent(), NoTangent(), ∂x, ∂y, NoTangent())
        end
    end

    return res, ∇autodiff_pullback
end

# jacobian(...) inside an AD call
for fname in (:autodiff_jacobian, :autodiff_jacobian_no_custom_rrule)
    @eval function $fname(jac_fn::J, grad_fn::G, f::F, x::AbstractArray, y) where {J,G,F}
        return jac_fn(Base.Fix2(f, y), x)
    end
end

function CRC.rrule(
    cfg::RuleConfig{>:HasReverseMode},
    ::typeof(autodiff_jacobian),
    jac_fn::J,
    grad_fn::G,
    f::F,
    x::AbstractArray,
    y,
) where {J,G,F}
    @static if !AUTOMATIC_NESTED_AD_SWITCHING
        return CRC.rrule_via_ad(
            cfg, autodiff_jacobian_no_custom_rrule, jac_fn, grad_fn, f, x, y
        )
    end

    res = autodiff_jacobian(jac_fn, grad_fn, f, x, y)
    ∇autodiff_jacobian = let res = res, grad_fn = grad_fn, f = f, x = x, y = y
        Δ -> begin
            (Δ isa NoTangent || Δ isa ZeroTangent) && return ntuple(Returns(NoTangent()), 6)

            # For Zygote and such which return a tuple
            (res isa Tuple || Δ isa Tuple) && (Δ = only(Δ))
            Δ = Utils.recursive_unthunk(Δ)
            Δ = compactify_if_structured_matrix(res isa Tuple ? only(res) : res, Δ)

            inner_grad_fn = @closure(i -> sum ∘ Base.Fix2(getindex, i:i) ∘ vec ∘ f)
            map_fn = @closure i -> begin
                Δᵢ = batched_row(Δ, i)
                __f = let fn = inner_grad_fn(i)
                    (x, y) -> grad_fn(fn, x, y)
                end
                return forwarddiff_jvp(__f, x, Δᵢ, y)
            end

            # FIXME: threading on CUDA cause unexpected errors on the first run to CUDNN
            #        when doing a algorithm lookup
            ∂x, ∂y = if get_device_type(x) <: CPUDevice
                tasks = map(i -> Threads.@spawn(map_fn(i)), 1:numrows(Δ))
                mapreduce(fetch, Lux.recursive_add!!, tasks)
            else
                mapreduce(map_fn, Lux.recursive_add!!, 1:numrows(Δ))
            end

            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂x, ∂y
        end
    end

    return res, ∇autodiff_jacobian
end

function forwarddiff_gradient end
function forwarddiff_jacobian end
