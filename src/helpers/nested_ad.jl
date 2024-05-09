function __forwarddiff_jvp end # Defined in ForwardDiff.jl extension

function __partials end  # DON'T REMOVE THIS (DEQs.jl is using it)
function __dualify end

#! format: off
const AD_CONVERTIBLE_FUNCTIONS = [
    # Input Gradient/Jacobian
    ComposedFunction{<:Any, <:StatefulLuxLayer},
    ComposedFunction{<:StatefulLuxLayer, <:Any},
    StatefulLuxLayer,
    # Parameter Gradient/Jacobian
    ComposedFunction{<:Any, <:Base.Fix1{<:StatefulLuxLayer}},
    ComposedFunction{<:Base.Fix1{<:StatefulLuxLayer}, <:Any},
    Base.Fix1{<:StatefulLuxLayer}
]
#! format: on

@inline function __rewrite_ad_call(f::F) where {F}
    # Input Gradient / Jacobian
    f isa ComposedFunction{<:Any, <:StatefulLuxLayer} && return f, f.inner.ps
    f isa ComposedFunction{<:StatefulLuxLayer, <:Any} &&
        return @closure((x, ps)->f.outer(f.inner(x), ps)), f.outer.ps
    f isa StatefulLuxLayer && return f, f.ps

    # Parameter Gradient / Jacobian
    f isa ComposedFunction{<:Any, <:Base.Fix1{<:StatefulLuxLayer}} &&
        return @closure((ps, x)->f.outer(f.inner.f(x, ps))), f.inner.x
    f isa ComposedFunction{<:Base.Fix1{<:StatefulLuxLayer}, <:Any} &&
        return @closure((ps, x)->f.outer.f(x, f.inner(ps))), f.outer.x
    f isa Base.Fix1{<:StatefulLuxLayer} && return @closure((ps, x)->f.f(x, ps)), f.x

    error("Unknown function type: $(typeof(f))")
end

# Nested Gradients
## Essentially computes the gradient of `f(x, y)` wrt x using the function `grad_fn`
## To compute the gradient of `f(x, y)` wrt y, just reorder the arguments with a wrapper
## over `f`
for fname in (:__internal_ad_gradient_call, :__internal_ad_gradient_call_no_custom_rrule)
    @eval @inline function $fname(grad_fn::G, f::F, x, y) where {G, F}
        return grad_fn(Base.Fix2(f, y), x)
    end
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__internal_ad_gradient_call), grad_fn::G, f::F, x, y) where {G, F}
    # Check if we can use the faster implementation
    if !Lux._is_extension_loaded(Val(:ForwardDiff)) || DISABLE_AUTOMATIC_NESTED_AD_SWITCH
        if !DISABLE_AUTOMATIC_NESTED_AD_SWITCH
            @warn "Load `ForwardDiff.jl` for better nested AD handling." maxlog=1
        end
        # Use the AD itself for whatever reason
        return CRC.rrule_via_ad(
            cfg, __internal_ad_gradient_call_no_custom_rrule, grad_fn, f, x, y)
    end

    res = __internal_ad_gradient_call(grad_fn, f, x, y)
    ∇internal_gradient_capture = @closure Δ_ -> begin
        (Δ_ isa CRC.NoTangent || Δ_ isa CRC.ZeroTangent) &&
            return ntuple(Returns(CRC.NoTangent()), 5)

        Δ = CRC.unthunk(Δ_)
        (res isa Tuple || Δ isa Tuple) && (Δ = only(Δ))  # For Zygote and such which return a tuple
        ∂x, ∂y = __forwarddiff_jvp(@closure((x, y)->grad_fn(f, x, y)), x, Δ, y)
        return CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y
    end

    return res, ∇internal_gradient_capture
end

# Nested Pullbacks
for fname in (:__internal_ad_pullback_call, :__internal_ad_pullback_call_no_custom_rrule)
    @eval @inline function $fname(pullback_fn::P, f::F, x, y, u) where {P, F}
        return only(last(pullback_fn(Base.Fix2(f, y), x))(u))
    end
end

function CRC.rrule(
        cfg::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(__internal_ad_pullback_call),
        pullback_fn::P, f::F, x, y, u) where {P, F}
    # Check if we can use the faster implementation
    if !Lux._is_extension_loaded(Val(:ForwardDiff)) || DISABLE_AUTOMATIC_NESTED_AD_SWITCH
        if !DISABLE_AUTOMATIC_NESTED_AD_SWITCH
            @warn "Load `ForwardDiff.jl` for better nested AD handling." maxlog=1
        end
        # Use the AD itself for whatever reason
        return CRC.rrule_via_ad(
            cfg, __internal_ad_pullback_call_no_custom_rrule, pullback_fn, f, x, y, u)
    end

    res = __internal_ad_pullback_call(pullback_fn, f, x, y, u)
    ∇internal_pullback_capture = let pullback_fn = pullback_fn,
        f = f,
        x = x,
        y = y,
        u = u,
        res = res

        Δ_ -> begin
            (Δ_ isa CRC.NoTangent || Δ_ isa CRC.ZeroTangent) &&
                return ntuple(Returns(CRC.NoTangent()), 6)

            Δ = CRC.unthunk(Δ_)
            (res isa Tuple || Δ isa Tuple) && (Δ = only(Δ))  # For Zygote and such which return a tuple
            ∂x, ∂y = __forwarddiff_jvp(x, Δ, y) do x_dual, y_
                return last(pullback_fn(f, x_dual, y_))(u)
            end
            return (
                CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
        end
    end

    return res, ∇internal_pullback_capture
end

# Nested Jacobians 
## `grad_fn` is not needed for the forward pass, we need it for the reverse pass HVP
for fname in (:__internal_ad_jacobian_call, :__internal_ad_jacobian_call_no_custom_rrule)
    @eval @inline function $fname(
            jac_fn::J, grad_fn::G, f::F, x::AbstractArray, y) where {J, G, F}
        return jac_fn(Base.Fix2(f, y), x)
    end
end

function CRC.rrule(
        cfg::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(__internal_ad_jacobian_call),
        jac_fn::J, grad_fn::G, f::F, x::AbstractArray, y) where {J, G, F}
    # Check if we can use the faster implementation
    if !Lux._is_extension_loaded(Val(:ForwardDiff)) || DISABLE_AUTOMATIC_NESTED_AD_SWITCH
        if !DISABLE_AUTOMATIC_NESTED_AD_SWITCH
            @warn "Load `ForwardDiff.jl` for better nested AD handling." maxlog=1
        end
        # Use the AD itself for whatever reason
        return CRC.rrule_via_ad(
            cfg, __internal_ad_jacobian_call_no_custom_rrule, jac_fn, grad_fn, f, x, y)
    end

    res = __internal_ad_jacobian_call(jac_fn, grad_fn, f, x, y)
    ∇internal_jacobian_capture = let res = res, grad_fn = grad_fn, f = f, x = x, y = y
        Δ_ -> begin
            (Δ_ isa CRC.NoTangent || Δ_ isa CRC.ZeroTangent) &&
                return ntuple(Returns(CRC.NoTangent()), 6)

            Δ = CRC.unthunk(Δ_)
            (res isa Tuple || Δ isa Tuple) && (Δ = only(Δ))  # For Zygote and such which return a tuple
            Δ = __compactify_if_structured_matrix(res isa Tuple ? only(res) : res, Δ)

            # TODO: Here we can potentially chunk the gradients for faster AD calls
            # TODO: Might be worth investigating if we can use Polyester or OhMyThreads
            #       to speed this up
            ∂x, ∂y = mapreduce(
                __internal_add, enumerate(__maybe_batched_eachrow(Δ))) do (i, Δᵢ)
                ∂xᵢ, ∂yᵢ = __forwarddiff_jvp(x, Δᵢ, y) do x, y
                    return grad_fn(sum ∘ Base.Fix2(getindex, i:i) ∘ vec ∘ f, x, y)
                end
                return ∂xᵢ, ∂yᵢ
            end

            return (
                CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y)
        end
    end

    return res, ∇internal_jacobian_capture
end

# Convert a structured Matrix to a General Matrix if it doesn't have fast scalar indexing
@inline function __compactify_if_structured_matrix(
        J::AbstractArray{T1, N}, Δ::AbstractArray{T2}) where {T1, T2, N}
    @assert N ∈ (2, 3) "Only 2D and 3D arrays are supported for compactifying."
    if !ArrayInterface.fast_scalar_indexing(J) && ArrayInterface.isstructured(Δ)
        J_ = similar(J)
        copyto!(J_, Δ)
        return J_
    end
    return reshape(Δ, size(J))
end

@inline __maybe_batched_eachrow(x::AbstractMatrix) = eachrow(x)
@inline function __maybe_batched_eachrow(x::AbstractArray{T, 3}) where {T}
    M, N, K = size(x)
    rowfn = @closure i -> begin
        k = (i - 1) ÷ M + 1
        i = mod1(i, M)
        y = similar(x, N * K)
        data = view(x, i, :, k)
        fill!(view(y, 1:(N * (K - 1))), zero(T))
        copyto!(view(y, (N * (k - 1) + 1):(N * k)), data)
        fill!(view(y, (N * k + 1):(N * K)), zero(T))
        return y
    end
    return (rowfn(i) for i in 1:(M * K))
end

## TODO: We can do an inplace addition but those are typically not giant bottlenecks
## This is used for adding up contributions to the gradient in extensions
@inline __internal_add(x::AbstractArray, y::AbstractArray) = x .+ y
@inline __internal_add(x::Tuple, y::Tuple) = map(__internal_add, x, y)
@inline function __internal_add(x::NamedTuple{F}, y::NamedTuple{F}) where {F}
    return NamedTuple{F}(map(__internal_add, values(x), values(y)))
end
@inline __internal_add(::Nothing, ::Nothing) = nothing
@inline __internal_add(x, y) = fmap(__internal_add, x, y)
