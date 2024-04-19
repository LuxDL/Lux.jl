function __generic_dense_bias_activation(::typeof(identity), weight::AbstractMatrix,
        x::AbstractMatrix, bias::Union{Nothing, AbstractVector})
    y = weight * x
    bias === nothing && return y
    return @. y + bias
end

function __generic_dense_bias_activation(act::F, weight::AbstractMatrix, x::AbstractMatrix,
        bias::Union{Nothing, AbstractVector}) where {F}
    y = weight * x
    bias === nothing && return @. act(y)
    return @. act(y + bias)
end

# Why are we catching the implementation at this point and not in `bias_act!` like NNlib?
# Turns out NVIDIA has been shipping a bunch of fused kernels for a while now. We can
# potentially use those here to fuse all the operations into a single kernel.
#
# Currently that is not implemented, but once implemented integrating them into Lux will be
# trivial.
#
# Alternatively we have a native julia version in https://github.com/JuliaGPU/GemmKernels.jl
# that we can use to fuse the operations till we get CUBLASLt working.

@inline function __fused_dense_bias_activation_impl(
        ::typeof(identity), weight::AbstractMatrix, x::AbstractMatrix, ::Nothing)
    return weight * x
end

@inline function __fused_dense_bias_activation_impl(
        ::typeof(identity), weight::AbstractMatrix, x::AbstractMatrix, b::AbstractVector)
    y = similar(weight, __get_concrete_fba_output_eltype(identity, weight, x, b),
        size(weight, 1), size(x, 2))
    mul!(y, weight, x)
    @. y += b
    return y
end

@inline function __fused_dense_bias_activation_impl(
        act::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Union{Nothing, AbstractVector}) where {F}
    y = similar(weight, __get_concrete_fba_output_eltype(act, weight, x, nothing),
        size(weight, 1), size(x, 2))
    mul!(y, weight, x)
    if b === nothing
        @. y = act(y)
    else
        @. y = act(y + b)
    end
    return y
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__fused_dense_bias_activation_impl), act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Union{AbstractVector, Nothing}) where {F}
    T = __get_concrete_fba_output_eltype(act, weight, x, b)
    y = similar(weight, T, size(weight, 1), size(x, 2))
    mul!(y, weight, x)

    # Case I: Activation Function doesn't require caching the intermediate value
    # See https://github.com/FluxML/NNlib.jl/blob/d85402aa39ddc6386d194e0dad88ab2e514ec5ea/src/bias_act.jl#L59-L60
    if act === identity ||
       isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        y = __apply_bias_activation!!(act, y, b, Val(false))
        ∇__fused_dense_bias_activation_impl_no_cached = @closure Δ -> begin
            ∂y = act === identity ? CRC.unthunk(Δ) :
                 only_derivative.(y, act, NotaNumber()) .* CRC.unthunk(Δ)
            ∂b = __added_bias_gradient(b, ∂y)
            ∂x = weight' * ∂y
            ∂w = ∂y * x'
            return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b
        end
        return y, ∇__fused_dense_bias_activation_impl_no_cached
    end

    # Case II: We can't overwrite `y` directly, but we can use the direct ChainRules
    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        z, y = __apply_bias_activation!!(act, y, b, Val(true))
        ∇__fused_dense_bias_activation_impl_cached_crc = @closure Δ -> begin
            ∂y = only_derivative.(z, act, y) .* CRC.unthunk(Δ)
            ∂b = __added_bias_gradient(b, ∂y)
            ∂x = weight' * ∂y
            ∂w = ∂y * x'
            return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b
        end
        return z, ∇__fused_dense_bias_activation_impl_cached_crc
    end

    # Case III: Activation Function requires caching the intermediate value
    z, pb_f = CRC.rrule_via_ad(cfg, __apply_bias_activation, act, y, b)
    ∇__fused_dense_bias_activation_impl_cached = @closure Δ -> begin
        _, _, ∂y, ∂b = pb_f(Δ)
        ∂x = weight' * ∂y
        ∂w = ∂y * x'
        return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b
    end
    return z, ∇__fused_dense_bias_activation_impl_cached
end
