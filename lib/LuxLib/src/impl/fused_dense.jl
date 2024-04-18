# Reference implmentation to verify correctness
function __generic_dense_bias_activation(act::F, weight::AbstractMatrix, x::AbstractMatrix,
        bias::Union{Nothing, AbstractVector}) where {F}
    y = weight * x
    bias === nothing && return @. act(y)
    return @. act(y + bias)
end

@inline function __get_concrete_fdba_output_eltype(
        act::F, ::AbstractMatrix{Tw}, ::AbstractMatrix{Tx},
        b::Union{Nothing, <:AbstractVector}) where {F, Tw, Tx}
    if b === nothing
        Ty = promote_type(Tw, Tx)
        Tact = Core.Compiler.return_type(act, Tuple{Ty})
        return isconcretetype(Tact) ? promote_type(Ty, Tact) : Ty
    end
    Ty = promote_type(Tw, Tx, eltype(b))
    Tact = Core.Compiler.return_type(act, Tuple{Ty})
    return isconcretetype(Tact) ? promote_type(Ty, Tact) : Ty
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

function __fused_dense_bias_activation_impl(
        act::F, weight::AbstractMatrix, x::AbstractMatrix, ::Nothing) where {F}
    y = similar(weight, __get_concrete_fdba_output_eltype(act, weight, x, nothing),
        size(weight, 1), size(x, 2))
    mul!(y, weight, x)
    @. y = act(y)
    return y
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__fused_dense_bias_activation_impl), act::F,
        weight::AbstractMatrix, x::AbstractMatrix, b::Nothing) where {F}
    T = __get_concrete_fdba_output_eltype(act, weight, x, b)
    y = similar(weight, T, size(weight, 1), size(x, 2))
    mul!(y, weight, x)

    # Case I: Activation Function doesn't require caching the intermediate value
    # See https://github.com/FluxML/NNlib.jl/blob/d85402aa39ddc6386d194e0dad88ab2e514ec5ea/src/bias_act.jl#L59-L60
    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        @. y = act(y)
        ∇__fused_dense_bias_activation_impl_no_cached = @closure Δ -> begin
            ∂y = only_derivative.(y, act, NotaNumber()) .* CRC.unthunk(Δ)
            ∂x = weight' * ∂y
            ∂w = ∂y * x'
            return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, CRC.NoTangent()
        end
        return y, ∇__fused_dense_bias_activation_impl_no_cached
    end

    # Case II: We can't overwrite `y` directly, but we can use the direct ChainRules
    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        z = @. act(y)
        ∇__fused_dense_bias_activation_impl_cached_crc = @closure Δ -> begin
            ∂y = only_derivative.(z, act, y) .* CRC.unthunk(Δ)
            ∂x = weight' * ∂y
            ∂w = ∂y * x'
            return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, CRC.NoTangent()
        end
        return z, ∇__fused_dense_bias_activation_impl_cached_crc
    end

    # Case III: Activation Function requires caching the intermediate value
    z, pb_f = CRC.rrule_via_ad(cfg, @closure(y->@.(act(y))), y)
    ∇__fused_dense_bias_activation_impl_cached = @closure Δ -> begin
        _, ∂y = pb_f(Δ)
        ∂x = weight' * ∂y
        ∂w = ∂y * x'
        return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, CRC.NoTangent()
    end
    return z, ∇__fused_dense_bias_activation_impl_cached
end

function __fused_dense_bias_activation_impl(
        ::typeof(identity), weight::AbstractMatrix, x::AbstractMatrix, b::AbstractVector)
    y = similar(weight, __get_concrete_fdba_output_eltype(identity, weight, x, b),
        size(weight, 1), size(x, 2))
    mul!(y, weight, x)
    @. y += b
    return y
end

function CRC.rrule(::typeof(__fused_dense_bias_activation_impl), ::typeof(identity),
        weight::AbstractMatrix, x::AbstractMatrix, b::AbstractVector)
    y = __fused_dense_bias_activation_impl(identity, weight, x, b)
    ∇__fused_dense_bias_activation_impl = @closure Δ -> begin
        ∂y = CRC.unthunk(Δ)
        ∂b = similar(b)
        sum!(∂b, ∂y)
        ∂x = weight' * ∂y
        ∂w = ∂y * x'
        return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b
    end
    return y, ∇__fused_dense_bias_activation_impl
end

function __fused_dense_bias_activation_impl(
        act::F, weight::AbstractMatrix, x::AbstractMatrix, b::AbstractVector) where {F}
    y = similar(weight, __get_concrete_fdba_output_eltype(act, weight, x, b),
        size(weight, 1), size(x, 2))
    mul!(y, weight, x)
    @. y = act(y + b)
    return y
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__fused_dense_bias_activation_impl), act::F,
        weight::AbstractMatrix, x::AbstractMatrix, b::AbstractVector) where {F}
    T = __get_concrete_fdba_output_eltype(act, weight, x, b)
    y = similar(weight, T, size(weight, 1), size(x, 2))
    mul!(y, weight, x)

    # Case I: Activation Function doesn't require caching the intermediate value
    # See https://github.com/FluxML/NNlib.jl/blob/d85402aa39ddc6386d194e0dad88ab2e514ec5ea/src/bias_act.jl#L59-L60
    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        @. y = act(y + b)
        ∇__fused_dense_bias_activation_impl_no_cached = @closure Δ -> begin
            ∂y = only_derivative.(y, act, NotaNumber()) .* CRC.unthunk(Δ)
            ∂b = similar(b)
            sum!(∂b, ∂y)
            ∂x = weight' * ∂y
            ∂w = ∂y * x'
            return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b
        end
        return y, ∇__fused_dense_bias_activation_impl_no_cached
    end

    # Case II: We can't overwrite `y` directly, but we can use the direct ChainRules
    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        @. y += b
        z = @. act(y)
        ∇__fused_dense_bias_activation_impl_cached_crc = @closure Δ -> begin
            ∂y = only_derivative.(z, act, y) .* CRC.unthunk(Δ)
            ∂b = similar(b)
            sum!(∂b, ∂y)
            ∂x = weight' * ∂y
            ∂w = ∂y * x'
            return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b
        end
        return z, ∇__fused_dense_bias_activation_impl_cached_crc
    end

    # Case III: Activation Function requires caching the intermediate value
    z, pb_f = CRC.rrule_via_ad(cfg, @closure((y, b)->@.(act(y + b))), y, b)
    ∇__fused_dense_bias_activation_impl_cached = @closure Δ -> begin
        _, ∂y, ∂b = pb_f(Δ)
        ∂x = weight' * ∂y
        ∂w = ∂y * x'
        return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b
    end
    return z, ∇__fused_dense_bias_activation_impl_cached
end
