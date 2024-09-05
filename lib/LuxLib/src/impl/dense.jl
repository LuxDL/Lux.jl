function cublasLt_fused_dense end # Defined in `LuxLibCUDAExt`
function cublasLt_fused_dense! end # Defined in `LuxLibCUDAExt`

function fused_dense(::typeof(identity), weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector})
    return matmuladd(weight, x, b)
end

function fused_dense(act::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Optional{<:AbstractVector}) where {F}
    return fused_dense(internal_operation_mode((weight, x, b)), act, weight, x, b)
end

function fused_dense(opmode::GenericBroadcastOp, act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    return bias_activation(act, matmul(opmode, weight, x), b)
end

@stable default_mode="disable" function fused_dense(
        opmode::AbstractInternalArrayOpMode, act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    y = similar(weight, concrete_bias_act_output_eltype(act, weight, x, b),
        size(weight, 1), size(x, 2))
    fused_dense!(y, opmode, act, weight, x, b)
    return y
end

function fused_dense!(y::AbstractMatrix, opmode::AbstractInternalArrayOpMode, act::F,
        weight::AbstractMatrix, x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    matmul!(y, opmode, weight, x)
    bias_activation!(y, opmode, act, y, b)
    return nothing
end

function fused_dense!(
        y::AbstractMatrix, ::GPUBroadcastOp{CUDADevice}, act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    cublasLt_fused_dense!(y, act, weight, x, b)
    return nothing
end

function CRC.rrule(cfg::CRC.RuleConfig{>:HasReverseMode}, ::typeof(fused_dense),
        opmode::AbstractInternalArrayOpMode, act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    T = concrete_bias_act_output_eltype(act, weight, x, b)
    𝒫weight, 𝒫x, 𝒫b = CRC.ProjectTo(weight), CRC.ProjectTo(x), CRC.ProjectTo(b)

    if unsafe_known(activation_intermediate_not_needed(act, T))
        y = fused_dense(opmode, act, weight, x, b)
        ∇fused_dense_no_intermediate = @closure Δ -> begin
            ∂y = ∇activation(CRC.unthunk(Δ), y, act, NotaNumber())
            ∂w, ∂x, ∂b = ∇matmul_bias(∂y, weight, x, b)
            return ∂∅, ∂∅, ∂∅, 𝒫weight(∂w), 𝒫x(∂x), 𝒫b(∂b)
        end
        return y, ∇fused_dense_no_intermediate
    end

    if unsafe_known(activation_has_rrule(act, T))
        y = matmuladd(weight, x, b)
        z = activation(opmode, act, y)
        ∇fused_dense_cached = @closure Δ -> begin
            ∂y = ∇activation(CRC.unthunk(Δ), z, act, y)
            ∂w, ∂x, ∂b = ∇matmul_bias(∂y, weight, x, b)
            return ∂∅, ∂∅, ∂∅, 𝒫weight(∂w), 𝒫x(∂x), 𝒫b(∂b)
        end
        return z, ∇fused_dense_cached
    end

    y = similar(weight, T, size(weight, 1), size(x, 2))
    matmul!(y, opmode, weight, x)
    z, ∇bias_activation = CRC.rrule_via_ad(cfg, bias_activation, act, y, b)
    ∇fused_dense_fallback = @closure Δ -> begin
        _, _, ∂y, ∂b = ∇bias_activation(Δ)
        ∂w, ∂x, _ = ∇matmul_bias(∂y, ∂b, weight, x, b)
        return ∂∅, ∂∅, ∂∅, 𝒫weight(∂w), 𝒫x(∂x), 𝒫b(∂b)
    end
    return z, ∇fused_dense_fallback
end

## Special Reverse Pass for gelu activation. All other cases, we don't need special handling
function CRC.rrule(
        ::typeof(fused_dense), ::GPUBroadcastOp{CUDADevice}, ::typeof(NNlib.gelu),
        weight::AbstractMatrix, x::AbstractMatrix, b::Optional{<:AbstractVector})
    z, y = cublasLt_fused_dense(NNlib.gelu, weight, x, b, True())
    𝒫weight, 𝒫x, 𝒫b = CRC.ProjectTo(weight), CRC.ProjectTo(x), CRC.ProjectTo(b)

    ∇fused_dense = @closure Δ -> begin
        ∂y = ∇activation(CRC.unthunk(Δ), z, NNlib.gelu, y)
        ∂w, ∂x, ∂b = ∇matmul_bias(∂y, weight, x, b)
        return ∂∅, ∂∅, ∂∅, 𝒫weight(∂w), 𝒫x(∂x), 𝒫b(∂b)
    end

    return z, ∇fused_dense
end

# TODO: We can optimize these a bit further by checking for cases where the forward pass
#       is not needed. We skip such optimizations for now
function EnzymeRules.augmented_primal(cfg, ::EnzymeCore.Const{typeof(fused_dense!)},
        ::Type{EnzymeCore.Const{Nothing}}, y::EnzymeCore.Annotation{<:AbstractMatrix},
        opmode::EnzymeCore.Const{<:AbstractInternalArrayOpMode}, act::EnzymeCore.Const,
        weight::EnzymeCore.Annotation{<:AbstractMatrix},
        x::EnzymeCore.Annotation{<:AbstractMatrix},
        b::EnzymeCore.Annotation{<:Optional{<:AbstractVector}})

    # NOTE: Here we are using the ChainRulesCore rrules if they are defined for simplicity
    all_const = weight isa EnzymeCore.Const && b isa EnzymeCore.Const &&
                x isa EnzymeCore.Const
    intermediate_not_needed = unsafe_known(activation_intermediate_not_needed(
        act.val, eltype(y.val))) || all_const

    weight_cache = EnzymeRules.overwritten(cfg)[5] && !(x isa EnzymeCore.Const) &&
                   !(y isa EnzymeCore.Const) ? copy(weight.val) : nothing
    x_cache = EnzymeRules.overwritten(cfg)[6] && !(weight isa EnzymeCore.Const) &&
              !(y isa EnzymeCore.Const) ? copy(x.val) : nothing

    case_specific_cache = if act.val === NNlib.gelu &&
                             opmode.val isa GPUBroadcastOp{CUDADevice}
        tmp = similar(y.val)
        cublasLt_fused_dense!(y.val, act.val, weight.val, x.val, b.val, tmp)
        (1, tmp)
    elseif intermediate_not_needed
        fused_dense!(y.val, opmode.val, act.val, weight.val, x.val, b.val)
        (1, NotaNumber())
    elseif unsafe_known(activation_has_rrule(act.val, eltype(y.val)))
        tmp = matmuladd(weight.val, x.val, b.val)
        activation!(y.val, opmode.val, act.val, tmp)
        (1, tmp)
    else
        # TODO: Here for performance we might want to fuse the bias and activation together.
        #       We skip this optimization for now
        if b.val !== nothing
            matmuladd!(y.val, opmode.val, weight.val, x.val, b.val)
        else
            matmul!(y.val, opmode.val, weight.val, x.val)
        end
        tmp = zero.(y.val)
        EnzymeCore.autodiff(EnzymeCore.Forward, EnzymeCore.Const(activation!),
            EnzymeCore.Duplicated(y.val, tmp), opmode, act,
            EnzymeCore.Duplicated(y.val, one.(y.val)))
        (2, tmp)
    end

    cache = (case_specific_cache, weight_cache, x_cache)

    return EnzymeRules.AugmentedReturn(nothing, nothing, cache)
end

function EnzymeRules.reverse(cfg, ::EnzymeCore.Const{typeof(fused_dense!)},
        ::Type{EnzymeCore.Const{Nothing}}, cache, y::EnzymeCore.Annotation{<:AbstractMatrix},
        opmode::EnzymeCore.Const{<:AbstractInternalArrayOpMode}, act::EnzymeCore.Const,
        weight::EnzymeCore.Annotation{<:AbstractMatrix},
        x::EnzymeCore.Annotation{<:AbstractMatrix},
        b::EnzymeCore.Annotation{<:Optional{<:AbstractVector}})
    case_specific_cache, weight_cache, x_cache = cache

    (case, tmp) = case_specific_cache

    if !(x isa EnzymeCore.Const) && !(y isa EnzymeCore.Const)
        if !EnzymeRules.overwritten(cfg)[5]
            weight_cache = weight.val
        end
    end

    if !(weight isa EnzymeCore.Const) && !(y isa EnzymeCore.Const)
        if !EnzymeRules.overwritten(cfg)[6]
            x_cache = x.val
        end
    end

    ∂ys = y.dval
    ∂xs = x isa EnzymeCore.Const ? ∂ys : x.dval
    ∂ws = weight isa EnzymeCore.Const ? ∂ys : weight.dval
    ∂bs = b isa EnzymeCore.Const ? ∂ys : b.dval

    if EnzymeRules.width(cfg) == 1
        ∂ys = (∂ys,)
        ∂xs = (∂xs,)
        ∂ws = (∂ws,)
        ∂bs = (∂bs,)
    end

    for (∂y, ∂w, ∂x, ∂b) in zip(∂ys, ∂ws, ∂xs, ∂bs)
        if !(y isa EnzymeCore.Const) && ∂y !== y.val
            # Compute preactivation gradients
            ∂pre_act = if case == 1
                ∇activation(∂y, y.val, act.val, tmp)
            elseif case == 2
                ∂y .* tmp
            else
                error("Unknown case: $case. This should not happen, open an issue.")
            end

            if !(b isa EnzymeCore.Const) && ∂b !== b.val
                sum!(∂b, ∂pre_act)
            end

            if !(weight isa EnzymeCore.Const) && ∂w !== weight.val
                # TODO: we don't use our faster matmul here since we lack the 5 arg version
                mul!(∂w, ∂pre_act, x_cache', true, true)
            end

            if !(x isa EnzymeCore.Const) && ∂x !== x.val
                # TODO: we don't use our faster matmul here since we lack the 5 arg version
                mul!(∂x, weight_cache', ∂pre_act, true, true)
            end

            ∂y .= 0
        end
    end

    return ntuple(Returns(nothing), 6)
end

∇matmul_bias(∂y, weight, x, bias) = ∇matmul_bias(∂y, ∇bias_add(bias, ∂y), weight, x, bias)
∇matmul_bias(∂y, ∂b, weight, x, _) = matmul(∂y, x'), matmul(weight', ∂y), ∂b
