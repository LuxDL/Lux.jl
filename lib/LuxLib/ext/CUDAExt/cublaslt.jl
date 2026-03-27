const TransOrAdjOrRegCuMatrix{T} = Union{
    Transpose{T,<:CuMatrix{T}},Adjoint{T,<:CuMatrix{T}},CuMatrix{T}
}

prepare_transposed(arr) = arr, false
prepare_transposed(arr::Union{Transpose,Adjoint}) = parent(arr), true

# fail if dispatch is not available
cublaslt_matmul_fused!(args...) = -1

function cublaslt_matmul_fused!(
    @nospecialize(y::TransOrAdjOrRegCuMatrix{<:Real}),
    σ::F,
    @nospecialize(w::TransOrAdjOrRegCuMatrix{<:Real}),
    @nospecialize(x::TransOrAdjOrRegCuMatrix{<:Real}),
    b::Optional{<:CuVector{<:Real}},
    aux::Optional{<:CuMatrix{<:Real}}=nothing,
) where {F}
    y, transy = prepare_transposed(y)
    x, transx = prepare_transposed(x)
    w, transw = prepare_transposed(w)
    return cublaslt_matmul_fused!(transy, y, σ, transw, w, transx, x, b, aux)
end

function cublaslt_matmul_fused!(
    transy::Bool,
    @nospecialize(y::CuMatrix{yT}),
    σ::F,
    transw::Bool,
    @nospecialize(w::CuMatrix{wT}),
    transx::Bool,
    @nospecialize(x::CuMatrix{xT}),
    b::Optional{<:CuVector},
    aux::Optional{<:CuMatrix},
) where {F,yT,wT,xT}
    bT = b === nothing ? Bool : eltype(b)
    auxT = aux === nothing ? Bool : eltype(aux)
    # cuBLASLt will give wrong results if the types are not correct. As a hack we are going
    # to promote the types to the largest type
    wxT = promote_type(wT, xT, bT, auxT)
    @warn "Mixed Precision Inputs received for `weight`: $(typeof(w)) and `x`: \
           $(typeof(x)). Promoting to $(wxT)." maxlog = 1
    return cublaslt_matmul_fused!(
        transy,
        y,
        σ,
        transw,
        ofeltype_array(wxT, w),
        transx,
        ofeltype_array(wxT, x),
        ofeltype_array(wxT, b),
        ofeltype_array(wxT, aux),
    )
end

function lt_handle_ctor(ctx)
    CUDA.context!(ctx) do
        @info "Creating cuBLASLt handle for context $(ctx)"
        handle = Ref{CUBLAS.cublasLtHandle_t}()
        CUBLAS.cublasLtCreate(handle)
        return handle[]
    end
end

function lt_handle_dtor(ctx, handle)
    CUDA.context!(ctx; skip_destroyed=true) do
        @info "Destroying cuBLASLt handle $(handle) for context $(ctx)"
        CUBLAS.cublasLtDestroy(handle)
    end
end

const idle_lt_handles = CUDA.APIUtils.HandleCache{CUDA.CuContext,CUBLAS.cublasLtHandle_t}(
    lt_handle_ctor, lt_handle_dtor
)

mutable struct CuBLASLtMatmulDesc
    desc::CUBLAS.cublasLtMatmulDesc_t

    function CuBLASLtMatmulDesc(args...)
        desc_ref = Ref{CUBLAS.cublasLtMatmulDesc_t}()
        CUBLAS.cublasLtMatmulDescCreate(desc_ref, args...)
        return finalizer(
            CUBLAS.cublasLtMatmulDescDestroy ∘ Base.Fix2(getproperty, :desc),
            new(desc_ref[]),
        )
    end
end
Base.getindex(x::CuBLASLtMatmulDesc) = x.desc
Base.unsafe_convert(::Type{CUBLAS.cublasLtMatmulDesc_t}, x::CuBLASLtMatmulDesc) = x.desc

const _CUBLASLT_MATMUL_DESC_ATTRIBUTES = Dict(
    :transa => (CUBLAS.CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS.cublasOperation_t),
    :transb => (CUBLAS.CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS.cublasOperation_t),
    :transc => (CUBLAS.CUBLASLT_MATMUL_DESC_TRANSC, CUBLAS.cublasOperation_t),
    :epilogue => (CUBLAS.CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLAS.cublasLtEpilogue_t),
    :bias_pointer => (CUBLAS.CUBLASLT_MATMUL_DESC_BIAS_POINTER, CuPtr{Cvoid}),
    :epilogue_aux_pointer =>
        (CUBLAS.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, CuPtr{Cvoid}),
    :epilogue_aux_ld => (CUBLAS.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, Csize_t),
)

function Base.setproperty!(desc::CuBLASLtMatmulDesc, name::Symbol, value)
    if haskey(_CUBLASLT_MATMUL_DESC_ATTRIBUTES, name)
        attr, T = _CUBLASLT_MATMUL_DESC_ATTRIBUTES[name]
        CUBLAS.cublasLtMatmulDescSetAttribute(desc.desc, attr, Ref{T}(value), sizeof(T))
        return value
    else
        return setfield!(desc, name, value)
    end
end

mutable struct CuBLASLtMatrixLayout
    layout::CUBLAS.cublasLtMatrixLayout_t

    function CuBLASLtMatrixLayout(args...)
        layout_ref = Ref{CUBLAS.cublasLtMatrixLayout_t}()
        CUBLAS.cublasLtMatrixLayoutCreate(layout_ref, args...)
        return finalizer(
            CUBLAS.cublasLtMatrixLayoutDestroy ∘ Base.Fix2(getproperty, :layout),
            new(layout_ref[]),
        )
    end
end
Base.getindex(x::CuBLASLtMatrixLayout) = x.layout
function Base.unsafe_convert(::Type{CUBLAS.cublasLtMatrixLayout_t}, x::CuBLASLtMatrixLayout)
    return x.layout
end

mutable struct CuBLASLtMatmulPreference
    pref::CUBLAS.cublasLtMatmulPreference_t

    function CuBLASLtMatmulPreference()
        pref_ref = Ref{CUBLAS.cublasLtMatmulPreference_t}()
        CUBLAS.cublasLtMatmulPreferenceCreate(pref_ref)
        return finalizer(
            CUBLAS.cublasLtMatmulPreferenceDestroy ∘ Base.Fix2(getproperty, :pref),
            new(pref_ref[]),
        )
    end
end
Base.getindex(x::CuBLASLtMatmulPreference) = x.pref
function Base.unsafe_convert(
    ::Type{CUBLAS.cublasLtMatmulPreference_t}, x::CuBLASLtMatmulPreference
)
    return x.pref
end

function get_cublaslt_handle()
    cuda = CUDA.active_state()

    # every task maintains library state per set of devices
    LibraryState = @NamedTuple{handle::CUBLAS.cublasLtHandle_t}
    states = get!(task_local_storage(), :CUBLASLt) do
        Dict{CUDA.CuContext,LibraryState}()
    end::Dict{CUDA.CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_lt_handles, cuda.context)
        finalizer(current_task()) do _
            push!(idle_lt_handles, cuda.context, new_handle)
        end
        return (; handle=new_handle)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.handle
end

# TODO: use https://docs.nvidia.com/cuda/cublas/#cublasltmatmul for a more robust
#       computeType mapping. Currently no one uses Lux with weird type combinations so we
#       don't need to worry about it too much and just fall back to the generic
#       implementation
# Returns: 0 if successful, -1 if unsuccessful
function cublaslt_matmul_fused!(
    transy::Bool,
    @nospecialize(y::CuMatrix{yT}),
    σ::F,
    transw::Bool,
    @nospecialize(w::CuMatrix{wxT}),
    transx::Bool,
    @nospecialize(x::CuMatrix{wxT}),
    b::Optional{<:CuVector},
    aux::Optional{<:CuMatrix},
) where {F,yT,wxT}
    m = size(y, 1)
    n = size(y, 2)
    k = size(w, 2)

    if b === nothing
        size(y, transy ? 2 : 1) == size(w, transw ? 2 : 1) ||
            throw(DimensionMismatch("size(y) = $(size(y)), size(w) = $(size(w))"))
    else
        size(y, transy ? 2 : 1) == size(w, transw ? 2 : 1) == size(b, 1) || throw(
            DimensionMismatch(
                "size(y) = $(size(y)), size(w) = $(size(w)), size(b) = $(size(b))"
            ),
        )
    end
    size(x, transx ? 2 : 1) == size(w, transw ? 1 : 2) ||
        throw(DimensionMismatch("size(x) = $(size(x)), size(w) = $(size(w))"))

    ## While querying the compute type, promote the types
    computeType = CUBLAS.gemmExComputeType(wxT, wxT, yT, m, k, n)
    computeType === nothing && return -1
    dataType = convert(CUDA.cudaDataType, yT)

    # Create the operation descriptor
    operationDesc = CuBLASLtMatmulDesc(computeType, dataType)

    # Set the matrix descriptors
    operationDesc.transa = transw ? CUBLAS.CUBLAS_OP_T : CUBLAS.CUBLAS_OP_N
    operationDesc.transb = transx ? CUBLAS.CUBLAS_OP_T : CUBLAS.CUBLAS_OP_N
    operationDesc.transc = transy ? CUBLAS.CUBLAS_OP_T : CUBLAS.CUBLAS_OP_N

    # Decide on the epilogue
    epilogue, activation_fused = epilogue_act(σ, b, aux)
    operationDesc.epilogue = epilogue

    # We have a bias so set the bias pointer
    if b !== nothing
        operationDesc.bias_pointer = pointer(b)
    end

    if aux !== nothing
        operationDesc.epilogue_aux_pointer = pointer(aux)
        operationDesc.epilogue_aux_ld = max(1, stride(aux, 2))
    end

    # Create the matrix layouts
    wdesc = CuBLASLtMatrixLayout(
        convert(CUDA.cudaDataType, wxT), m, k, max(1, stride(w, 2))
    )
    xdesc = CuBLASLtMatrixLayout(
        convert(CUDA.cudaDataType, wxT), size(x)..., max(1, stride(x, 2))
    )
    ydesc = CuBLASLtMatrixLayout(convert(CUDA.cudaDataType, yT), m, n, max(1, stride(y, 2)))

    # Create the preference. we can customize this but we will stick to the defaults
    preference = CuBLASLtMatmulPreference()

    # Create the light handle
    lthandle = get_cublaslt_handle()

    # Search for the best algorithm
    heuristic = Ref{CUBLAS.cublasLtMatmulHeuristicResult_t}()
    returnedResults = Ref{Cint}(0)
    CUBLAS.cublasLtMatmulAlgoGetHeuristic(
        lthandle,
        operationDesc,
        wdesc,
        xdesc,
        ydesc,
        ydesc,
        preference,
        1,
        heuristic,
        returnedResults,
    )

    returnedResults[] == 0 && return -1

    CUBLAS.cublasLtMatmul(
        lthandle,
        operationDesc,
        Ref{wxT}(1),
        w,
        wdesc,
        x,
        xdesc,
        Ref{yT}(0),
        y,
        ydesc,
        y,
        ydesc,
        Ref(heuristic[].algo),
        CUDA.CU_NULL,
        0,
        CUDA.stream(),
    )

    !activation_fused && (@. y = σ(y))

    return 0
end

function epilogue_act(f::F, b, aux) where {F}
    if f === identity
        @assert aux === nothing "`aux` must be `nothing` for `identity` activation."
        b === nothing && return CUBLAS.CUBLASLT_EPILOGUE_DEFAULT, true
        return CUBLAS.CUBLASLT_EPILOGUE_BIAS, true
    elseif f === NNlib.relu
        if b === nothing
            aux === nothing && return CUBLAS.CUBLASLT_EPILOGUE_RELU, true
            return CUBLAS.CUBLASLT_EPILOGUE_RELU_AUX, true
        else
            aux === nothing && return CUBLAS.CUBLASLT_EPILOGUE_RELU_BIAS, true
            return CUBLAS.CUBLASLT_EPILOGUE_RELU_AUX_BIAS, true
        end
    elseif f === NNlib.gelu
        if b === nothing
            aux === nothing && return CUBLAS.CUBLASLT_EPILOGUE_GELU, true
            return CUBLAS.CUBLASLT_EPILOGUE_GELU_AUX, true
        else
            aux === nothing && return CUBLAS.CUBLASLT_EPILOGUE_GELU_BIAS, true
            return CUBLAS.CUBLASLT_EPILOGUE_GELU_AUX_BIAS, true
        end
    else
        @assert aux === nothing "`aux` must be `nothing` for `$(f)` activation."
        b === nothing && return CUBLAS.CUBLASLT_EPILOGUE_DEFAULT, false
        return CUBLAS.CUBLASLT_EPILOGUE_BIAS, false
    end
end

len(x) = length(x)
len(::Nothing) = nothing

function LuxLib.Impl.cublasLt_fused_dense(
    act::F,
    weight::AbstractMatrix,
    x::AbstractMatrix,
    b::Optional{<:AbstractVector},
    ::False,
) where {F}
    z = similar(
        x, LuxLib.concrete_fba_output_eltype(act, weight, x, b), size(weight, 1), size(x, 2)
    )
    LuxLib.cublasLt_fused_dense!(z, act, weight, x, b)
    return z, nothing
end

function LuxLib.Impl.cublasLt_fused_dense(
    act::F, weight::AbstractMatrix, x::AbstractMatrix, b::Optional{<:AbstractVector}, ::True
) where {F}
    z = similar(
        x, LuxLib.concrete_fba_output_eltype(act, weight, x, b), size(weight, 1), size(x, 2)
    )
    y = similar(z)
    LuxLib.cublasLt_fused_dense!(z, act, weight, x, b, y)
    return z, y
end

function LuxLib.Impl.cublasLt_fused_dense!(
    z::AbstractMatrix,
    act::F,
    weight::AbstractMatrix,
    x::AbstractMatrix,
    b::Optional{<:AbstractVector},
    y::Optional{<:AbstractMatrix}=nothing,
) where {F}
    if hasmethod(
        cublaslt_matmul_fused!,
        (typeof(z), typeof(act), typeof(weight), typeof(x), typeof(b), typeof(y)),
    )
        retcode = cublaslt_matmul_fused!(z, act, weight, x, b, y)
        retcode == 0 && return nothing
        warn_msg = LazyString(
            "cuBLASLt failed for the given inputs ",
            act,
            ", ",
            typeof(weight),
            " [",
            size(weight),
            "], ",
            typeof(x),
            " [",
            size(x),
            "], ",
            typeof(b),
            " [",
            len(b),
            "]. Falling back to generic implementation.",
        )
        @warn warn_msg maxlog = 1
    else
        @debug "cuBLASLt not available. Falling back to generic implementation." maxlog = 1
    end
    # Generic fallback
    if y === nothing
        LinearAlgebra.mul!(z, weight, x)
        if b === nothing
            broadcast!(act, z, z)
        else
            broadcast!(act ∘ +, z, z, reshape(b, :, 1))
        end
        return nothing
    else
        LinearAlgebra.mul!(y, weight, x)
        if b !== nothing
            broadcast!(+, y, y, reshape(b, :, 1))
        end
        broadcast!(act, z, y)
        return nothing
    end
end
