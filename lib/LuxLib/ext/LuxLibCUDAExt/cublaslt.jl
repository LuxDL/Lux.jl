const TransOrAdjOrRegStridedCuMatrix{T} = Union{Transpose{T, <:StridedCuMatrix{T}},
    Adjoint{T, <:StridedCuMatrix{T}}, StridedCuMatrix{T}}

function _cublaslt_matmul_fused!(
        @nospecialize(y::TransOrAdjOrRegStridedCuMatrix{<:Real}), σ::F,
        @nospecialize(w::TransOrAdjOrRegStridedCuMatrix{<:Real}),
        @nospecialize(x::TransOrAdjOrRegStridedCuMatrix{<:Real}),
        b::Optional{<:StridedCuVector{<:Real}},
        aux::Optional{<:StridedCuMatrix{<:Real}}=nothing) where {F}
    transy = y isa Transpose || y isa Adjoint
    transx = x isa Transpose || x isa Adjoint
    transw = w isa Transpose || x isa Adjoint
    return _cublaslt_matmul_fused!(
        transy, parent(y), σ, transw, parent(w), transx, parent(x), b, aux)
end

function _cublaslt_matmul_fused!(transy::Bool, @nospecialize(y::StridedCuMatrix{yT}), σ::F,
        transw::Bool, @nospecialize(w::StridedCuMatrix{wT}), transx::Bool,
        @nospecialize(x::StridedCuMatrix{xT}), b::Optional{<:StridedCuVector},
        aux::Optional{<:StridedCuMatrix}) where {F, yT, wT, xT}
    bT = b === nothing ? Bool : eltype(b)
    auxT = aux === nothing ? Bool : eltype(aux)
    # cuBLASLt will give wrong results if the types are not correct. As a hack we are going
    # to promote the types to the largest type
    wxT = promote_type(wT, xT, bT, auxT)
    @warn "Mixed Precision Inputs received for `weight`: $(typeof(w)) and `x`: \
           $(typeof(x)). Promoting to $(wxT)." maxlog=1
    return _cublaslt_matmul_fused!(transy, y, σ, transw, LuxLib._oftype_array(wxT, w),
        transx, LuxLib._oftype_array(wxT, x),
        LuxLib._oftype_array(wxT, b), LuxLib._oftype_array(wxT, aux))
end

# TODO: use https://docs.nvidia.com/cuda/cublas/#cublasltmatmul for a more robust
#       computeType mapping. Currently no one uses Lux with weird type combinations so we
#       don't need to worry about it too much and just fall back to the generic
#       implementation
# Returns: 0 if successful, -1 if unsuccessful
function _cublaslt_matmul_fused!(transy::Bool, @nospecialize(y::StridedCuMatrix{yT}), σ::F,
        transw::Bool, @nospecialize(w::StridedCuMatrix{wxT}), transx::Bool,
        @nospecialize(x::StridedCuMatrix{wxT}), b::Optional{<:StridedCuVector},
        aux::Optional{<:StridedCuMatrix}) where {F, yT, wxT}
    m = size(y, 1)
    n = size(y, 2)
    k = size(w, 2)

    if b === nothing
        size(y, transy ? 2 : 1) == size(w, transw ? 2 : 1) ||
            throw(DimensionMismatch("size(y) = $(size(y)), size(w) = $(size(w))"))
    else
        size(y, transy ? 2 : 1) == size(w, transw ? 2 : 1) == size(b, 1) ||
            throw(DimensionMismatch("size(y) = $(size(y)), size(w) = $(size(w)), size(b) = $(size(b))"))
    end
    size(x, transx ? 2 : 1) == size(w, transw ? 1 : 2) ||
        throw(DimensionMismatch("size(x) = $(size(x)), size(w) = $(size(w))"))

    # Create the operation descriptor
    operationDesc = Ref{CUBLAS.cublasLtMatmulDesc_t}()

    ## While querying the compute type, promote the types
    computeType = CUBLAS.gemmExComputeType(wxT, wxT, yT, m, k, n)
    computeType === nothing && return -1
    dataType = convert(CUDA.cudaDataType, yT)
    CUBLAS.cublasLtMatmulDescCreate(operationDesc, computeType, dataType)

    # Set the matrix descriptors
    ytransop = transy ? CUBLAS.CUBLAS_OP_T : CUBLAS.CUBLAS_OP_N
    wtransop = transw ? CUBLAS.CUBLAS_OP_T : CUBLAS.CUBLAS_OP_N
    xtransop = transx ? CUBLAS.CUBLAS_OP_T : CUBLAS.CUBLAS_OP_N

    CUBLAS.cublasLtMatmulDescSetAttribute(
        operationDesc[], CUBLAS.CUBLASLT_MATMUL_DESC_TRANSA,
        Ref{CUBLAS.cublasOperation_t}(wtransop), sizeof(wtransop))
    CUBLAS.cublasLtMatmulDescSetAttribute(
        operationDesc[], CUBLAS.CUBLASLT_MATMUL_DESC_TRANSB,
        Ref{CUBLAS.cublasOperation_t}(xtransop), sizeof(xtransop))
    CUBLAS.cublasLtMatmulDescSetAttribute(
        operationDesc[], CUBLAS.CUBLASLT_MATMUL_DESC_TRANSC,
        Ref{CUBLAS.cublasOperation_t}(ytransop), sizeof(ytransop))

    # Decide on the epilogue
    epilogue, activation_fused = __epilogue_act(σ, b, aux)
    CUBLAS.cublasLtMatmulDescSetAttribute(
        operationDesc[], CUBLAS.CUBLASLT_MATMUL_DESC_EPILOGUE,
        Ref{CUBLAS.cublasLtEpilogue_t}(epilogue), sizeof(epilogue))

    # We have a bias so set the bias pointer
    if b !== nothing
        bias_ptr = Ref{CuPtr{Cvoid}}(pointer(b))
        CUBLAS.cublasLtMatmulDescSetAttribute(
            operationDesc[], CUBLAS.CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            bias_ptr, sizeof(bias_ptr))
    end

    if aux !== nothing
        aux_ptr = Ref{CuPtr{Cvoid}}(pointer(aux))
        CUBLAS.cublasLtMatmulDescSetAttribute(
            operationDesc[], CUBLAS.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
            aux_ptr, sizeof(aux_ptr))
        ldaux = max(1, stride(aux, 2))
        CUBLAS.cublasLtMatmulDescSetAttribute(
            operationDesc[], CUBLAS.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
            Ref{Csize_t}(ldaux), sizeof(ldaux))
    end

    # Create the matrix layouts
    wdesc = Ref{CUBLAS.cublasLtMatrixLayout_t}()
    xdesc = Ref{CUBLAS.cublasLtMatrixLayout_t}()
    ydesc = Ref{CUBLAS.cublasLtMatrixLayout_t}()

    CUBLAS.cublasLtMatrixLayoutCreate(
        wdesc, convert(CUDA.cudaDataType, wxT), m, k, max(1, stride(w, 2)))
    CUBLAS.cublasLtMatrixLayoutCreate(
        xdesc, convert(CUDA.cudaDataType, wxT), k, n, max(1, stride(x, 2)))
    CUBLAS.cublasLtMatrixLayoutCreate(
        ydesc, convert(CUDA.cudaDataType, yT), m, n, max(1, stride(y, 2)))

    # Create the preference. we can customize this but we will stick to the defaults
    preference = Ref{CUBLAS.cublasLtMatmulPreference_t}()
    CUBLAS.cublasLtMatmulPreferenceCreate(preference)

    # Create the light handle
    lthandle = Ref{CUBLAS.cublasLtHandle_t}()
    CUBLAS.cublasLtCreate(lthandle)

    # Search for the best algorithm
    heuristic = Ref{CUBLAS.cublasLtMatmulHeuristicResult_t}()
    returnedResults = Ref{Cint}(0)
    CUBLAS.cublasLtMatmulAlgoGetHeuristic(
        lthandle[], operationDesc[], wdesc[], xdesc[], ydesc[],
        ydesc[], preference[], 1, heuristic, returnedResults)

    returnedResults[] == 0 && return -1

    CUBLAS.cublasLtMatmul(
        lthandle[], operationDesc[], Ref{wxT}(1), w, wdesc[], x, xdesc[], Ref{yT}(0),
        y, ydesc[], y, ydesc[], Ref(heuristic[].algo), CUDA.CU_NULL, 0, CUDA.stream())

    !activation_fused && (@. y = σ(y))

    return 0
end

function __epilogue_act(f::F, b, aux) where {F}
    if f === identity
        @assert aux===nothing "`aux` must be `nothing` for `identity` activation."
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
        @assert aux===nothing "`aux` must be `nothing` for `$(f)` activation."
        b === nothing && return CUBLAS.CUBLASLT_EPILOGUE_DEFAULT, false
        return CUBLAS.CUBLASLT_EPILOGUE_BIAS, false
    end
end
