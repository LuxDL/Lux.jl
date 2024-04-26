const TransOrAdjOrRegStridedCuMatrix{T} = Union{Transpose{T, <:StridedCuMatrix{T}},
    Adjoint{T, <:StridedCuMatrix{T}}, StridedCuMatrix{T}}

function LuxLib._cublaslt_matmul_fused!(
        @nospecialize(y::TransOrAdjOrRegStridedCuMatrix), σ::F,
        @nospecialize(w::TransOrAdjOrRegStridedCuMatrix),
        @nospecialize(x::TransOrAdjOrRegStridedCuMatrix),
        b::Union{Nothing, StridedCuVector}) where {F}
    transy = y isa Transpose || y isa Adjoint
    transx = x isa Transpose || x isa Adjoint
    transw = w isa Transpose || w isa Adjoint
    return LuxLib._cublaslt_matmul_fused!(
        transy, parent(y), σ, transw, parent(w), transx, parent(x), b)
end

function LuxLib._cublaslt_matmul_fused!(
        transy::Bool, @nospecialize(y::StridedCuMatrix{yT}), σ::F,
        transw::Bool, @nospecialize(w::StridedCuMatrix{wT}),
        transx::Bool, @nospecialize(x::StridedCuMatrix{xT}),
        b::Union{Nothing, StridedCuVector}) where {F, yT, wT, xT}
    wxT = promote_type(wT, xT)
    @warn "Mixed Precision Inputs received for `weight`: $(typeof(w)) and `x`: \
           $(typeof(x)). Promoting to $(wxT)." maxlog=1
    return LuxLib._cublaslt_matmul_fused!(
        transy, y, σ, transw, LuxLib._oftype_array(wxT, w),
        transx, LuxLib._oftype_array(wxT, x), b)
end

# TODO: use https://docs.nvidia.com/cuda/cublas/#cublasltmatmul for a more robust
#       computeType mapping. Currently no one uses Lux with weird type combinations so we
#       don't need to worry about it too much and just fall back to the generic
#       implementation
# Returns: 0 if successful, -1 if unsuccessful
function LuxLib._cublaslt_matmul_fused!(
        transy::Bool, @nospecialize(y::StridedCuMatrix{yT}), σ::F,
        transw::Bool, @nospecialize(w::StridedCuMatrix{wxT}),
        transx::Bool, @nospecialize(x::StridedCuMatrix{wxT}),
        b::Union{Nothing, StridedCuVector}) where {F, yT, wxT}
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
    epilogue, activation_fused = __epilogue_act(σ, b)
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

    # Seach for the best algorithm
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

@inline __epilogue_act(::typeof(identity), ::Nothing) = (
    CUBLAS.CUBLASLT_EPILOGUE_DEFAULT, true)
@inline __epilogue_act(::typeof(identity), ::StridedCuVector) = (
    CUBLAS.CUBLASLT_EPILOGUE_BIAS, true)
@inline __epilogue_act(::typeof(NNlib.relu), ::Nothing) = (
    CUBLAS.CUBLASLT_EPILOGUE_RELU, true)
@inline __epilogue_act(::typeof(NNlib.relu), ::StridedCuVector) = (
    CUBLAS.CUBLASLT_EPILOGUE_RELU_BIAS, true)
@inline __epilogue_act(::typeof(NNlib.gelu), ::Nothing) = (
    CUBLAS.CUBLASLT_EPILOGUE_GELU, true)
@inline __epilogue_act(::typeof(NNlib.gelu), ::StridedCuVector) = (
    CUBLAS.CUBLASLT_EPILOGUE_GELU_BIAS, true)
@inline __epilogue_act(::F, ::Nothing) where {F} = (CUBLAS.CUBLASLT_EPILOGUE_DEFAULT, false)
@inline __epilogue_act(::F, ::StridedCuVector) where {F} = (
    CUBLAS.CUBLASLT_EPILOGUE_BIAS, false)
