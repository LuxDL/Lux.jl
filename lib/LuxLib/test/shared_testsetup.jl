@testsetup module SharedTestSetup
import Reexport: @reexport

using LuxLib, MLDataDevices
@reexport using LuxTestUtils, StableRNGs, Test, Enzyme, Zygote, NNlib

LuxTestUtils.jet_target_modules!(["LuxLib"])

const LUXLIB_BLAS_BACKEND = lowercase(get(ENV, "LUXLIB_BLAS_BACKEND", "default"))

if parse(Bool, get(ENV, "LUXLIB_LOAD_LOOPVEC", "true"))
    using LoopVectorization: LoopVectorization
    using Octavian: Octavian
end

if LUXLIB_BLAS_BACKEND == "default"
    @info "Using default BLAS backend: OpenBLAS"
elseif LUXLIB_BLAS_BACKEND == "appleaccelerate"
    @info "Using AppleAccelerate BLAS backend"
    using AppleAccelerate
elseif LUXLIB_BLAS_BACKEND == "blis"
    @info "Using BLIS BLAS backend"
    using BLISBLAS
elseif LUXLIB_BLAS_BACKEND == "mkl"
    @info "Using MKL BLAS backend"
    using MKL
else
    error("Unknown BLAS backend: $(LUXLIB_BLAS_BACKEND)")
end

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "All"))

if LuxTestUtils.test_cuda(BACKEND_GROUP)
    using LuxCUDA
end

if LuxTestUtils.test_amdgpu(BACKEND_GROUP)
    using AMDGPU
end

if LuxTestUtils.test_oneapi(BACKEND_GROUP)
    using oneAPI
end

if LuxTestUtils.test_metal(BACKEND_GROUP)
    using Metal
end

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return LuxTestUtils.test_cuda(BACKEND_GROUP) && MLDataDevices.functional(CUDADevice)
end
function amdgpu_testing()
    return LuxTestUtils.test_amdgpu(BACKEND_GROUP) && MLDataDevices.functional(AMDGPUDevice)
end
function oneapi_testing()
    return LuxTestUtils.test_oneapi(BACKEND_GROUP) && MLDataDevices.functional(oneAPIDevice)
end
function metal_testing()
    return LuxTestUtils.test_metal(BACKEND_GROUP) && MLDataDevices.functional(MetalDevice)
end

const MODES = begin
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, false, true))
    cuda_testing() && push!(modes, ("cuda", CuArray, true, true))
    amdgpu_testing() && push!(modes, ("amdgpu", ROCArray, true, true))
    oneapi_testing() && push!(modes, ("oneapi", oneArray, true, false))
    metal_testing() && push!(modes, ("metal", MtlArray, true, false))
    modes
end

generate_fixed_array(::Type{T}, sz...) where {T} = generate_fixed_array(T, sz)
function generate_fixed_array(::Type{T}, sz) where {T}
    return reshape(T.(collect(1:prod(sz)) ./ prod(sz)), sz...)
end
generate_fixed_array(::Type{T}, sz::Int) where {T} = T.(collect(1:sz) ./ sz)

sumabs2first(f::F, args...) where {F} = sum(abs2, first(f(args...)))

export MODES, StableRNG, generate_fixed_array, BACKEND_GROUP, sumabs2first

end
