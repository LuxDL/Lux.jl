@testsetup module SharedTestSetup
import Reexport: @reexport

using LuxLib, MLDataDevices
@reexport using LuxTestUtils, StableRNGs, Test, Enzyme, Zygote, NNlib

LuxTestUtils.jet_target_modules!(["LuxLib"])

const LUXLIB_BLAS_BACKEND = lowercase(get(ENV, "LUXLIB_BLAS_BACKEND", "default"))

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

if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using LuxCUDA
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "oneapi"
    using oneAPI
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "metal"
    using Metal
end

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") &&
           MLDataDevices.functional(CUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
           MLDataDevices.functional(AMDGPUDevice)
end
function oneapi_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "oneapi") &&
           MLDataDevices.functional(oneAPIDevice)
end
function metal_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "metal") &&
           MLDataDevices.functional(MetalDevice)
end

const MODES = begin
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, false))
    cuda_testing() && push!(modes, ("cuda", CuArray, true))
    amdgpu_testing() && push!(modes, ("amdgpu", ROCArray, true))
    oneapi_testing() && push!(modes, ("oneapi", oneArray, true))
    metal_testing() && push!(modes, ("metal", MtlArray, true))
    modes
end

generate_fixed_array(::Type{T}, sz...) where {T} = generate_fixed_array(T, sz)
function generate_fixed_array(::Type{T}, sz) where {T}
    return reshape(T.(collect(1:prod(sz)) ./ prod(sz)), sz...)
end
generate_fixed_array(::Type{T}, sz::Int) where {T} = T.(collect(1:sz) ./ sz)

export MODES, StableRNG, generate_fixed_array, BACKEND_GROUP

end
