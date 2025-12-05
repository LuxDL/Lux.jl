function has_cuda()
    try
        return run(`nvidia-smi -L`) === nothing ? false : true
    catch
        return false
    end
end

function has_amdgpu()
    try
        out = read(`rocminfo`, String)
        return occursin("GPU", out)
    catch
        return false
    end
end

has_metal() = Sys.isapple() && Sys.KERNEL === :Darwin

function has_oneapi()
    return isdir("/dev/dri") && any(occursin("render", f) for f in readdir("/dev/dri"))
end

for backend_group in ("cuda", "amdgpu", "metal", "oneapi")
    fnanme = Symbol(:test_, backend_group)
    has_fnname = Symbol(:has_, backend_group)
    @eval function $(fnanme)(backend_group::String="all")
        backend_group == $(QuoteNode(backend_group)) && return true
        $(has_fnname)($(QuoteNode(backend_group))) && return true
        return false
    end
end

function packages_to_install(backend_group::String="all")
    backend_group = lowercase(backend_group)
    @assert backend_group in ("all", "cuda", "amdgpu", "metal", "oneapi")

    pkgs = PackageSpec[]
    test_cuda(backend_group) && push!(pkgs, PackageSpec(; name="CUDA"))
    test_amdgpu(backend_group) && push!(pkgs, PackageSpec(; name="AMDGPU"))
    test_metal(backend_group) && push!(pkgs, PackageSpec(; name="Metal", version="1.9"))
    test_oneapi(backend_group) && push!(pkgs, PackageSpec("oneAPI"))
    return pkgs
end
