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
    try
        return isdir("/dev/dri") && any(occursin("render", f) for f in readdir("/dev/dri"))
    catch
        return false
    end
end

has_opencl() = true

for backend_group in ("cuda", "amdgpu", "metal", "oneapi", "opencl")
    fnanme = Symbol(:test_, backend_group)
    has_fnname = Symbol(:has_, backend_group)
    @eval function $(fnanme)(backend_group::String="all")
        backend_group == $(QuoteNode(backend_group)) && return true
        backend_group != "all" && return false
        $(has_fnname)() && return true
        return false
    end
end
