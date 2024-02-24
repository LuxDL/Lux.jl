module LuxCUDANCCLExt

using CUDA, Lux, NCCL

# FIXME: Remove before merging
function __init__()
    @info "CUDA information:\n" * sprint(io -> CUDA.versioninfo(io))
    @info "NCCL version: $(NCCL.version())"
end

end
