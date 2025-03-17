using Zygote

using LuxCUDA # NVIDIA GPUs
# using AMDGPU # Install and load AMDGPU to train models on AMD GPUs with ROCm

include(joinpath(@__DIR__, "../common.jl"))

is_distributed() = false
should_log() = true

const gdev = gpu_device()
const cdev = cpu_device()
