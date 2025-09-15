using Lux, MPI, Optimisers, Random, Test

const input_args = length(ARGS) == 2 ? ARGS : ("cpu", "mpi")

if input_args[1] == "cuda"
    using LuxCUDA
end
if input_args[1] == "amdgpu"
    using AMDGPU
end

const backend_type = input_args[2] == "nccl" ? NCCLBackend : MPIBackend

if input_args[2] == "nccl"
    using NCCL
end

const dev = if input_args[1] == "cpu"
    CPUDevice()
else
    (input_args[1] == "cuda" ? CUDADevice() : AMDGPUDevice())
end

DistributedUtils.initialize(backend_type)
backend = DistributedUtils.get_distributed_backend(backend_type)

opt = Adam(0.001f0)
ps = dev((a=zeros(4), b=zeros(4)))
st_opt = Optimisers.setup(opt, ps)

dopt = DistributedUtils.DistributedOptimizer(backend, opt)
st_dopt = Optimisers.setup(dopt, ps)

@test st_dopt.a.state == st_opt.a.state
@test st_dopt.b.state == st_opt.b.state

@test DistributedUtils.synchronize!!(backend, st_dopt) isa Any

gs = dev((a=ones(4), b=ones(4)))

_, ps_dopt = Optimisers.update(st_dopt, ps, gs)
_, ps_opt = Optimisers.update(st_opt, ps, gs)

@test ps_dopt.a ≈ ps_opt.a atol = 1.0e-5 rtol = 1.0e-5
@test ps_dopt.b ≈ ps_opt.b atol = 1.0e-5 rtol = 1.0e-5

@test Optimisers.adjust(st_dopt, 0.1f0) isa Any
