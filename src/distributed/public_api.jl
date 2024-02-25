module DistributedUtils

import ChainRulesCore as CRC
import Functors: fmap
import ..Lux: AbstractLuxDistributedBackend, MPIBackend, NCCLBackend
import Optimisers: Leaf
import Setfield: @set!

const NCCL_Initialized = Ref(false)
const MPI_Initialized = Ref(false)

"""
    initialized(backend::Val)

Check if the given backend is initialized.
"""
initialized(::Val{:MPI}) = MPI_Initialized[]
initialized(::Val{:NCCL}) = NCCL_Initialized[]

function initialize(backend::Val; kwargs...)
    initialized(backend) && return
    __initialize(backend; kwargs...)
    return
end

function __initialize end

"""
    get_distributed_backend(backend::Val)

Get the distributed backend for the given backend type. Possible values are:

  - `Val(:MPI)`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `Val(:NCCL)`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed.
"""
function get_distributed_backend end

CRC.@non_differentiable get_distributed_backend(::Any...)

"""
    local_rank(backend::AbstractLuxDistributedBackend)

Get the local rank for the given backend.
"""
function local_rank end

CRC.@non_differentiable local_rank(::Any...)

"""
    total_workers(backend::AbstractLuxDistributedBackend)

Get the total number of workers for the given backend.
"""
function total_workers end

CRC.@non_differentiable total_workers(::Any...)

function bcast! end

CRC.@non_differentiable bcast!(::Any...)

function allreduce! end

CRC.@non_differentiable allreduce!(::Any...)

function reduce! end

CRC.@non_differentiable reduce!(::Any...)

# syncronize!
"""
    synchronize!!(backend::AbstractLuxDistributedBackend, ps; root::Int=0)

Synchronize the given structure `ps` using the given backend. The value at `root` will be
broadcasted to all other workers.
"""
function synchronize!!(backend::AbstractLuxDistributedBackend, ps::Tuple; root::Int=0)
    length(ps) == 0 && return ps
    return map(x -> synchronize!!(backend, x; root), ps)
end

function synchronize!!(backend::AbstractLuxDistributedBackend,
        ps::NamedTuple{fields}; root::Int=0) where {fields}
    length(ps) == 0 && return ps
    return NamedTuple{fields}(map(x -> synchronize!!(backend, x; root), values(ps)))
end

function synchronize!!(
        backend::AbstractLuxDistributedBackend, ps::AbstractArray{T}; root::Int=0) where {T}
    if isbitstype(T)
        bcast!(backend, ps; root)
        return ps
    end
    return map(x -> synchronize!!(backend, x; root), ps)
end

function synchronize!!(backend::AbstractLuxDistributedBackend, ps::Leaf; root::Int=0)
    @set! ps.state = synchronize!!(backend, ps.state; root)
    return ps
end

function synchronize!!(backend::AbstractLuxDistributedBackend, ps::T; root::Int=0) where {T}
    isbitstype(T) && return bcast!(backend, [ps]; root)[]
    return ps # If we don't know how to synchronize, just return the value. For ex, Symbol, String, etc.
end

end
