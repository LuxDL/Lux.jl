CUDA.allowscalar(false)

function unsafe_free! end

if LuxCUDA.functional()
    function unsafe_free!(x)
        return hasmethod(CUDA.unsafe_free!, Tuple{typeof(x)}) ? CUDA.unsafe_free!(x) :
               nothing
    end
    unsafe_free!(x::OneHotArray) = CUDA.unsafe_free!(x.indices)
elseif LuxAMDGPU.functional()
    function unsafe_free!(x)
        return hasmethod(AMDGPU.unsafe_free!, Tuple{typeof(x)}) ? AMDGPU.unsafe_free!(x) :
               nothing
    end
    unsafe_free!(x::OneHotArray) = AMDGPU.unsafe_free!(x.indices)
end

function reclaim_all()
    GC.gc(true)
    LuxCUDA.functional() && CUDA.reclaim()
    LuxAMDGPU.functional() && AMDGPU.reclaim()
    return
end

# Loss Function
logitcrossentropyloss(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))

function logitcrossentropyloss(x, y, model, ps, st)
    ŷ, st_ = model(x, ps, st)
    return logitcrossentropyloss(ŷ, y), ŷ, st_
end

# Random
get_prng(seed::Int) = Xoshiro(seed)

# Accuracy
function accuracy(ŷ, y, topk=(1,))
    maxk = maximum(topk)

    pred_labels = partialsortperm.(eachcol(ŷ), (1:maxk,), rev=true)
    true_labels = onecold(y)

    accuracies = Vector{Float32}(undef, length(topk))

    for (i, k) in enumerate(topk)
        accuracies[i] = sum(map(
            (a, b) -> sum(view(a, 1:k) .== b), pred_labels, true_labels))
    end

    return accuracies .* 100 ./ size(y, ndims(y))
end

# Checkpointing
function save_checkpoint(state::NamedTuple; is_best::Bool, filename::String)
    @assert last(splitext(filename))==".jld2" "Filename should have a .jld2 extension."
    isdir(dirname(filename)) || mkpath(dirname(filename))
    save(filename; state)
    is_best && _symlink_safe(filename, joinpath(dirname(filename), "model_best.jld2"))
    _symlink_safe(filename, joinpath(dirname(filename), "model_current.jld2"))
    return nothing
end

function _symlink_safe(src, dest)
    rm(dest; force=true)
    return symlink(src, dest)
end

function load_checkpoint(fname::String)
    try
        # NOTE(@avik-pal): ispath is failing for symlinks?
        return JLD2[:state]
    catch
        @warn "$fname could not be loaded. This might be because the file is absent or is \
               corrupt. Proceeding by returning `nothing`."
        return nothing
    end
end

# Tracking
@kwdef mutable struct AverageMeter
    fmtstr
    val::Float64 = 0.0
    sum::Float64 = 0.0
    count::Int = 0
    average::Float64 = 0
end

function AverageMeter(name::String, fmt::String)
    fmtstr = FormatExpr("$name {1:$fmt} ({2:$fmt})")
    return AverageMeter(; fmtstr=fmtstr)
end

function (meter::AverageMeter)(val, n::Int)
    meter.val = val
    s = val * n
    if is_distributed()
        v = [s, typeof(val)(n)]
        v = DistributedUtils.allreduce!(backend, v, +)
        s = v[1]
        n = Int(v[2])
    end
    meter.sum += s
    meter.count += n
    meter.average = meter.sum / meter.count
    return meter.average
end

function reset_meter!(meter::AverageMeter)
    meter.val = 0.0
    meter.sum = 0.0
    meter.count = 0
    meter.average = 0.0
    return meter
end

function print_meter(meter::AverageMeter)
    return printfmt(meter.fmtstr, meter.val, meter.average)
end

# ProgressMeter
struct ProgressMeter{N}
    batch_fmtstr
    meters::NTuple{N, AverageMeter}
end

function ProgressMeter(num_batches::Int, meters::NTuple{N}, prefix::String="") where {N}
    fmt = "%" * string(length(string(num_batches))) * "d"
    fmt2 = "{1:" * string(length(string(num_batches))) * "d}"
    prefix = prefix != "" ? endswith(prefix, " ") ? prefix : prefix * " " : ""
    batch_fmtstr = FormatExpr("$prefix[$fmt2/" * cfmt(fmt, num_batches) * "]")
    return ProgressMeter{N}(batch_fmtstr, meters)
end

function reset_meter!(meter::ProgressMeter)
    reset_meter!.(meter.meters)
    return meter
end

function print_meter(meter::ProgressMeter, batch::Int)
    printfmt(meter.batch_fmtstr, batch)
    foreach(x -> (print("\t"); print_meter(x)), meter.meters[1:end])
    println()
    return nothing
end

get_loggable_values(meter::ProgressMeter) = getproperty.(meter.meters, :average)

# Optimisers State
function (dev::LuxDeviceUtils.AbstractLuxDevice)(l::Optimisers.Leaf)
    @set! l.state = dev(l.state)
    return l
end

logitcrossentropy(y_pred, y; dims=1) = mean(-sum(y .* logsoftmax(y_pred; dims); dims))
