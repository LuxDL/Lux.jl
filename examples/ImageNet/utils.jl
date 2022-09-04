CUDA.allowscalar(false)

# unsafe_free OneHotArrays
CUDA.unsafe_free!(x::OneHotArray) = CUDA.unsafe_free!(x.indices)

# Loss Function
logitcrossentropyloss(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))

function logitcrossentropyloss(x, y, model, ps, st)
    ŷ, st_ = model(x, ps, st)
    return logitcrossentropyloss(ŷ, y), ŷ, st_
end

# Random
function get_prng(seed::Int)
    @static if VERSION >= v"1.7"
        return Xoshiro(seed)
    else
        return MersenneTwister(seed)
    end
end

# Accuracy
function accuracy(ŷ, y, topk=(1,))
    maxk = maximum(topk)

    pred_labels = partialsortperm.(eachcol(ŷ), (1:maxk,), rev=true)
    true_labels = onecold(y)

    accuracies = Vector{Float32}(undef, length(topk))

    for (i, k) in enumerate(topk)
        accuracies[i] = sum(map((a, b) -> sum(view(a, 1:k) .== b), pred_labels,
                                true_labels))
    end

    return accuracies .* 100 ./ size(y, ndims(y))
end

# Distributed Utils
is_distributed() = FluxMPI.Initialized() && total_workers() > 1
should_log() = !FluxMPI.Initialized() || local_rank() == 0

# Checkpointing
function save_checkpoint(state::NamedTuple; is_best::Bool, filename::String)
    isdir(dirname(filename)) || mkpath(dirname(filename))
    JLSO.save(filename, :state => state)
    is_best && _symlink_safe(filename, joinpath(dirname(filename), "model_best.jlso"))
    _symlink_safe(filename, joinpath(dirname(filename), "model_current.jlso"))
    return nothing
end

function _symlink_safe(src, dest)
    rm(dest; force=true)
    return symlink(src, dest)
end

function load_checkpoint(fname::String)
    try
        # NOTE(@avik-pal): ispath is failing for symlinks?
        return JLSO.load(fname)[:state]
    catch
        @warn """$fname could not be loaded. This might be because the file is absent or is
                corrupt. Proceeding by returning `nothing`."""
        return nothing
    end
end

# Parameter Scheduling
## Copied from ParameterSchedulers.jl due to its heavy dependencies
struct CosineAnnealSchedule{restart, T, S <: Integer}
    range::T
    offset::T
    dampen::T
    period::S

    function CosineAnnealSchedule(lambda_0, lambda_1, period; restart::Bool=true,
                                  dampen=1.0f0)
        range = abs(lambda_0 - lambda_1)
        offset = min(lambda_0, lambda_1)
        return new{restart, typeof(range), typeof(period)}(range, offset, dampen, period)
    end
end

function (s::CosineAnnealSchedule{true})(t)
    d = s.dampen^div(t - 1, s.period)
    return (s.range * (1 + cos(pi * mod(t - 1, s.period) / s.period)) / 2 + s.offset) / d
end

function (s::CosineAnnealSchedule{false})(t)
    return s.range * (1 + cos(pi * (t - 1) / s.period)) / 2 + s.offset
end

struct Step{T, S}
    start::T
    decay::T
    step_sizes::S

    function Step(start::T, decay::T, step_sizes::S) where {T, S}
        _step_sizes = (S <: Integer) ? Iterators.repeated(step_sizes) : step_sizes

        return new{T, typeof(_step_sizes)}(start, decay, _step_sizes)
    end
end

(s::Step)(t) = s.start * s.decay^(searchsortedfirst(s.step_sizes, t - 1) - 1)

struct ConstantSchedule{T}
    val::T
end

(s::ConstantSchedule)(t) = s.val

# Tracking
Base.@kwdef mutable struct AverageMeter
    fmtstr::Any
    val::Float64 = 0.0
    sum::Float64 = 0.0
    count::Int = 0
    average::Float64 = 0
end

function AverageMeter(name::String, fmt::String)
    fmtstr = Formatting.FormatExpr("$name {1:$fmt} ({2:$fmt})")
    return AverageMeter(; fmtstr=fmtstr)
end

function (meter::AverageMeter)(val, n::Int)
    meter.val = val
    s = val * n
    if is_distributed()
        v = [s, typeof(val)(n)]
        v = FluxMPI.MPIExtensions.allreduce!(v, +, FluxMPI.MPI.COMM_WORLD)
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
    return Formatting.printfmt(meter.fmtstr, meter.val, meter.average)
end

# ProgressMeter
struct ProgressMeter{N}
    batch_fmtstr::Any
    meters::NTuple{N, AverageMeter}
end

function ProgressMeter(num_batches::Int, meters::NTuple{N}, prefix::String="") where {N}
    fmt = "%" * string(length(string(num_batches))) * "d"
    prefix = prefix != "" ? endswith(prefix, " ") ? prefix : prefix * " " : ""
    batch_fmtstr = Formatting.generate_formatter("$prefix[$fmt/" *
                                                 Formatting.sprintf1(fmt, num_batches) *
                                                 "]")
    return ProgressMeter{N}(batch_fmtstr, meters)
end

function reset_meter!(meter::ProgressMeter)
    reset_meter!.(meter.meters)
    return meter
end

function print_meter(meter::ProgressMeter, batch::Int)
    base_str = meter.batch_fmtstr(batch)
    print(base_str)
    foreach(x -> (print("\t"); print_meter(x)), meter.meters[1:end])
    println()
    return nothing
end

get_loggable_values(meter::ProgressMeter) = getproperty.(meter.meters, :average)

# Optimisers State
function Lux.cpu(l::Optimisers.Leaf)
    @set! l.state = cpu(l.state)
    return l
end

function Lux.gpu(l::Optimisers.Leaf)
    @set! l.state = gpu(l.state)
    return l
end

function logitcrossentropy(y_pred, y; dims=1)
    return mean(-sum(y .* logsoftmax(y_pred; dims=dims); dims=dims))
end
