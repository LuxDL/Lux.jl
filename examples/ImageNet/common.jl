using Boltz, Lux, MLDataDevices
# import Metalhead # Install and load this package to use the Metalhead models with Lux

using Dates, Random, LoggingExtras, Format
using ImageNetDataset, DataAugmentation, MLUtils, OneHotArrays
using Optimisers, ParameterSchedulers

using JLD2

# Forward Declare some of the functions that we will have to be defined later
function is_distributed end
function should_log end

# Data Loading for ImageNet

# Model & Optimiser Construction
function construct_model(;
    rng::AbstractRNG, model_name::String, model_args, pretrained::Bool=false
)
    model = getproperty(Vision, Symbol(model_name))(model_args...; pretrained)
    ps, st = Lux.setup(rng, model)

    @info "=> model `$(model_name)` created."
    pretrained && @info "==> using pre-trained model`"
    @info "==> number of trainable parameters: $(Lux.parameterlength(ps))"
    @info "==> number of states: $(Lux.statelength(st))"

    return model, ps, st
end

# Training Functions
const logitcrossentropy = CrossEntropyLoss(; logits=Val(true))

function loss_function(model, ps, st, (img, y))
    ŷ, stₙ = model(img, ps, st)
    return logitcrossentropy(ŷ, y), stₙ, (; prediction=ŷ)
end

function accuracy(ŷ::AbstractMatrix, y::AbstractMatrix, topk=(1,))
    pred_labels = partialsortperm.(eachcol(cdev(ŷ)), Ref(1:maximum(topk)); rev=true)
    true_labels = onecold(cdev(y))
    accuracies = Vector{Float64}(undef, length(topk))
    for (i, k) in enumerate(topk)
        accuracies[i] = sum(
            map((a, b) -> sum(view(a, 1:k) .== b), pred_labels, true_labels)
        )
    end
    accuracies .= accuracies .* 100 ./ size(y, 2)
    return accuracies
end

# Checkpointing
function save_checkpoint(state::NamedTuple; is_best::Bool, filename::String)
    should_log() || return nothing
    @assert last(splitext(filename)) == ".jld2" "Filename should have a .jld2 extension."
    isdir(dirname(filename)) || mkpath(dirname(filename))
    save(filename; state)
    @info "=> saved checkpoint `$(filename)`."
    if is_best
        symlink_safe(filename, joinpath(dirname(filename), "model_best.jld2"))
        @info "=> best model updated to `$(filename)`!"
    end
    symlink_safe(filename, joinpath(dirname(filename), "model_current.jld2"))
    return nothing
end

function symlink_safe(src, dest)
    rm(dest; force=true)
    symlink(src, dest)
    return nothing
end

function load_checkpoint(filename::String)
    try # NOTE(@avik-pal): ispath is failing for symlinks?
        return JLD2.load(filename)[:state]
    catch
        @info "$(filename) could not be loaded. This might be because the file \
               is absent or is corrupt. Proceeding by returning `nothing`."
        return nothing
    end
end

# Average Meter

# Misc Functions
function full_gc_and_reclaim()
    GC.gc(true)
    MLDataDevices.functional(CUDADevice) && CUDA.reclaim()
    MLDataDevices.functional(AMDGPUDevice) && AMDGPU.reclaim()
    return nothing
end
