using Lux
using Random
using Images
using Optimisers
using ProgressBars
using CUDA
using BSON
using Plots
using Comonicon
using Printf

# https://discourse.julialang.org/t/deactivate-plot-display-to-avoid-need-for-x-server/19359
ENV["GKSwstype"] = "nul"

include("./model.jl")

function load_checkpoint(path)
    d = BSON.load(path)
    return d[:ps], d[:st], d[:opt_st]
end

function save_as_gif(images_each_step::Vector{<:AbstractArray{T, 4}},
                     output_dir) where {T <: AbstractFloat}
    num_images = size(images_each_step[1], 4)

    for image_id in ProgressBar(1:num_images)
        images = [view(x, :, :, :, image_id) for x in images_each_step]
        save_as_gif(image_id, images, output_dir)
    end
end

function save_as_gif(image_id::Int, images, output_dir)
    outpath = joinpath(output_dir, @sprintf("img_%.3d.gif", image_id))
    anim = @animate for (step, img) in enumerate(images)
        img = colorview(RGB, permutedims(img, (3, 1, 2)))
        plot(img; legend=false, size=(300, 300), axis=([], false), title="step=$(step)")
    end
    return gif(anim, outpath; fps=10)
end

@main function main(; checkpoint::String, image_size::Int=64, num_images::Int=10,
                    diffusion_steps::Int=80, output_dir::String="output/generate",
                    # model hyper params
                    channels::Vector{Int}=[32, 64, 96, 128], block_depth::Int=2,
                    min_freq::Float32=1.0f0, max_freq::Float32=1000.0f0,
                    embedding_dims::Int=32, min_signal_rate::Float32=0.02f0,
                    max_signal_rate::Float32=0.95f0)
    rng = Random.MersenneTwister()
    Random.seed!(rng, 1234)

    mkpath(output_dir)

    if CUDA.functional()
        println("GPU is available.")
    else
        println("GPU is not available.")
    end

    ddim = DenoisingDiffusionImplicitModel((image_size, image_size); channels=channels,
                                           block_depth=block_depth, min_freq=min_freq,
                                           max_freq=max_freq, embedding_dims=embedding_dims,
                                           min_signal_rate=min_signal_rate,
                                           max_signal_rate=max_signal_rate)

    ps, st, _ = load_checkpoint(checkpoint) .|> gpu

    println("Generate images.")
    st = Lux.testmode(st)
    _, images_each_step = generate(ddim, rng, (image_size, image_size, 3, num_images),
                                   diffusion_steps, ps, st; save_each_step=true)

    images_each_step = images_each_step |> cpu

    println("Save diffusion as GIF.")
    return save_as_gif(images_each_step, output_dir)
end
