using AlgebraOfGraphics, DataFrames, JSON3, OrderedCollections, CairoMakie
const AoG = AlgebraOfGraphics

reactant_results = JSON3.read(joinpath(@__DIR__, "reactant.json"), Dict)
cudajl_results = JSON3.read(joinpath(@__DIR__, "cudajl.json"), Dict)
jax_results = JSON3.read(joinpath(@__DIR__, "jax.json"), Dict)

df = DataFrame(
    OrderedDict(
        "Framework" => String[],
        "Model Depth" => String[],
        "Batch Size" => String[],
        "Forward Time" => Float64[],
        "Backward Time" => Float64[],
    ),
)

for (framework, results) in
    [("Reactant.jl", reactant_results), ("CUDA.jl", cudajl_results), ("JAX", jax_results)]
    for (depth, depth_results) in pairs(results)
        for (batch_size, times) in pairs(depth_results)
            bsize = parse(Int, batch_size)
            push!(
                df,
                OrderedDict(
                    "Framework" => framework,
                    "Model Depth" => depth,
                    "Batch Size" => batch_size,
                    "Forward Time" => times["forward"],
                    "Backward Time" => bsize == 1 ? -1.0 : times["backward"],
                ),
            )
        end
    end
end

# Remove the backward time for batch size 1
df_clean = filter(row -> row."Backward Time" ≥ 0, df)

df_long = stack(
    df_clean, ["Forward Time", "Backward Time"]; variable_name="Mode", value_name="Time (s)"
)

fig = draw(
    data(df_long) *
    mapping(
        "Batch Size",
        "Time (s)";
        col="Model Depth" => (x -> "Model Depth: $x"),
        row="Mode",
        color="Framework",
        dodge="Framework",
    ) *
    visual(BarPlot; strokewidth=2);
    figure=(;
        size=(1000, 500),
        title="ResNet Model Runtimes",
        titlealign=:center,
        backgroundcolor=:transparent,
    ),
    axis=(; backgroundcolor=:transparent,),
    legend=(; position=:bottom, title="Framework", backgroundcolor=:transparent),
)
save(joinpath(@__DIR__, "resnet_runtimes.svg"), fig)
