using AlgebraOfGraphics, DataFrames, JSON3, CairoMakie
const AoG = AlgebraOfGraphics

set_aog_theme!()

reactant_results = JSON3.read(joinpath(@__DIR__, "reactant.json"), Dict)
cudajl_results = JSON3.read(joinpath(@__DIR__, "cudajl.json"), Dict)
xla_results = JSON3.read(joinpath(@__DIR__, "xla.json"), Dict)

# Model label mapping for prettier display
function model_label(name)
    name == "kan_base_act" && return "KAN"
    name == "kan_no_base_act" && return "KAN (no base act.)"
    return name
end

# Build DataFrame for runtimes
begin
    df = DataFrame(
        "Framework" => String[],
        "Model" => String[],
        "Primal Time" => Float64[],
        "Reverse Time" => Float64[],
    )

    for (framework, results) in [
        ("Reactant.jl", reactant_results), ("CUDA.jl", cudajl_results), ("XLA", xla_results)
    ]
        for (model_name, times) in pairs(results)
            push!(
                df,
                Dict(
                    "Framework" => framework,
                    "Model" => String(model_name),
                    "Primal Time" => times["forward"],
                    "Reverse Time" => times["backward"],
                ),
            )
        end
    end

    df_long = stack(
        df, ["Primal Time", "Reverse Time"]; variable_name="Mode", value_name="Time (s)"
    )

    df_long = filter(row -> row["Time (s)"] ≥ 0, df_long)
end

# Plot runtimes
for (color, fname) in ((:white, "kan_runtimes_dark"), (:black, "kan_runtimes"))
    fig = draw(
        data(df_long) *
        mapping(
            "Model" => model_label => "Model",
            "Time (s)";
            col="Mode",
            color="Framework" => "",
            dodge="Framework",
        ) *
        visual(BarPlot; strokewidth=2, strokecolor=color),
        scales(; Color=(; palette=:tab10));
        figure=(;
            size=(800, 400),
            title="KAN Model Runtimes (Lower is Better)",
            titlealign=:center,
            backgroundcolor=:transparent,
            titlecolor=color,
        ),
        axis=(;
            backgroundcolor=:transparent,
            xticklabelcolor=color,
            yticklabelcolor=color,
            xlabelcolor=color,
            ylabelcolor=color,
            titlecolor=color,
        ),
        legend=(;
            position=:bottom,
            title="Framework",
            backgroundcolor=:transparent,
            labelcolor=color,
        ),
        facet=(; linkyaxes=:minimal),
    )

    if isdefined(Main, :VSCodeServer)
        display(fig)
    end

    save(joinpath(@__DIR__, fname * ".svg"), fig)
    save(joinpath(@__DIR__, fname * ".pdf"), fig)
end

# Build DataFrame for speedups
begin
    df_speedups = DataFrame(
        "Comparison" => String[],
        "Model" => String[],
        "Primal Time" => Float64[],
        "Reverse Time" => Float64[],
    )
    df_hlines = DataFrame(
        "Comparison" => String[], "Model" => String[], "Hline" => Float64[]
    )

    models = ["kan_base_act", "kan_no_base_act"]

    for (i, model) in enumerate(models)
        if !(model in keys(reactant_results))
            continue
        end

        # XLA / Reactant.jl comparison
        if model in keys(xla_results)
            push!(
                df_speedups,
                Dict(
                    "Comparison" => "XLA / Reactant.jl",
                    "Model" => model,
                    "Primal Time" =>
                        xla_results[model]["forward"] / reactant_results[model]["forward"],
                    "Reverse Time" =>
                        xla_results[model]["backward"] /
                        reactant_results[model]["backward"],
                ),
            )

            if i == 1
                push!(
                    df_hlines,
                    Dict(
                        "Comparison" => "XLA / Reactant.jl",
                        "Model" => model,
                        "Hline" => 1.0,
                    ),
                )
            end
        end

        # CUDA.jl / Reactant.jl comparison
        if model in keys(cudajl_results)
            push!(
                df_speedups,
                Dict(
                    "Comparison" => "CUDA.jl / Reactant.jl",
                    "Model" => model,
                    "Primal Time" =>
                        cudajl_results[model]["forward"] /
                        reactant_results[model]["forward"],
                    "Reverse Time" =>
                        cudajl_results[model]["backward"] /
                        reactant_results[model]["backward"],
                ),
            )

            if i == 1
                push!(
                    df_hlines,
                    Dict(
                        "Comparison" => "CUDA.jl / Reactant.jl",
                        "Model" => model,
                        "Hline" => 1.0,
                    ),
                )
            end
        end
    end

    df_speedups_long = stack(
        df_speedups,
        ["Primal Time", "Reverse Time"];
        variable_name="Mode",
        value_name="Time Ratio",
    )

    df_speedups_long = filter(row -> row["Time Ratio"] ≥ 0, df_speedups_long)
end

# Plot speedups
for (color, fname) in ((:white, "kan_speedups_dark"), (:black, "kan_speedups"))
    fig = draw(
        (
            (
                data(df_speedups_long) *
                mapping(
                    "Model" => model_label => "Model",
                    "Time Ratio";
                    color="Mode" => "",
                    dodge="Mode",
                ) *
                visual(BarPlot; strokewidth=2, strokecolor=color)
            ) + (
                data(df_hlines) *
                mapping("Hline") *
                visual(HLines; linestyle=:dash, color=color, linewidth=2)
            )
        ) * mapping(; row="Comparison"),
        scales(; Color=(; palette=:tab10));
        figure=(;
            size=(600, 500),
            title="Speedup relative to other frameworks (Higher is Better)",
            titlealign=:center,
            backgroundcolor=:transparent,
            titlecolor=color,
        ),
        facet=(; linkyaxes=:minimal),
        axis=(;
            backgroundcolor=:transparent,
            xticklabelcolor=color,
            yticklabelcolor=color,
            xlabelcolor=color,
            ylabelcolor=color,
            titlecolor=color,
        ),
        legend=(; position=:bottom, backgroundcolor=:transparent, labelcolor=color),
    )

    if isdefined(Main, :VSCodeServer)
        display(fig)
    end

    save(joinpath(@__DIR__, fname * ".svg"), fig)
    save(joinpath(@__DIR__, fname * ".pdf"), fig)
end
