using AlgebraOfGraphics, DataFrames, JSON3, CairoMakie
const AoG = AlgebraOfGraphics

set_aog_theme!()

reactant_results = JSON3.read(joinpath(@__DIR__, "reactant.json"), Dict)
cudajl_results = JSON3.read(joinpath(@__DIR__, "cudajl.json"), Dict)
jax_results = JSON3.read(joinpath(@__DIR__, "jax.json"), Dict)

begin
    df = DataFrame(
        "Framework" => String[],
        "Model Depth" => String[],
        "Batch Size" => String[],
        "Primal Time" => Float64[],
        "Reverse Time" => Float64[],
    )

    for (framework, results) in [
        ("Reactant.jl", reactant_results), ("CUDA.jl", cudajl_results), ("JAX", jax_results)
    ]
        for (depth, depth_results) in pairs(results)
            for (batch_size, times) in pairs(depth_results)
                push!(
                    df,
                    Dict(
                        "Framework" => framework,
                        "Model Depth" => depth,
                        "Batch Size" => batch_size,
                        "Primal Time" => times["forward"],
                        "Reverse Time" =>
                            parse(Int, batch_size) == 1 ? -1.0 : times["backward"],
                    ),
                )
            end
        end
    end

    df_long = stack(
        df, ["Primal Time", "Reverse Time"]; variable_name="Mode", value_name="Time (s)"
    )

    df_long = filter(row -> row["Time (s)"] ≥ 0, df_long)
end

for (color, fname) in ((:white, "resnet_runtimes_dark"), (:black, "resnet_runtimes"))
    fig = draw(
        data(df_long) *
        mapping(
            "Batch Size",
            "Time (s)";
            col="Model Depth" => (x -> "Model Depth: $x"),
            row="Mode",
            color="Framework" => "",
            dodge="Framework",
        ) *
        visual(BarPlot; strokewidth=2, strokecolor=color),
        scales(; Color=(; palette=:tab10));
        figure=(;
            size=(1000, 500),
            title="ResNet Model Runtimes (Lower is Better)",
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

begin
    df_speedups = DataFrame(
        "Comparison" => String[],
        "Model Depth" => String[],
        "Batch Size" => String[],
        "Primal Time" => Float64[],
        "Reverse Time" => Float64[],
    )
    df_hlines = DataFrame(
        "Comparison" => String[], "Model Depth" => String[], "Hline" => Float64[]
    )

    for depth in ["18", "34", "50", "101", "152"]
        for (i, batch_size) in enumerate(["1", "4", "32", "128"])
            if !(
                depth in keys(reactant_results) &&
                batch_size in keys(reactant_results[depth])
            )
                continue
            end

            if (depth in keys(jax_results) && batch_size in keys(jax_results[depth]))
                push!(
                    df_speedups,
                    Dict(
                        "Comparison" => "JAX / Reactant.jl",
                        "Model Depth" => depth,
                        "Batch Size" => batch_size,
                        "Primal Time" =>
                            jax_results[depth][batch_size]["forward"] /
                            reactant_results[depth][batch_size]["forward"],
                        "Reverse Time" =>
                            jax_results[depth][batch_size]["backward"] /
                            reactant_results[depth][batch_size]["backward"],
                    ),
                )

                if i == 1
                    push!(
                        df_hlines,
                        Dict(
                            "Comparison" => "JAX / Reactant.jl",
                            "Model Depth" => depth,
                            "Hline" => 1.0,
                        ),
                    )
                end
            end

            if (depth in keys(cudajl_results) && batch_size in keys(cudajl_results[depth]))
                push!(
                    df_speedups,
                    Dict(
                        "Comparison" => "CUDA.jl / Reactant.jl",
                        "Model Depth" => depth,
                        "Batch Size" => batch_size,
                        "Primal Time" =>
                            cudajl_results[depth][batch_size]["forward"] /
                            reactant_results[depth][batch_size]["forward"],
                        "Reverse Time" =>
                            cudajl_results[depth][batch_size]["backward"] /
                            reactant_results[depth][batch_size]["backward"],
                    ),
                )

                if i == 1
                    push!(
                        df_hlines,
                        Dict(
                            "Comparison" => "CUDA.jl / Reactant.jl",
                            "Model Depth" => depth,
                            "Hline" => 1.0,
                        ),
                    )
                end
            end
        end
    end

    df_speedups_long = stack(
        df_speedups,
        ["Primal Time", "Reverse Time"];
        variable_name="Mode",
        value_name="Time Ratio",
    )

    df_speedups_long = filter(
        row ->
            row["Time Ratio"] ≥ 0 &&
                !(row["Mode"] == "Reverse Time" && row["Batch Size"] == "1"),
        df_speedups_long,
    )
end

for (color, fname) in ((:white, "resnet_speedups_dark"), (:black, "resnet_speedups"))
    fig = draw(
        (
            (
                data(df_speedups_long) *
                mapping("Batch Size", "Time Ratio"; color="Mode" => "", dodge="Mode") *
                visual(BarPlot; strokewidth=2, strokecolor=color)
            ) + (
                data(df_hlines) *
                mapping("Hline") *
                visual(HLines; linestyle=:dash, color=color, linewidth=2)
            )
        ) * mapping(; row="Comparison", col="Model Depth" => (x -> "Model Depth: $x")),
        scales(; Color=(; palette=:tab10));
        figure=(;
            size=(1000, 500),
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
