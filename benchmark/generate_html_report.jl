using Dates: Dates
using DataFrames: DataFrame
using PlotlyJS: PlotlyJS

# Prettify the benchmark results
function restructure_benchmark_results(bench)
    benches = sort(bench; by=x -> Dates.DateTime(x["date"]))
    data = Dict{String, Dict}()

    total_plots = 0
    for bench in benches, single_bench in bench["results"]
        model_name, device_name, op_name, prop_name, size_name = split(
            single_bench["benchmark"], "/")
        map_main = get!(data, model_name, Dict())
        map_device = get!(map_main, device_name, Dict())
        map_op = get!(map_device, op_name, Dict())
        length(keys(map_op)) == 0 && (total_plots += 1)
        final_name = join((prop_name, size_name), "/")

        res = get!(map_op, final_name,
            (; median=Float64[], min=Float64[], max=Float64[],
                commit_and_date=Tuple{String, Dates.DateTime}[]))
        push!(res.median, single_bench["median"])
        push!(res.min, single_bench["min"])
        push!(res.max, single_bench["max"])
        push!(res.commit_and_date, (bench["commit"], Dates.DateTime(bench["date"])))
    end

    all_plots = Vector{Any}(undef, total_plots)

    i = 1
    for (model_name, model_data) in pairs(data)
        for (device_name, device_data) in pairs(model_data)
            for (op_name, op_data) in pairs(device_data)
                names = keys(op_data)
                data_list = [op_data[name] for name in names]
                commit_and_date = union(broadcast(x -> x.commit_and_date, data_list)...)
                benchmark_timings = fill(NaN, length(commit_and_date), length(names))
                benchmark_timings_min = fill(NaN, length(commit_and_date), length(names))
                benchmark_timings_max = fill(NaN, length(commit_and_date), length(names))
                for (j, name) in enumerate(names),
                    (i, com_date) in enumerate(commit_and_date)

                    data_name = data_list[j]
                    if com_date in data_name.commit_and_date
                        idx = findfirst(x -> x == com_date, data_name.commit_and_date)
                        benchmark_timings[i, j] = data_name.median[idx]
                        benchmark_timings_min[i, j] = data_name.min[idx]
                        benchmark_timings_max[i, j] = data_name.max[idx]
                    end
                end
                benchmark_names = reduce(
                    vcat, [repeat([name], length(commit_and_date))
                           for name in names])
                df = DataFrame(
                    "Commit" => repeat([x[1][1:7] for x in commit_and_date], length(names)),
                    "Date" => repeat([x[2] for x in commit_and_date], length(names)),
                    "Key" => benchmark_names, "Time" => vec(benchmark_timings),
                    "Time (Min)" => vec(benchmark_timings_min),
                    "Time (Max)" => vec(benchmark_timings_max))

                plt = PlotlyJS.plot(
                    df, PlotlyJS.Layout(
                        ;
                        title="$(model_name) + $(device_name) + $(op_name)",
                    );
                    x=:Commit,
                    y=:Time,
                    mode="markers+lines",
                    color=:Key,
                    error_x=PlotlyJS.attr(; type="data", array=Symbol("Time (Max)"),
                        arrayminus=Symbol("Time (Min)"), visible=true),
                    labels=Dict(
                        "Time" => "Time (s)",
                        "Commit" => "Commit"
                    ))

                all_plots[i] = plt.plot
                i += 1
            end
        end
    end

    open(joinpath(@__DIR__, "wip", "all_plots.html"), "w") do io
        for (i, plt) in enumerate(all_plots)
            PlotlyJS.PlotlyBase.to_html(
                io, plt; default_height="400px", default_width="800px", full_html=i == 1)
        end
        println(io, "<br><br><br>")
    end

    return data
end
