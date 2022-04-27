using DataFrames, VegaLite

include("benchmark.jl")

btimes = Dict()

# Linear
input_size = (784,)
model_name = "Dense 784 => 1024 with bias"
run_benchmark_flux!(Dense(784, 1024; bias=true), input_size, sum; model_name, btimes)
run_benchmark_efl!(Lux.Dense(784, 1024; bias=true), input_size, sum ∘ first; model_name, btimes)
model_name = "Dense 784 => 1024 without bias"
run_benchmark_flux!(Dense(784, 1024; bias=false), input_size, sum; model_name, btimes)
run_benchmark_efl!(Lux.Dense(784, 1024; bias=false), input_size, sum ∘ first; model_name, btimes)

# Convolution
input_size = (32, 32, 3)
model_name = "Conv 32 x 32 x 3 => 32 x 32 x 16"
run_benchmark_flux!(Conv((3, 3), 3 => 32; pad=SamePad(), stride=1), input_size, sum; model_name, btimes)
run_benchmark_efl!(
    Lux.Conv((3, 3), 3 => 32; pad=SamePad(), stride=1), input_size, sum ∘ first; model_name, btimes
)
model_name = "Conv 32 x 32 x 3 => 32 x 32 x 16 (Input Size Fixed)"
run_benchmark_efl!(
    Lux.Conv((3, 3), 3 => 32; pad=SamePad(), stride=1, input_size=input_size[1:2]),
    input_size,
    sum ∘ first;
    model_name,
    btimes,
)

# Processing the data
function postprocess(
    btimes;
    df=DataFrame(
        "Framework" => String[],
        "Layer" => String[],
        "Description" => String[],
        "Device" => String[],
        "Pass" => String[],
        "Batch Size" => Int[],
        "Timing (in s)" => Float64[],
    ),
)
    for framework in keys(btimes)
        for layer_desc in keys(btimes[framework])
            layer, desc = split(layer_desc, " "; limit=2)
            for which_pass in keys(btimes[framework][layer_desc])
                pass, device = rsplit(which_pass, " "; limit=2)
                for bsize_desc in keys(btimes[framework][layer_desc][which_pass])
                    bsize = parse(Int, last(rsplit(bsize_desc, " "; limit=2)))
                    push!(
                        df,
                        (
                            framework,
                            layer,
                            layer_desc,
                            device,
                            pass,
                            bsize,
                            btimes[framework][layer_desc][which_pass][bsize_desc],
                        ),
                    )
                end
            end
        end
    end
    return df
end

df = postprocess(btimes)
df[!, "log₂(Batch Size)"] = Int64.(log2.(df[!, "Batch Size"]))
df[!, "log₂(Timing (in s))"] = log2.(df[!, "Timing (in s)"])

p = @vlplot(
    :point,
    x = Symbol("log₂(Batch Size)"),
    y = Symbol("log₂(Timing (in s))"),
    color = :Framework,
    row = :Device,
    column = :Pass,
    shape = :Description,
)(
    df
)
