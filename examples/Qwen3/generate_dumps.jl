using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

include("main.jl")

model, ps, st, tokenizer = get_model_and_tokenizer("8B", true)

generate_text(
    model,
    "Discuss the effects of artifical intelligence on the future of work.",
    ps,
    st,
    100_000,
    tokenizer,
)

Reactant.with_profiler(joinpath(@__DIR__, "traces")) do
    generate_text(
        model,
        "Discuss the effects of artifical intelligence on the future of work.",
        ps,
        st,
        100_000,
        tokenizer,
    )
end
