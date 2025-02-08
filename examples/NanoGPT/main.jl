# Taken from https://github.com/FluxML/model-zoo/pull/410
using ConcreteStructs, MLUtils, Lux, Random, Optimisers, Printf, Statistics, NNlib,
      DataDeps, StatsBase, OneHotArrays, JLD2, Reactant, Enzyme, BytePairEncoding
using Comonicon: @main

if !haskey(DataDeps.registry, "nanogpt")
    register(DataDep(
        "nanogpt",
        "Shakespeare Input Text for training NanoGPT",
        "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
        "59a0ad62833b2e15ec811c548618876359e902717431236e52699a0e2bc253ca"
    ))
end

# Setup the model definition
@concrete struct CausalSelfAttention <:
                 AbstractLuxContainerLayer{(:causal_attn, :proj, :attn_drop)}
    causal_attn
    proj
    attn_drop
    n_embed::Int
    n_heads::Int
end

function CausalSelfAttention(; n_embed, n_heads, dropout_rate, use_bias)
    causal_attn = Dense(n_embed, 3 * n_embed; use_bias)
    proj = Chain(
        Dense(n_embed, n_embed; use_bias),
        Dropout(dropout_rate)
    )
    attn_drop = Dropout(dropout_rate)
    return CausalSelfAttention(causal_attn, proj, attn_drop, n_embed, n_heads)
end

function (attn::CausalSelfAttention)(x::AbstractArray{T, 3}, ps, st) where {T}
    qkv, qkv_st = attn.causal_attn(x, ps.causal_attn, st.causal_attn)
    q, k, v = (
        selectdim(qkv, 1, 1:(attn.n_heads)),
        selectdim(qkv, 1, (attn.n_heads + 1):(2 * attn.n_heads)),
        selectdim(qkv, 1, (2 * attn.n_heads + 1):(3 * attn.n_heads))
    )
    dp = StatefulLuxLayer{true}(attn.attn_drop, ps.attn_drop, st.attn_drop)
    mha, _ = NNlib.dot_product_attention(
        q, k, v, nothing; mask=NNlib.make_causal_mask(x), fdrop=dp, nheads=attn.n_heads
    )
    proj, proj_st = attn.proj(mha, ps.proj, st.proj)
    return proj, (; causal_attn=qkv_st, proj=proj_st, attn_drop=dp.attn_drop)
end

@concrete struct GPTBlock <: AbstractLuxWrapperLayer{:block}
    block
end

function GPTBlock(; n_embed, n_heads, dropout_rate, use_bias)
    return GPTBlock(Chain(
        SkipConnection(
            Chain(
                LayerNorm((n_embed, 1)),
                CausalSelfAttention(; n_embed, n_heads, dropout_rate, use_bias)
            ),
            +
        ),
        SkipConnection(
            Chain(
                LayerNorm((n_embed, 1)),
                Dense(n_embed => 4 * n_embed, gelu; use_bias),
                Dense(4 * n_embed => n_embed; use_bias),
                Dropout(dropout_rate)
            ),
            +
        )
    ))
end

struct PositionalEmbedding{E} <: AbstractLuxWrapperLayer{:embedding}
    embedding::E

    function PositionalEmbedding(args...; kwargs...)
        embed = Embedding(args...; kwargs...)
        return new{typeof(embed)}(embed)
    end
end

(pe::PositionalEmbedding)(x, ps, st) = pe.embedding(1:size(x, 1), ps, st)

@concrete struct GPT <: AbstractLuxWrapperLayer{:layer}
    layer
end

function GPT(; n_vocab, n_embed, block_size, n_layers, dropout_rate, n_heads, use_bias)
    return GPT(Chain(
        Parallel(
            +,
            Embedding(n_vocab => n_embed),
            PositionalEmbedding(block_size => n_embed)
        ),
        Dropout(dropout_rate),
        Chain(ntuple(
            Returns(GPTBlock(; n_embed, n_heads, dropout_rate, use_bias)), n_layers
        )...),
        LayerNorm((n_embed, 1)),
        Dense(n_embed => n_vocab; use_bias)
    ))
end

#=

dev = reactant_device(; force=true)
rng = Random.default_rng()

model = GPT(;
    n_vocab=50304, n_embed=768, block_size=1024, n_layers=12, dropout_rate=0.0, n_heads=12,
    use_bias=true
)
ps, st = Lux.setup(rng, model) |> dev

=#

# Use the model to generate some text.
function generate_text(
        model, ps, st, seed; alphabet, output_length, sequence_length
)
    dev = get_device((ps, st))
    @assert !(dev isa ReactantDevice) "Currently we don't support running inference of \
                                       dynamically sized tensors."

    seed = copy(seed)
    seed_len = maximum(length, seed)
    extra_letters = zeros(Int, length(seed))
    for (i, s) in enumerate(seed)
        if seed_len != length(s)
            extra_letters[i] = seed_len - length(s)
            seed[i] = "_"^extra_letters[i] * s
        end
    end
    original_output_length = output_length
    output_length += maximum(extra_letters)

    st = Lux.testmode(st)

    x = zeros(Int, output_length, length(seed))
    for (i, s) in enumerate(seed), j in 1:seed_len
        x[j, i] = findfirst(==(s[j]), alphabet)
    end
    for i in (seed_len + 1):output_length
        tail = x[max(1, i - sequence_length + 1):(i - 1), :] |> dev
        y = model(tail, ps, st)[1] |> cpu_device()
        p = softmax(y[:, end, 1])
        x[i, :] .= sample(1:length(alphabet), Weights(p))
    end

    res = [String(map(Base.Fix1(getindex, alphabet), x[:, i])) for i in axes(x, 2)]
    for i in eachindex(res)
        res[i] = res[i][(extra_letters[i] + 1):end][1:original_output_length]
    end

    return res
end

# Load data from input file, and partition into training and testing subsets.
function get_nanogpt_data(; sequence_length, test_split)
    data_file = joinpath(datadep"nanogpt", "shakespeare_input.txt")
    text = String(read(data_file))

    idx = ceil(Int, length(text) * (1 - test_split))
    train_text = text[1:idx]
    test_text = text[(idx + 1):end]

    tokenizer = BytePairEncoding.load_gpt2()

    train_tokens = tokenizer(train_text)
    test_tokens = tokenizer(test_text)

    B = (length(text) - 1) ÷ sequence_length
    # We must collect() before indexing, because String indexing does strange things with multi-byte
    # characters and we could end up with the wrong length.
    Xs = reshape(collect(text)[1:(B * sequence_length)], sequence_length, B)
    Ys = reshape(collect(text)[2:(B * sequence_length + 1)], sequence_length, B)

    # Input string starts with stop character '_', representing zero context.
    Xs[1, :] .= stop

    # Xs (input) should consist of indices into `alphabet` because this is what Embedding expects.
    # Ys (output) should be one-hot because this is what logitcrossentropy expects.
    Xs = map(c -> Int32(findfirst(==(c), alphabet)), Xs)
    Ys = onehotbatch(Ys, alphabet)

    trainX, testX = MLUtils.splitobs(Xs; at=1 - test_split)
    trainY, testY = MLUtils.splitobs(Ys; at=1 - test_split)

    return alphabet, Array(trainX), Array(trainY), Array(testX), Array(testY)
end

@main function main(;
        n_embed::Int=64, n_hidden::Int=256, n_heads::Int=4, qk_dim::Int=16,
        v_dim::Int=16, n_layers::Int=6, sequence_length::Int=64, batchsize::Int=128,
        dropout_rate::Float32=0.0f0, test_split::Float64=0.1, lr::Float64=1e-2,
        epochs::Int=100,
        # Only inference options
        inference::Bool=false, model_path::String="",
        seed::Union{String, Vector{String}}=["_", "The", "Julia", "Lux.jl"],
        output_length::Int=1024
)
    rng = Random.default_rng()
    Random.seed!(rng, 1234)

    dev = reactant_device()
    cdev = cpu_device()

    if inference
        @printf "[Info] Inference mode enabled.\n"

        @assert !isempty(model_path) "Please provide a path to a model checkpoint."

        @printf "[Info] Loading model from %s.\n" model_path
        model_config = JLD2.load(model_path, "model_config")
        model = GPT(; model_config...)
        ps = JLD2.load(model_path, "parameters")
        st = JLD2.load(model_path, "states")
        alphabet = JLD2.load(model_path, "alphabet")
        sequence_length = model_config.sequence_length

        texts = generate_text(
            model, ps, st, seed; alphabet, output_length, sequence_length
        )

        for (i, (text, s)) in enumerate(zip(texts, seed))
            @printf "[Info] Seed [%d]: %s\n" i s
            @printf "[Generated Text] %s\n\n" text
        end

        return
    end

    alphabet, trainX, trainY, testX, testY = get_nanogpt_data(; sequence_length, test_split)

    @printf "[Info] Alphabet size: %d\n" length(alphabet)
    @printf "[Info] Training size: %d sequences.\n" size(trainX, 2)
    @printf "[Info] Testing  size: %d sequences.\n\n" size(testX, 2)

    train_loader = DataLoader(
        (trainX, trainY); batchsize, shuffle=true, parallel=true
    ) |> dev

    model_config = (;
        n_vocab=length(alphabet), n_embed, sequence_length, n_hidden,
        n_layers, dropout_rate, n_heads, qk_dim, v_dim
    )
    model = GPT(; model_config...)
    ps, st = Lux.setup(rng, model) |> dev
    @printf "[Info] Number of parameters: %d\n" Lux.parameterlength(ps)
    @printf "[Info] Number of states: %d\n\n" Lux.statelength(st)

    opt = Adam(lr)
    train_state = Training.TrainState(model, ps, st, opt)

    @printf "[Info] Compiling Inference Model...\n"
    testX, testY = (testX, testY) |> dev
    start_time = time()
    model_compiled = @compile model(testX, ps, Lux.testmode(st))
    time_to_compile = time() - start_time
    best_test_loss = Inf

    @printf "[Info] Time taken to compile inference model: %0.5fs\n" time_to_compile
    @printf "[Info] Starting Model Training...\n\n"

    loss_fn = CrossEntropyLoss(; logits=Val(true))

    iter = 0
    for epoch in 1:epochs
        for (x, y) in train_loader
            iter += 1

            start_time = time()
            _, loss, _, train_state = Training.single_train_step!(
                AutoEnzyme(), loss_fn, (x, y), train_state
            )
            time_taken = time() - start_time

            if iter % 100 == 0
                @printf "[Train] Epoch %3d\tIteration %6d\tLoss %.8e\tTime per \
                         Iteration %0.5f\n" epoch iter loss time_taken
            end
        end

        test_loss = loss_fn(
            Array(first(model_compiled(testX, ps, Lux.testmode(st)))), testY
        )
        @printf "[Test] Epoch %3d\tTest Loss %.8e\n" epoch test_loss

        # Generate some text here...
        texts = generate_text(
            model, ps |> cdev, st |> cdev, seed;
            alphabet, output_length, sequence_length
        )
        for (i, (text, s)) in enumerate(zip(texts, seed))
            @printf "[Info] Seed [%d]: %s\n" i s
            @printf "[Generated Text] %s\n\n" text
        end

        if test_loss < best_test_loss
            best_test_loss = test_loss
            @printf "[Info] New best test loss: %.8e\n" best_test_loss
            @printf "[Info] Saving model...\n"
            jldsave(
                joinpath(@__DIR__, "nanogpt.jld2");
                parameters=train_state.parameters |> cdev,
                states=train_state.states |> cdev,
                alphabet=alphabet,
                model_config=model_config
            )
        end
    end
end
