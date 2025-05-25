ENV["XLA_REACTANT_GPU_MEM_FRACTION"] = "0.98"

using ConcreteStructs,
    MLUtils,
    Lux,
    Random,
    Optimisers,
    Printf,
    Statistics,
    DataDeps,
    OneHotArrays,
    Reactant,
    Enzyme,
    BytePairEncoding,
    NNlib
using Comonicon: @main

if !haskey(DataDeps.registry, "shakespeare_char")
    register(
        DataDep(
            "shakespeare_char",
            "Shakespeare Input Text for training NanoGPT",
            "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
            "59a0ad62833b2e15ec811c548618876359e902717431236e52699a0e2bc253ca",
        ),
    )
end

# Setup the model definition

@concrete struct CausalSelfAttention <: AbstractLuxWrapperLayer{:mha}
    mha

    function CausalSelfAttention(args...; kwargs...)
        mha = MultiHeadAttention(args...; kwargs...)
        return new{typeof(mha)}(mha)
    end
end

function (attn::CausalSelfAttention)(x::AbstractArray{T,3}, ps, st) where {T}
    (y, α), stₙ = attn.mha((x, x, x, NNlib.make_causal_mask(x)), ps, st)
    return y, stₙ
end

@concrete struct GPT2Block <: AbstractLuxWrapperLayer{:block}
    block
end

function GPT2Block(; embed_dim, num_heads, hidden_dim, dropout_rate)
    return GPT2Block(
        Chain(
            SkipConnection(
                Chain(
                    LayerNorm(embed_dim; dims=nothing),
                    CausalSelfAttention(
                        embed_dim;
                        nheads=num_heads,
                        attention_dropout_probability=dropout_rate,
                        dense_kwargs=(; init_weight=glorot_uniform, init_bias=zeros32),
                    ),
                ),
                +,
            ),
            SkipConnection(
                Chain(
                    LayerNorm(embed_dim; dims=nothing),
                    Dense(
                        embed_dim => hidden_dim,
                        gelu;
                        init_weight=glorot_uniform,
                        init_bias=zeros32,
                    ),
                    Dense(
                        hidden_dim => embed_dim;
                        init_weight=glorot_uniform,
                        init_bias=zeros32,
                    ),
                    Dropout(dropout_rate),
                ),
                +,
            ),
        ),
    )
end

@concrete struct GPT2 <: AbstractLuxContainerLayer{(:tok_emb, :pos_emb, :gpt_blocks)}
    tok_emb
    pos_emb
    gpt_blocks
end

function GPT2(;
    n_vocab, embed_dim, num_heads, hidden_dim, dropout_rate, block_size, n_layers
)
    return GPT2(
        Embedding(n_vocab => embed_dim),
        Embedding(block_size => embed_dim),
        Chain(
            Dropout(dropout_rate),
            Chain(
                ntuple(
                    Returns(GPT2Block(; embed_dim, num_heads, dropout_rate, hidden_dim)),
                    n_layers,
                )...,
            ),
            LayerNorm(embed_dim; dims=nothing),
        ),
    )
end

function (model::GPT2)(x, ps, st)
    token_embeddings, st_tok_emb = model.tok_emb(x, ps.tok_emb, st.tok_emb)
    pos_embeddings, st_pos_emb = model.pos_emb(1:size(x, 1), ps.pos_emb, st.pos_emb)
    embedding_output = token_embeddings .+ pos_embeddings

    query, st_gpt_blocks = model.gpt_blocks(embedding_output, ps.gpt_blocks, st.gpt_blocks)
    _, seq_len, batch_size = size(query)
    outputs = reshape(
        ps.tok_emb.weight' * reshape(query, :, seq_len * batch_size), :, seq_len, batch_size
    )

    return outputs, (; tok_emb=st_tok_emb, pos_emb=st_pos_emb, gpt_blocks=st_gpt_blocks)
end

#=
dev = reactant_device(; force=true)
rng = Random.default_rng()

model = GPT2(;
    n_vocab=50304,
    embed_dim=1024,
    hidden_dim=3072,
    block_size=1024,
    n_layers=3,
    dropout_rate=0.0,
    num_heads=16,
)
ps, st = Lux.setup(rng, model) |> dev;

x = rand(1:50304, 48, 32) |> dev;

@code_hlo model(x, ps, st)

sumabs2first(layer, x, ps, st) = sum(abs2, first(layer(x, ps, st)))

@code_hlo Enzyme.gradient(Reverse, sumabs2first, Const(model), x, ps, Const(st))
=#

# Use the model to generate some text.

function weighted_sample!(rng, items::AbstractVector, weights::AbstractVector, n::Int)
    @assert length(items) == length(weights)

    weights = weights ./ sum(weights)
    cumprobs = reshape(cumsum(weights), :, 1)
    random_vals = rand(rng, 1, n)

    indices = dropdims(sum(cumprobs .< random_vals; dims=1); dims=1) .+ 1
    return items[indices]
end

function generate_text(model, ps, st, seed; alphabet, output_length, sequence_length)
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
    embed_dim::Int=64,
    n_hidden::Int=256,
    n_heads::Int=4,
    qk_dim::Int=16,
    v_dim::Int=16,
    n_layers::Int=6,
    sequence_length::Int=64,
    batchsize::Int=128,
    dropout_rate::Float32=0.0f0,
    test_split::Float64=0.1,
    lr::Float64=1e-2,
    epochs::Int=100,
    # Only inference options
    inference::Bool=false,
    model_path::String="",
    seed::Union{String,Vector{String}}=["_", "The", "Julia", "Lux.jl"],
    output_length::Int=1024,
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

        texts = generate_text(model, ps, st, seed; alphabet, output_length, sequence_length)

        for (i, (text, s)) in enumerate(zip(texts, seed))
            @printf "[Info] Seed [%d]: %s\n" i s
            @printf "[Generated Text] %s\n\n" text
        end

        return nothing
    end

    alphabet, trainX, trainY, testX, testY = get_nanogpt_data(; sequence_length, test_split)

    @printf "[Info] Alphabet size: %d\n" length(alphabet)
    @printf "[Info] Training size: %d sequences.\n" size(trainX, 2)
    @printf "[Info] Testing  size: %d sequences.\n\n" size(testX, 2)

    train_loader =
        DataLoader((trainX, trainY); batchsize, shuffle=true, parallel=true) |> dev

    model_config = (;
        n_vocab=length(alphabet),
        embed_dim,
        sequence_length,
        n_hidden,
        n_layers,
        dropout_rate,
        n_heads,
        qk_dim,
        v_dim,
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
            model, ps |> cdev, st |> cdev, seed; alphabet, output_length, sequence_length
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
                model_config=model_config,
            )
        end
    end
end
