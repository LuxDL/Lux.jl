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
    NNlib,
    ArgParse

if !haskey(DataDeps.registry, "shakespeare")
    register(
        DataDep(
            "shakespeare",
            "Shakespeare Input Text for training NanoGPT",
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
        ),
    )
end

# Setup the model definition
@concrete struct CausalSelfAttention <: AbstractLuxWrapperLayer{:mha}
    mha
end

function CausalSelfAttention(args...; kwargs...)
    return CausalSelfAttention(MultiHeadAttention(args...; kwargs..., is_causal=true))
end

function (attn::CausalSelfAttention)(x::AbstractArray{T,3}, ps, st) where {T}
    (y, _), stₙ = attn.mha(x, ps, st)
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
                    ),
                ),
                +,
            ),
            SkipConnection(
                Chain(
                    LayerNorm(embed_dim; dims=nothing),
                    Dense(embed_dim => hidden_dim, gelu),
                    Dense(hidden_dim => embed_dim),
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
    pos_embeddings, st_pos_emb = model.pos_emb(
        Int32(1):Int32(size(x, 1)), ps.pos_emb, st.pos_emb
    )
    embedding_output = token_embeddings .+ pos_embeddings

    query, st_gpt_blocks = model.gpt_blocks(embedding_output, ps.gpt_blocks, st.gpt_blocks)

    _, seq_len, batch_size = size(query)
    outputs = reshape(
        ps.tok_emb.weight' * reshape(query, :, seq_len * batch_size), :, seq_len, batch_size
    )

    return outputs, (; tok_emb=st_tok_emb, pos_emb=st_pos_emb, gpt_blocks=st_gpt_blocks)
end

# ## Text Generation

# Use the model to generate some text.

function weighted_sample!(rng, items::AbstractVector, weights::AbstractVector, n::Int)
    @assert length(items) == length(weights)

    weights = weights ./ sum(weights)
    cumprobs = reshape(cumsum(weights), :, 1)
    random_vals = rand(rng, 1, n)

    indices = dropdims(sum(cumprobs .< random_vals; dims=1); dims=1) .+ 1
    return items[indices]
end

## TODO: update this based on Qwen3/main.jl
# function generate_text(model, ps, st, seed; alphabet, output_length, sequence_length)
#     dev = get_device((ps, st))
#     @assert !(dev isa ReactantDevice) "Currently we don't support running inference of \
#                                        dynamically sized tensors."

#     seed = copy(seed)
#     seed_len = maximum(length, seed)
#     extra_letters = zeros(Int, length(seed))
#     for (i, s) in enumerate(seed)
#         if seed_len != length(s)
#             extra_letters[i] = seed_len - length(s)
#             seed[i] = "_"^extra_letters[i] * s
#         end
#     end
#     original_output_length = output_length
#     output_length += maximum(extra_letters)

#     st = Lux.testmode(st)

#     x = zeros(Int, output_length, length(seed))
#     for (i, s) in enumerate(seed), j in 1:seed_len
#         x[j, i] = findfirst(==(s[j]), alphabet)
#     end
#     for i in (seed_len + 1):output_length
#         tail = x[max(1, i - sequence_length + 1):(i - 1), :] |> dev
#         y = model(tail, ps, st)[1] |> cpu_device()
#         p = softmax(y[:, end, 1])
#         x[i, :] .= sample(1:length(alphabet), Weights(p))
#     end

#     res = [String(map(Base.Fix1(getindex, alphabet), x[:, i])) for i in axes(x, 2)]
#     for i in eachindex(res)
#         res[i] = res[i][(extra_letters[i] + 1):end][1:original_output_length]
#     end

#     return res
# end

# ## Data Loading

# Load data from input file, and partition into training and testing subsets.

function get_nanogpt_data(; sequence_length, test_split)
    text = String(read(joinpath(datadep"shakespeare", "input.txt")))
    tokenizer = BytePairEncoding.load_tiktoken_encoder("gpt2")

    text_encoding = tokenizer.encode(text)
    start_token = tokenizer.encode("_")[1]

    n_batches = (length(text_encoding) - 1) ÷ sequence_length

    Xs = reshape(text_encoding[1:(n_batches * sequence_length)], sequence_length, n_batches)
    ## Input string starts with stop character '_', representing zero context.
    Xs[1, :] .= start_token

    Ys = reshape(
        text_encoding[2:(n_batches * sequence_length + 1)], sequence_length, n_batches
    )

    n_vocab = max(maximum(Xs), maximum(Ys))

    ## One-hot encode the target sequences.
    Ys = onehotbatch(Ys, 1:n_vocab)

    Xs_train, Xs_test = MLUtils.splitobs(Xs; at=1 - test_split)
    Ys_train, Ys_test = MLUtils.splitobs(Ys; at=1 - test_split)

    return n_vocab, (Xs_train, Ys_train), (Xs_test, Ys_test)
end

# ## Entry Point

function parse_command_line_arguments()
    settings = ArgParseSettings()
    #! format: off
    @add_arg_table! settings begin
        "--embedding_dim"
            help = "Dimension of the embedding"
            arg_type = Int
            default = 64
        "--n_hidden"
            help = "Number of hidden units"
            arg_type = Int
            default = 256
        "--n_heads"
            help = "Number of attention heads"
            arg_type = Int
            default = 4
        "--n_layers"
            help = "Number of transformer layers"
            arg_type = Int
            default = 6
        "--sequence_length"
            help = "Length of the input sequences"
            arg_type = Int
            default = 64
        "--batchsize"
            help = "Batch size for training"
            arg_type = Int
            default = 128
        "--dropout_rate"
            help = "Dropout rate"
            arg_type = Float32
            default = 0.0f0
        "--test_split"
            help = "Fraction of data to use for testing"
            arg_type = Float64
            default = 0.1
        "--lr"
            help = "Learning rate"
            arg_type = Float64
            default = 1e-2
        "--epochs"
            help = "Number of training epochs"
            arg_type = Int
            default = 100
        "--inference"
            help = "Enable inference mode"
            arg_type = Bool
            default = false
        "--model_path"
            help = "Path to the model checkpoint"
            arg_type = String
            default = ""
        "--seed"
            help = "Seed text for generation"
            arg_type = String
            default = "_The Julia Programming Language is a"
        "--max_output_length"
            help = "Maximum length of the generated output"
            arg_type = Int
            default = 1024
    end
    #! format: on
    return parse_args(ARGS, settings)
end

function main()
    parsed_args = parse_command_line_arguments()

    rng = Random.default_rng()
    Random.seed!(rng, 1234)

    dev = reactant_device(; force=true)
    cdev = cpu_device()

    if parsed_args["inference"]
        error("TODO: implement this path")
        # @printf "[Info] Inference mode enabled.\n"

        # @assert !isempty(model_path) "Please provide a path to a model checkpoint."

        # @printf "[Info] Loading model from %s.\n" model_path
        # model_config = JLD2.load(model_path, "model_config")
        # model = GPT(; model_config...)
        # ps = JLD2.load(model_path, "parameters")
        # st = JLD2.load(model_path, "states")
        # alphabet = JLD2.load(model_path, "alphabet")
        # sequence_length = model_config.sequence_length

        # texts = generate_text(model, ps, st, seed; alphabet, output_length, sequence_length)

        # for (i, (text, s)) in enumerate(zip(texts, seed))
        #     @printf "[Info] Seed [%d]: %s\n" i s
        #     @printf "[Generated Text] %s\n\n" text
        # end

        # return nothing
    end

    n_vocab, (trainX, trainY), (testX, testY) = get_nanogpt_data(;
        sequence_length=parsed_args["sequence_length"], test_split=parsed_args["test_split"]
    )

    @printf "[Info] Vocabulary size: %d\n" n_vocab
    @printf "[Info] Training size: %d sequences.\n" size(trainX, 2)
    @printf "[Info] Testing  size: %d sequences.\n\n" size(testX, 2)

    train_loader =
        DataLoader(
            (trainX, trainY);
            batchsize=parsed_args["batchsize"],
            shuffle=true,
            parallel=true,
        ) |> dev

    model_config = (;
        n_vocab,
        embed_dim=parsed_args["embedding_dim"],
        hidden_dim=parsed_args["n_hidden"],
        n_layers=parsed_args["n_layers"],
        dropout_rate=parsed_args["dropout_rate"],
        num_heads=parsed_args["n_heads"],
        block_size=parsed_args["sequence_length"],
    )

    model = GPT2(; model_config...)
    ps, st = Lux.setup(rng, model) |> dev
    @printf "[Info] Number of parameters: %d\n" Lux.parameterlength(ps)
    @printf "[Info] Number of states: %d\n\n" Lux.statelength(st)

    opt = Adam(parsed_args["lr"])
    train_state = Training.TrainState(model, ps, st, opt)

    ## TODO: implement lr scheduler

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
