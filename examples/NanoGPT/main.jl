# Taken from https://github.com/FluxML/model-zoo/pull/410
using MLUtils, Lux, Random, Optimisers, Printf, Statistics, NNlib, DataDeps, StatsBase,
      OneHotArrays
using Reactant, Enzyme
using Comonicon: @main

if !haskey(DataDeps.registry, "nanogpt")
    register(DataDep(
        "nanogpt",
        "Shakespeare Input Text for training NanoGPT",
        "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
        "59a0ad62833b2e15ec811c548618876359e902717431236e52699a0e2bc253ca"
    ))
end

function gpt_block(; n_embed, n_hidden, qk_dim, v_dim, n_heads, dropout_rate)
    @assert qk_dim % n_heads == 0
    @assert v_dim % n_heads == 0
    return @compact(;
        name="GPTBlock(; n_embed=$n_embed, n_hidden=$n_hidden, qk_dim=$qk_dim, v_dim=$v_dim, n_heads=$n_heads, dropout_rate=$dropout_rate)",
        ln=LayerNorm((n_embed,)),
        qlayer=Dense(n_embed => qk_dim; use_bias=false),
        klayer=Dense(n_embed => qk_dim; use_bias=false),
        vlayer=Dense(n_embed => v_dim; use_bias=false),
        attn_drop=Dropout(dropout_rate),
        proj=Dense(v_dim => n_embed; use_bias=false),
        mlp=Chain(
            LayerNorm((n_embed,)),
            Dense(n_embed => n_hidden, gelu),
            Dense(n_hidden => n_embed),
            Dropout(dropout_rate)
        )) do x
        qkv_in = ln(x)
        q = qlayer(qkv_in)
        k = klayer(qkv_in)
        v = vlayer(qkv_in)
        mha, _ = NNlib.dot_product_attention(
            q, k, v, nothing; mask=NNlib.make_causal_mask(x)
        )
        x = x .+ proj(mha)
        x = x .+ mlp(x)
        @return x
    end
end

function GPT(;
        n_vocab, n_embed, sequence_length, n_hidden,
        n_layers, dropout_rate, n_heads, qk_dim, v_dim
)
    return @compact(;
        token_embedding=Embedding(n_vocab => n_embed),
        position_embedding=Embedding(sequence_length => n_embed),
        drop=Dropout(dropout_rate),
        blocks=ntuple(n_layers) do i
            return gpt_block(; n_embed, n_hidden, qk_dim, v_dim, n_heads, dropout_rate)
        end,
        ln=LayerNorm((n_embed,)),
        output_layer=Dense(n_embed => n_vocab)) do tokens
        te = token_embedding(tokens)
        pe = position_embedding(1:size(tokens, 1))
        x = drop(te .+ pe)
        for blk in blocks
            x = blk(x)
        end
        x = ln(x)
        x = output_layer(x)
        @return x
    end
end

# Use the model to generate some text.
# function generate(model, seed, outlen)
#     seqlen = context_length(model)
#     if isempty(seed)
#         seed = "_"
#     end
#     x = map(c -> findfirst(==(c), model.alphabet)::Int64, collect(seed))
#     while length(x) < outlen
#         tail = x[max(1, end-seqlen+1):end]
#         tail = reshape(tail, length(tail), 1)
#         y = model(tail |> device) |> cpu
#         p = softmax(y[:,end,1])
#         j = sample(1:length(model.alphabet), Weights(p))
#         #j = argmax(p)
#         #x = vcat(x, [j])
#         push!(x, j)
#     end
#     String(map(j -> model.alphabet[j], x))
# end

# Load data from input file, and partition into training and testing subsets.
function get_nanogpt_data(; sequence_length, test_split)
    data_file = joinpath(datadep"nanogpt", "shakespeare_input.txt")
    text = String(read(data_file))

    # For aesthetic reasons, replace newlines with strings.  This is not necessary, but makes
    # strings print nicer.
    text = replace(text, r"\r?\n" => " ")

    ## an array of all unique characters
    alphabet = [unique(text)..., '_']
    stop = alphabet[end]

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

@main function train_nanogpt(;
        n_embed::Int=64, n_hidden::Int=256, n_heads::Int=4, qk_dim::Int=16,
        v_dim::Int=16, n_layers::Int=6, sequence_length::Int=64, batchsize::Int=128,
        dropout_rate::Float32=0.0f0, test_split::Float64=0.1, lr::Float64=1e-2,
        epochs::Int=20
)
    alphabet, trainX, trainY, testX, testY = get_nanogpt_data(; sequence_length, test_split)

    @printf "[Info] Alphabet size: %d\n" length(alphabet)
    @printf "[Info] Training size: %d sequences.\n" size(trainX, 2)
    @printf "[Info] Testing  size: %d sequences.\n\n" size(testX, 2)

    rng = Random.default_rng()
    Random.seed!(rng, 1234)

    dev = reactant_device()
    cdev = cpu_device()

    train_loader = DataLoader(
        (trainX, trainY); batchsize, shuffle=true, parallel=true
    ) |> dev

    model = GPT(;
        n_vocab=length(alphabet), n_embed, sequence_length, n_hidden,
        n_layers, dropout_rate, n_heads, qk_dim, v_dim
    )
    ps, st = Lux.setup(rng, model) |> dev
    @printf "[Info] Number of parameters: %d\n" Lux.parameterlength(ps)
    @printf "[Info] Number of states: %d\n\n" Lux.statelength(st)

    opt = Adam(lr)
    train_state = Training.TrainState(model, ps, st, opt)

    testX, testY = (testX, testY) |> dev
    model_compiled = @compile model(testX, ps, Lux.testmode(st))
    best_test_loss = Inf

    loss_fn = CrossEntropyLoss(; logits=Val(true))

    iter = 0
    for epoch in 1:epochs
        for (x, y) in train_loader
            iter += 1

            _, loss, _, train_state = Training.single_train_step!(
                AutoEnzyme(), loss_fn, (x, y), train_state
            )

            if iter % 100 == 0
                @printf "[Train] Epoch %3d\tIteration %6d\tLoss %.8e\n" epoch iter loss
            end
        end

        test_loss = loss_fn(
            Array(first(model_compiled(testX, ps, Lux.testmode(st)))), testY
        )
        @printf "[Test] Epoch %3d\tTest Loss %.8e\n" epoch test_loss

        # XXX: Also generate some text here...

        if test_loss < best_test_loss
            best_test_loss = test_loss
            @printf "[Info] New best test loss: %.8e\n" best_test_loss
            @printf "[Info] Saving model...\n"
            jldsave(
                joinpath(@__DIR__, "nanogpt.jld2");
                parameters=train_state.parameters |> cdev,
                states=train_state.states |> cdev,
                alphabet=alphabet
            )
        end
    end
end

# # Load a model from a checkpoint (see `jldsave` above).
# function load_model(filename)
#     args = JLD2.load(filename, "args")
#     alphabet = JLD2.load(filename, "alphabet")
#     model = GPT(args, alphabet)
#     model_state = JLD2.load(filename, "model_state")
#     model = Flux.loadmodel!(model, model_state);
#     return args, model
# end

# if true
#     args, model = train()
# else
#     args, model = load_model("model-checkpoint.jld2") |> device
# end

# generate(model, "The", 50)
