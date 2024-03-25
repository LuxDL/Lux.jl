


<a id='Training-a-Character-Level-RNN'></a>

# Training a Character Level RNN


![char-rnn](https://d2l.ai/_images/rnn-train.svg) Source: [Deep Dive Into Deep Learning](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#rnn-based-character-level-language-models)


In this tutorial we will go over Karpathy's character level RNN tutorial. This is not an original text, rather it has been drawn from multiple sources:


  * [Deep Dive Into Deep Learning: Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#rnn-based-character-level-language-models)
  * [Flux: Character Level RNN](https://github.com/FluxML/model-zoo/blob/master/text/char-rnn/char-rnn.jl)
  * [Karpathy's Char RNN in Lua](https://github.com/karpathy/char-rnn)


To understand more about RNNs, have a look into the following resources:


  * [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  * [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  * [Illustrated Guide to Recurrent Neural Networks: Understanding the Intuition](https://www.youtube.com/watch?v=LHXXI4-IEns)


In this example, we create a character-level recurrent neural network. A recurrent neural network (RNN) outputs a prediction and a hidden state at each step of the computation. The hidden state captures historical information of a sequence (i.e., the neural network has memory) and the output is the final prediction of the model. We use this type of neural network to model sequences such as text or time series.


This example demonstrates the use of Lux’s implementation of


  * [`RNNCell`](../../api/Lux/layers#Lux.RNNCell)
  * [`LSTMCell`](../../api/Lux/layers#Lux.LSTMCell)
  * [`GRUCell`](../../api/Lux/layers#Lux.GRUCell)


We won't go into the mathematical details for each of these Cells, but you can read more about them in their respective documentations.


<a id='Package-Imports'></a>

## Package Imports


```julia
using Lux, LuxAMDGPU, LuxCUDA, MLUtils, Optimisers, Zygote, Random, Statistics,
    DataDeps, OneHotArrays, Functors, StatsBase
```


<a id='Hyperparameters'></a>

## Hyperparameters


We set default values for the hyperparameters:


```julia
@enum CellType LSTM GRU RNN

Base.@kwdef mutable struct HyperParameters
    cell_type::CellType = LSTM
    lr::Float64 = 1e-2          # Learning rate
    seqlen::Int = 32            # Length of batch sequences
    batchsize::Int = 32         # Number of sequences in each batch
    epochs::Int = 3             # Number of epochs
    testpercent::Float64 = 0.05 # Percent of corpus examples to use for testing
end
```


```
Main.var"##225".HyperParameters
```


<a id='Data'></a>

## Data


We create the function `getdata` to download the training data and create arrays of batches for training the model.


For this tutorial we could have used Julia's built in `download` function, but we use `DataDeps.jl` for a more principled data handling solution. First we register the datadep.


```julia
register(DataDep("tinyshakespeare",
    "Tiny Shakespeare Corpus from Karpathy's Char RNN",
    "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
    "59a0ad62833b2e15ec811c548618876359e902717431236e52699a0e2bc253ca"))

function getdata(args::HyperParameters)
    text = String(read(joinpath(datadep"tinyshakespeare", "shakespeare_input.txt")))

    # an array of all unique characters
    alphabet = [unique(text)..., '_']
    stop = '_'

    N = length(alphabet)

    # Partitioning the data as sequence of batches, which are then collected
    # as array of batches
    Xs = Iterators.partition(batchseq(collect.(chunk(text, args.batchsize)), stop),
        args.seqlen)
    Ys = Iterators.partition(batchseq(collect.(chunk(text[2:end], args.batchsize)), stop),
        args.seqlen)
    Xs = [onehotbatch.(bs, (alphabet,)) for bs in Xs]
    Ys = [onehotbatch.(bs, (alphabet,)) for bs in Ys]

    return Xs, Ys, N, alphabet
end
```


```
getdata (generic function with 1 method)
```


The function `getdata` performs the following tasks:


  * Downloads a dataset of [all of Shakespeare's works (concatenated)](https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt)
  * Gets the alphabet. It consists of the unique characters of the data and the stop character ‘_’.
  * One-hot encodes the alphabet and the stop character.
  * Gets the size of the alphabet N.
  * Partitions the data as an array of batches. Note that the `Xs` array contains the sequence of characters in the text whereas the `Ys` array contains the next character of the sequence.


<a id='Model'></a>

## Model


We create the RNN with two Lux’s LSTM layers and an output layer of the size of the alphabet.


```julia
function build_model(args::HyperParameters, N::Int)
    cell = if args.cell_type == LSTM
        LSTMCell
    elseif args.cell_type == GRU
        GRUCell
    elseif args.cell_type == RNN
        RNNCell
    end
    return Chain(Recurrence(cell(N => 128); return_sequence=true),
        Recurrence(cell(128 => 128); return_sequence=true),
        x -> stack(x; dims=2), Dense(128 => N),
        x -> unstack(x; dims=2))
end
```


```
build_model (generic function with 1 method)
```


The size of the input and output layers is the same as the size of the alphabet. Also, we set the size of the hidden layers to 128.


<a id='Train-the-model'></a>

## Train the model


Now, we define the function `train` that creates the model and the loss function as well as the training loop:


```julia
function logitcrossentropy((ŷ, y))
    return mean(.-sum(y .* logsoftmax(ŷ; dims=1); dims=1))
end

function compute_loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return sum(logitcrossentropy, zip(y_pred, y)), y_pred, st
end

function train(; kwargs...)
    # Initialize the parameters
    args = HyperParameters(; kwargs...)

    # Select the correct device
    device = gpu_device()
    rng = Xoshiro(0)

    # Get Data
    Xs, Ys, N, alphabet = getdata(args)

    # Shuffle and create a train/test split
    L = length(Xs)
    perm = shuffle(rng, 1:length(Xs))
    trsplit = floor(Int, (1 - args.testpercent) * L)

    trainX, trainY = Xs[perm[1:trsplit]], Ys[perm[1:trsplit]]
    testX, testY = Xs[perm[(trsplit + 1):end]], Ys[perm[(trsplit + 1):end]]

    # Constructing Model
    model = build_model(args, N)
    ps, st = Lux.setup(rng, model) |> device

    # Training
    opt = Adam(args.lr)
    opt_state = Optimisers.setup(opt, ps)

    for epoch in 1:(args.epochs)
        for (i, (x, y)) in enumerate(zip(trainX, trainY))
            x = fmap(x -> device(float(x)), x)
            y = fmap(device, y)
            (loss, y_pred, st), back = pullback(compute_loss, x, y, model, ps, st)
            if !isnan(loss)
                gs = back((one(loss), nothing, nothing))[4]
                Optimisers.update!(opt_state, ps, gs)
                if i % 250 == 0
                    println("Epoch [$epoch]: Iteration [$(i)/$(length(trainX))] Loss $loss")
                end
            else
                error("Iteration [$(i)/$(length(trainX))] has NaNs")
            end
        end

        # Show loss-per-character over the test set
        test_loss = 0.0f0
        st_ = Lux.testmode(st)
        for (x, y) in zip(testX, testY)
            x = fmap(x -> device(float(x)), x)
            y = fmap(device, y)
            test_loss += compute_loss(x, y, model, ps, st_)[1]
        end
        println("Epoch [$epoch]: Test Loss: ",
            test_loss / (length(testX) * args.seqlen * args.batchsize))
    end

    return model, ps, st, alphabet
end
```


```
train (generic function with 1 method)
```


The function `train` performs the following tasks:


  * Calls the function `getdata` to obtain the train and test data as well as the alphabet and its size.
  * Calls the function `build_model` to create the RNN.
  * Defines the loss function. For this type of neural network, we use the logitcrossentropy loss function. Notice that it is important that we call the function reset! before computing the loss so that it resets the hidden state of a recurrent layer back to its original value
  * Sets the AdamW optimizer with the learning rate *lr* we defined above.
  * Runs the training loop using Optimisers and Zygote.


<a id='Test-the-model'></a>

## Test the model


We define the function `sample_data` to test the model. It generates samples of text with the alphabet that the function `getdata` computed. Notice that it obtains the model’s prediction by calling the softmax function to get the probability distribution of the output and then it chooses randomly the prediction.


```julia
function sample_data(m, alphabet, len, ps, st; seed="")
    cdev = cpu_device()
    model = Lux.Experimental.StatefulLuxLayer(m, ps |> cdev, st |> cdev)
    buf = IOBuffer()
    seed == "" && (seed = string(rand(alphabet)))
    write(buf, seed)
    c = StatsBase.wsample(alphabet,
        softmax(model([float.(onehotbatch(collect(seed), alphabet))])[end])[:, 1])
    for i in 1:len
        write(buf, c)
        c = StatsBase.wsample(alphabet,
            softmax(model([float.(onehotbatch([c], alphabet))])[end])[:, 1])
    end
    return String(take!(buf))
end
```


```
sample_data (generic function with 1 method)
```


Finally, to run this example we call the functions `train` and `sample_data`:


First let's try an RNN


```julia
model_rnn, ps_rnn, st_rnn, alphabet = train(; cell_type=RNN)
sample_data(model_rnn, alphabet, 1000, ps_rnn, st_rnn)
```


```
"Noxwwxxrowcww:wowwxwnowowocwxowwwxew:twcwwg,:wxwrwx,w:hw:cxxwwxr:www:www,wswj:wwgrwxwxaiowwowxwsowcoc!xw:w:.woowxw:hwxwow:wot,,ocw,wwwxwow orowxx,owcnw:wxwwoawwrwe,awcoawnwwxwwnwoccxwcoxotr,o:wwrwo!wwxo:wcw,whw:wwxw,oaxwxwwwrxscx,www:hxowooc:w:w,owcwwor:wsoonwcw:wo:swxw:ctowxxwor:wowwwwt:txxorw:wwox:wxxww,:wwww,:wnoowow:www:wwwwowncwcwwwwxswohwx:tocowwxwwxowsoaw:wc,,:c:wxt sxosrwwxwtwwwtxwwwo:twwwooooowkxw:nw:rowwwx,owxrww:::wwwowwwxowwww:wwwwowwowoxwo:wwxxrxwowocwxwwxwowwawwwwx:wxoowwxwxoww,wxwwcww;wwowwwwwowowow,:w,w ooxw:wwxw,wwx::wwxo:r:oxoroxo:wwwxs:rxwsxxswwxc:www,axwoxxw:w:w::wowrwoxwwow:wr how:w:wowwaxwoww:cwh:rwwwownxowwx::wwwxaao,wwxw:wwhowowowwwooowowx:,:txwoowt:swatoxwwrcxnwwgrw:wc::wctwhoww:woswcxtxwwo,awatxrw::w:cxx,owwo:hwwwwnwwwonwatwwxrww,\norww:woxocwtfox:w,oowacohwwxwww,,::ton,ow:gcxwa,oxwowxrrwwoxwss:w,wx:ww,wx:cw,,,owowx,wxcooiwxwooxwo::w.xw,ow,t:wcowoxxoo:w,wwxwwwowwwxwecbowww:wo\nww-nxwrwaxwwww:tww,wxowxoxotoioowwxwwww,wwesbawwwnooaoxrwfowrrrwwoxwwwwwoxocxrxcoow?wxo"
```


Now let's try an LSTM


```julia
model_lstm, ps_lstm, st_lstm, alphabet = train(; cell_type=LSTM)
sample_data(model_lstm, alphabet, 1000, ps_lstm, st_lstm)
```


```
"Je\n\n\n\n\n\n\n\n\n :\n\n\n\n\n\n Je\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\ns\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n:\n\n:\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\ns\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nJD\ns\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n:\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n s\n\n\n\n\n\n \n\n\n \n\n\n\nl\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nS\n\n\n\n\n\n\n\n\n\n\n:\n\n\n\n\n b\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n\n\n\n\nN!\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n Ja\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n?\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
```


Finally let's try a GRU


```julia
model_gru, ps_gru, st_gru, alphabet = train(; cell_type=GRU)
sample_data(model_gru, alphabet, 1000, ps_gru, st_gru)
```


```
"u Amin h and An h ind d A fin ES s A A fu WAn A An\n\n\n\n\n\n\n\n\n\n\nA A By A A A An A A A Angnbmnh onn o h A A A ADin,\n\n\nBy h o h in:\n\n\n\n\n\n\nPy Se A Ann A A h Ann An Neh An s Henjenn Th o finy An s An A d A s By A An A han Anginofnnn'T Any o A Th Anann Th A An? AL'th h\n\nwy A A o Anpy onnn h ojo i An ombt omam By A By A A OL ASin ng A AKnin Anombin Am A ojomy A A A:\n\nAnan A on Th An fn'xu An A ofin os Air o o By h; in h A A A An'CHinnbmin an'Th ominpannn A h\nAnin Annnje Annjoflod ALJomy alan s A ennbbe A fn A: hmban h Aninjind in An A h'Lofn d h, Anjinpy odan Aninn A h An A: ann An fn A ofn Anndis Amn A ha A A An A A Jon h By By h n A ES omby A? iofins aft s An AP An Aninnbmy A A h An'Cobobannoge A Anzofinbofr An A As d Ann bobr An A h, An t An Enzin os ffinas h d oman.\n\n\nAn A ann o o A APHe A: A A An. o An A h A A hy An Py of Ai innve As An Ty A:\n\n\n\nDe A Angobr Annofy An An An Aninpy A A fth S h A Annfy Anifn AWan A Th s Annini Fin obmy An Hd AByn By h o'Pin APy A h By A A AFin-Jo do on'min ofn"
```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

