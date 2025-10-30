import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.DyLrLAX_.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="Dataset" tabindex="-1">Dataset <a class="header-anchor" href="#Dataset" aria-label="Permalink to &quot;Dataset {#Dataset}&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.43/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.43/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.43/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.43/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    result </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ifelse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">zero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(result), result)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> binarycrossentropy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> eps</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@.</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_true, y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_pred))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> binarycrossentropy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y), st, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_pred)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            gs, loss, _, train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">compute_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), compute_loss, (x, y), train_state)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">apply_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state, gs)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss, st_, ret </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ret</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_pred, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56132</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51074</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48181</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44810</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42317</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40091</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39723</span></span>
<span class="line"><span>Validation: Loss 0.37201 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36918</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34622</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34155</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31626</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29674</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28854</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27443</span></span>
<span class="line"><span>Validation: Loss 0.26064 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26549 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25732</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24376</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23882</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22258</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20766</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19906</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18394</span></span>
<span class="line"><span>Validation: Loss 0.18155 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18416 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17947</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17133</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16255</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15358</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14881</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14334</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13706</span></span>
<span class="line"><span>Validation: Loss 0.12954 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13143 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12712</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12359</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11750</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11399</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10623</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10050</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09945</span></span>
<span class="line"><span>Validation: Loss 0.09426 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09584 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09248</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08818</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08470</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08173</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07955</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07632</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07354</span></span>
<span class="line"><span>Validation: Loss 0.06968 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07116 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07120</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06517</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06348</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06015</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05742</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05560</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05251</span></span>
<span class="line"><span>Validation: Loss 0.05208 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05340 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05130</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04879</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04860</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04503</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04272</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04154</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04257</span></span>
<span class="line"><span>Validation: Loss 0.03926 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04047 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03865</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03883</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03406</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03526</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03231</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03198</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02734</span></span>
<span class="line"><span>Validation: Loss 0.03004 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03113 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02842</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02839</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02681</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02614</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02555</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02628</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02475</span></span>
<span class="line"><span>Validation: Loss 0.02369 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02468 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02226</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02295</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02330</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02098</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01924</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02010</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01953</span></span>
<span class="line"><span>Validation: Loss 0.01930 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02017 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01858</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01843</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01740</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01764</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01674</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01744</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01540</span></span>
<span class="line"><span>Validation: Loss 0.01623 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01699 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01604</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01572</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01421</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01462</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01435</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01519</span></span>
<span class="line"><span>Validation: Loss 0.01402 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01468 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01406</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01348</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01314</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01201</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01241</span></span>
<span class="line"><span>Validation: Loss 0.01235 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01294 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01267</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01041</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01179</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01123</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01224</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.00970</span></span>
<span class="line"><span>Validation: Loss 0.01105 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01159 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01048</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01004</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00966</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01046</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01050</span></span>
<span class="line"><span>Validation: Loss 0.01000 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01050 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00910</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00989</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00882</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00967</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00916</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00960</span></span>
<span class="line"><span>Validation: Loss 0.00913 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00959 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00940</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00848</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00821</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00882</span></span>
<span class="line"><span>Validation: Loss 0.00839 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00881 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00732</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00635</span></span>
<span class="line"><span>Validation: Loss 0.00775 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00814 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00704</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00733</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00723</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00689</span></span>
<span class="line"><span>Validation: Loss 0.00720 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00757 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00718</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00687</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00624</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00673</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00611</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00645</span></span>
<span class="line"><span>Validation: Loss 0.00671 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00705 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00660</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00629</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00628</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00633</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00642</span></span>
<span class="line"><span>Validation: Loss 0.00628 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00660 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00638</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00584</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00602</span></span>
<span class="line"><span>Validation: Loss 0.00589 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00620 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00569</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00543</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00542</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00572</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00517</span></span>
<span class="line"><span>Validation: Loss 0.00554 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00583 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00543</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00507</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00534</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00530</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00501</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00553</span></span>
<span class="line"><span>Validation: Loss 0.00523 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00550 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56320</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50688</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47485</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44997</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41991</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40964</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38900</span></span>
<span class="line"><span>Validation: Loss 0.36221 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36008 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36470</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35689</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33001</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31333</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30289</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28937</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26127</span></span>
<span class="line"><span>Validation: Loss 0.25398 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25250 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25480</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24259</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23591</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21819</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21153</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19697</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18812</span></span>
<span class="line"><span>Validation: Loss 0.17751 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17659 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18090</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17095</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16177</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15417</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14574</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13981</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13289</span></span>
<span class="line"><span>Validation: Loss 0.12647 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12583 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12735</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12259</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11597</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11090</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10527</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10102</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09435</span></span>
<span class="line"><span>Validation: Loss 0.09151 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09093 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09252</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08814</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08369</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08165</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07729</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07459</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07101</span></span>
<span class="line"><span>Validation: Loss 0.06708 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06651 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06971</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06537</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06160</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05999</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05670</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05536</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05269</span></span>
<span class="line"><span>Validation: Loss 0.04978 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04925 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05002</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04987</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04692</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04414</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04299</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04186</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04046</span></span>
<span class="line"><span>Validation: Loss 0.03729 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03681 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03880</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03559</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03756</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03449</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03111</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03128</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03077</span></span>
<span class="line"><span>Validation: Loss 0.02838 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02795 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02943</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02718</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02736</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02627</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02596</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02498</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02429</span></span>
<span class="line"><span>Validation: Loss 0.02226 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02189 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02352</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02248</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02078</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02137</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02084</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01982</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01858</span></span>
<span class="line"><span>Validation: Loss 0.01809 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01778 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01843</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01829</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01879</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01726</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01697</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01652</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01521</span></span>
<span class="line"><span>Validation: Loss 0.01520 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01492 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01684</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01501</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01435</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01533</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01344</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01377</span></span>
<span class="line"><span>Validation: Loss 0.01313 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01290 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01305</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01367</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01375</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01357</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01450</span></span>
<span class="line"><span>Validation: Loss 0.01158 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01137 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01164</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01214</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01183</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01025</span></span>
<span class="line"><span>Validation: Loss 0.01036 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01017 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01060</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00988</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00985</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00960</span></span>
<span class="line"><span>Validation: Loss 0.00937 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00920 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00905</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00945</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00945</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00902</span></span>
<span class="line"><span>Validation: Loss 0.00856 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00839 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00911</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00860</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00882</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00779</span></span>
<span class="line"><span>Validation: Loss 0.00786 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00771 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00817</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00768</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00694</span></span>
<span class="line"><span>Validation: Loss 0.00726 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00713 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00718</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00744</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00734</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00769</span></span>
<span class="line"><span>Validation: Loss 0.00675 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00661 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00700</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00657</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00678</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00580</span></span>
<span class="line"><span>Validation: Loss 0.00629 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00617 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00653</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00677</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00622</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00622</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00566</span></span>
<span class="line"><span>Validation: Loss 0.00588 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00576 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00624</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00600</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00630</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00605</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00646</span></span>
<span class="line"><span>Validation: Loss 0.00552 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00541 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00566</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00526</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00602</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00526</span></span>
<span class="line"><span>Validation: Loss 0.00519 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00509 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00518</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00533</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00529</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00532</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00494</span></span>
<span class="line"><span>Validation: Loss 0.00490 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00480 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxAMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Julia Version 1.10.3</span></span>
<span class="line"><span>Commit 0b4590a5507 (2024-04-30 10:59 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 8 default, 0 interactive, 4 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_AMDGPU_LOGGING_ENABLED = true</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 8</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.4, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.4</span></span>
<span class="line"><span>NVIDIA driver 550.54.15</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.4.5</span></span>
<span class="line"><span>- CURAND: 10.3.5</span></span>
<span class="line"><span>- CUFFT: 11.2.1</span></span>
<span class="line"><span>- CUSOLVER: 11.6.1</span></span>
<span class="line"><span>- CUSPARSE: 12.3.1</span></span>
<span class="line"><span>- CUPTI: 22.0.0</span></span>
<span class="line"><span>- NVML: 12.0.0+550.54.15</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.3.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.8.1+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.12.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.3</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.379 GiB / 4.750 GiB available)</span></span>
<span class="line"><span>┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.</span></span>
<span class="line"><span>└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,k,c,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
