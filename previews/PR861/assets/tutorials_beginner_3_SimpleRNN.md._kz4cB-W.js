import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.0PdpScDf.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mROCm MIOpen is not available for AMDGPU.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mNNlib has limited functionality for AMDGPU.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ NNlibAMDGPUExt ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/NNlib/f92hx/ext/NNlibAMDGPUExt/NNlibAMDGPUExt.jl:58\x1B[39m</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR861/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR861/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR861/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR861/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56404</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51129</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47968</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45363</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43102</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40468</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38505</span></span>
<span class="line"><span>Validation: Loss 0.37549 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35977 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38832</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34728</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33769</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30749</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30834</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28470</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27471</span></span>
<span class="line"><span>Validation: Loss 0.26364 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25416 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25645</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25104</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23383</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22156</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21319</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20380</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18924</span></span>
<span class="line"><span>Validation: Loss 0.18415 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18407</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17354</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16441</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15809</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14791</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14417</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13835</span></span>
<span class="line"><span>Validation: Loss 0.13168 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12842 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12979</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12480</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11908</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11559</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10877</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10300</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09894</span></span>
<span class="line"><span>Validation: Loss 0.09622 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09330 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09656</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08984</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08568</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08358</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07918</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07943</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07158</span></span>
<span class="line"><span>Validation: Loss 0.07142 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06835 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07078</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06750</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06472</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06231</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05989</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05666</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05298</span></span>
<span class="line"><span>Validation: Loss 0.05359 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05054 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05404</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04987</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04783</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04558</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04431</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04377</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04230</span></span>
<span class="line"><span>Validation: Loss 0.04052 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03762 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04124</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03815</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03617</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03394</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03489</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03064</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03494</span></span>
<span class="line"><span>Validation: Loss 0.03105 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02841 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03004</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02781</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02704</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02766</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02794</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02557</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02443</span></span>
<span class="line"><span>Validation: Loss 0.02444 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02212 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02440</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02225</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02208</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02137</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02230</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02009</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01771</span></span>
<span class="line"><span>Validation: Loss 0.01988 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01788 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01857</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01878</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01829</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01850</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01774</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01625</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01793</span></span>
<span class="line"><span>Validation: Loss 0.01673 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01497 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01658</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01560</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01477</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01490</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01528</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01389</span></span>
<span class="line"><span>Validation: Loss 0.01443 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01289 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01352</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01311</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01143</span></span>
<span class="line"><span>Validation: Loss 0.01271 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01134 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01184</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01220</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01204</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01090</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01377</span></span>
<span class="line"><span>Validation: Loss 0.01138 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01013 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01071</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01052</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01108</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01067</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00947</span></span>
<span class="line"><span>Validation: Loss 0.01028 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00915 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00884</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01032</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00861</span></span>
<span class="line"><span>Validation: Loss 0.00938 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00834 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00905</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00893</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00899</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01021</span></span>
<span class="line"><span>Validation: Loss 0.00862 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00765 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00828</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00749</span></span>
<span class="line"><span>Validation: Loss 0.00796 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00705 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00786</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00787</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00694</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00709</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00770</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00745</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00719</span></span>
<span class="line"><span>Validation: Loss 0.00738 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00654 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00698</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00715</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00723</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00660</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00660</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00710</span></span>
<span class="line"><span>Validation: Loss 0.00688 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00609 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00653</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00677</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00677</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00670</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00631</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00569</span></span>
<span class="line"><span>Validation: Loss 0.00643 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00569 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00666</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00627</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00584</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00612</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00588</span></span>
<span class="line"><span>Validation: Loss 0.00604 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00533 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00578</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00589</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00531</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00537</span></span>
<span class="line"><span>Validation: Loss 0.00568 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00502 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00507</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00566</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00526</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00547</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00522</span></span>
<span class="line"><span>Validation: Loss 0.00536 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00473 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56109</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50793</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48056</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46682</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42926</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39476</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41121</span></span>
<span class="line"><span>Validation: Loss 0.37486 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35448 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36416</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35938</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33604</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32641</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30335</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29135</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27271</span></span>
<span class="line"><span>Validation: Loss 0.26311 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25222 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25972</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24797</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23639</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22483</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21349</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20487</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19648</span></span>
<span class="line"><span>Validation: Loss 0.18436 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17937 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18629</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17384</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16615</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15959</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15279</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14523</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13823</span></span>
<span class="line"><span>Validation: Loss 0.13261 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12978 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13063</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12709</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12087</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11687</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11106</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10626</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10496</span></span>
<span class="line"><span>Validation: Loss 0.09746 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09443 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09630</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09306</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09042</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08612</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08239</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07805</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07401</span></span>
<span class="line"><span>Validation: Loss 0.07255 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06918 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07381</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07044</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06525</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06370</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06253</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05625</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05538</span></span>
<span class="line"><span>Validation: Loss 0.05448 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05089 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05404</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05213</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04796</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04791</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04755</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04397</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04296</span></span>
<span class="line"><span>Validation: Loss 0.04117 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03754 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04094</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03833</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03732</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03682</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03518</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03350</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03136</span></span>
<span class="line"><span>Validation: Loss 0.03153 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02810 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03243</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03003</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02846</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02737</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02762</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02534</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02367</span></span>
<span class="line"><span>Validation: Loss 0.02484 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02174 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02341</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02466</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02316</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02088</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02334</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02049</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01970</span></span>
<span class="line"><span>Validation: Loss 0.02023 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01751 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01908</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02025</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01932</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01831</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01810</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01705</span></span>
<span class="line"><span>Validation: Loss 0.01700 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01462 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01620</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01515</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01492</span></span>
<span class="line"><span>Validation: Loss 0.01466 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01257 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01387</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01315</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01430</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01324</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01310</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01321</span></span>
<span class="line"><span>Validation: Loss 0.01291 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01103 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01198</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01287</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01192</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01244</span></span>
<span class="line"><span>Validation: Loss 0.01153 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00984 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01121</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01080</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01050</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01061</span></span>
<span class="line"><span>Validation: Loss 0.01042 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00887 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01004</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01048</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00934</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01051</span></span>
<span class="line"><span>Validation: Loss 0.00949 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00945</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00889</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00908</span></span>
<span class="line"><span>Validation: Loss 0.00871 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00739 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00823</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00789</span></span>
<span class="line"><span>Validation: Loss 0.00803 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00681 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00768</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00720</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00768</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00731</span></span>
<span class="line"><span>Validation: Loss 0.00746 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00631 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00760</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00703</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00718</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00678</span></span>
<span class="line"><span>Validation: Loss 0.00694 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00587 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00659</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00660</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00659</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00640</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00663</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00597</span></span>
<span class="line"><span>Validation: Loss 0.00649 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00548 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00625</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00676</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00619</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00603</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00608</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00597</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00575</span></span>
<span class="line"><span>Validation: Loss 0.00609 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00513 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00558</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00594</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00578</span></span>
<span class="line"><span>Validation: Loss 0.00573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00482 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00515</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00583</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00515</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00564</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00547</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00514</span></span>
<span class="line"><span>Validation: Loss 0.00540 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00454 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxDeviceUtils)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.5</span></span>
<span class="line"><span>Commit 6f3fdf7b362 (2024-08-27 14:19 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 4 default, 0 interactive, 2 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 4</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.3</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.3</span></span>
<span class="line"><span>- CUSPARSE: 12.5.1</span></span>
<span class="line"><span>- CUPTI: 2024.2.1 (API 23.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.2+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.5</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
