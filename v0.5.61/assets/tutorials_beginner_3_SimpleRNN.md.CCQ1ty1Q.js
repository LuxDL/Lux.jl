import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.DIjdMxER.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mROCm MIOpen is not available for AMDGPU.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mNNlib has limited functionality for AMDGPU.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ NNlibAMDGPUExt ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/NNlib/eQ6vp/ext/NNlibAMDGPUExt/NNlibAMDGPUExt.jl:58\x1B[39m</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.61/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.61/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.61/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.61/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56350</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51453</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47433</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44137</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43006</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40833</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38753</span></span>
<span class="line"><span>Validation: Loss 0.37580 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36346 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37355</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34486</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33399</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30980</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30326</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29391</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26928</span></span>
<span class="line"><span>Validation: Loss 0.26247 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25448 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25137</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24501</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23609</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21788</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21389</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19847</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19351</span></span>
<span class="line"><span>Validation: Loss 0.18215 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17772 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18062</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17237</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16286</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15353</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14761</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14001</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13638</span></span>
<span class="line"><span>Validation: Loss 0.12937 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12672 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12760</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12263</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11618</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11112</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10528</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10329</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09882</span></span>
<span class="line"><span>Validation: Loss 0.09414 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09183 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09267</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08960</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08629</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08112</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07771</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07359</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07208</span></span>
<span class="line"><span>Validation: Loss 0.06969 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06744 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06837</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06402</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06285</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06195</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05720</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05667</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05286</span></span>
<span class="line"><span>Validation: Loss 0.05227 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05007 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05113</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04863</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04724</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04539</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04323</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04140</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04206</span></span>
<span class="line"><span>Validation: Loss 0.03957 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03750 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03751</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03735</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03606</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03371</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03195</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03359</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02880</span></span>
<span class="line"><span>Validation: Loss 0.03040 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02851 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02957</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02908</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02807</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02736</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02556</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02278</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02108</span></span>
<span class="line"><span>Validation: Loss 0.02398 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02230 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02355</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02224</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02202</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02191</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01987</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01932</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01776</span></span>
<span class="line"><span>Validation: Loss 0.01958 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01809 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01934</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01761</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01772</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01776</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01770</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01560</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01623</span></span>
<span class="line"><span>Validation: Loss 0.01648 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01517 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01585</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01504</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01411</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01312</span></span>
<span class="line"><span>Validation: Loss 0.01423 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01309 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01368</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01239</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01217</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01467</span></span>
<span class="line"><span>Validation: Loss 0.01255 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01152 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01204</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01171</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01001</span></span>
<span class="line"><span>Validation: Loss 0.01122 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01030 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01095</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01067</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01007</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01022</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00931</span></span>
<span class="line"><span>Validation: Loss 0.01015 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00931 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00897</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00947</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00975</span></span>
<span class="line"><span>Validation: Loss 0.00928 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00850 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00860</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00877</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00889</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00770</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00825</span></span>
<span class="line"><span>Validation: Loss 0.00852 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00780 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00760</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00745</span></span>
<span class="line"><span>Validation: Loss 0.00788 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00721 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00840</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00730</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00744</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00680</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00716</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00690</span></span>
<span class="line"><span>Validation: Loss 0.00732 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00669 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00694</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00677</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00691</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00677</span></span>
<span class="line"><span>Validation: Loss 0.00682 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00623 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00652</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00671</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00626</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00639</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00630</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00598</span></span>
<span class="line"><span>Validation: Loss 0.00638 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00583 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00641</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00566</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00581</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00622</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00563</span></span>
<span class="line"><span>Validation: Loss 0.00599 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00547 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00520</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00584</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00583</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00559</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00547</span></span>
<span class="line"><span>Validation: Loss 0.00564 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00515 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00581</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00509</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00463</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00483</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00483</span></span>
<span class="line"><span>Validation: Loss 0.00532 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00485 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56286</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50983</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46973</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44483</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42347</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40620</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38588</span></span>
<span class="line"><span>Validation: Loss 0.36146 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37129 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36281</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35283</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32973</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31831</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29661</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28635</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26714</span></span>
<span class="line"><span>Validation: Loss 0.25263 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25854 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25712</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23973</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23282</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21882</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20755</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19662</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18458</span></span>
<span class="line"><span>Validation: Loss 0.17590 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17882 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17560</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17018</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16171</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15266</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14415</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13940</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13286</span></span>
<span class="line"><span>Validation: Loss 0.12478 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12671 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12508</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12120</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11532</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10921</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10483</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09907</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09294</span></span>
<span class="line"><span>Validation: Loss 0.09004 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09179 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09041</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08713</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08328</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07901</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07604</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07409</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07091</span></span>
<span class="line"><span>Validation: Loss 0.06590 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06775 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06764</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06543</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06177</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05778</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05483</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05453</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05193</span></span>
<span class="line"><span>Validation: Loss 0.04888 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05062 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04909</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04956</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04446</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04449</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04121</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04167</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03847</span></span>
<span class="line"><span>Validation: Loss 0.03659 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03824 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03732</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03579</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03477</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03416</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03260</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02985</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02901</span></span>
<span class="line"><span>Validation: Loss 0.02785 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02938 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02920</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02880</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02628</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02451</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02573</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02319</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02323</span></span>
<span class="line"><span>Validation: Loss 0.02185 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02321 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02243</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02211</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02129</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01975</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02010</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01959</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02029</span></span>
<span class="line"><span>Validation: Loss 0.01778 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01896 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01829</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01772</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01710</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01737</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01686</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01621</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01579</span></span>
<span class="line"><span>Validation: Loss 0.01494 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01596 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01511</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01551</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01540</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01298</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01383</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01418</span></span>
<span class="line"><span>Validation: Loss 0.01290 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01379 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01283</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01277</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01279</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01181</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01042</span></span>
<span class="line"><span>Validation: Loss 0.01138 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01217 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01179</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01131</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01083</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01115</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01047</span></span>
<span class="line"><span>Validation: Loss 0.01019 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01091 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01097</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01010</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01071</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00997</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00993</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01047</span></span>
<span class="line"><span>Validation: Loss 0.00923 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00988 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00909</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00955</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00953</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00891</span></span>
<span class="line"><span>Validation: Loss 0.00843 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00903 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00841</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00809</span></span>
<span class="line"><span>Validation: Loss 0.00774 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00830 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00759</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00851</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00717</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00753</span></span>
<span class="line"><span>Validation: Loss 0.00715 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00767 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00712</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00692</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00693</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00684</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00677</span></span>
<span class="line"><span>Validation: Loss 0.00664 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00713 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00688</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00671</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00651</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00694</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00666</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00648</span></span>
<span class="line"><span>Validation: Loss 0.00619 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00665 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00680</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00678</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00627</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00600</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00641</span></span>
<span class="line"><span>Validation: Loss 0.00579 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00622 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00581</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00629</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00552</span></span>
<span class="line"><span>Validation: Loss 0.00544 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00584 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00552</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00534</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00535</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00504</span></span>
<span class="line"><span>Validation: Loss 0.00511 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00550 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00534</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00522</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00520</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00524</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00527</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00487</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00537</span></span>
<span class="line"><span>Validation: Loss 0.00483 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00519 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.4</span></span>
<span class="line"><span>Commit 48d4fd48430 (2024-06-04 10:41 UTC)</span></span>
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
<span class="line"><span>  JULIA_AMDGPU_LOGGING_ENABLED = true</span></span>
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
<span class="line"><span>- CUDA_Driver_jll: 0.9.1+1</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.484 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
