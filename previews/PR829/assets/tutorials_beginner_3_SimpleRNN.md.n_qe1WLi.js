import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.GYfaOXHm.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mROCm MIOpen is not available for AMDGPU.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mNNlib has limited functionality for AMDGPU.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ NNlibAMDGPUExt ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/NNlib/PmySZ/ext/NNlibAMDGPUExt/NNlibAMDGPUExt.jl:58\x1B[39m</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR829/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR829/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR829/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR829/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56327</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51215</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47503</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44794</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42969</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40057</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38346</span></span>
<span class="line"><span>Validation: Loss 0.37245 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37788 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36474</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35310</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33224</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31717</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29575</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28970</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27506</span></span>
<span class="line"><span>Validation: Loss 0.26033 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26321 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25888</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24684</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22931</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21883</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20477</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20230</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18645</span></span>
<span class="line"><span>Validation: Loss 0.18062 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18172 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17968</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17295</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16092</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15277</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14722</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13950</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13131</span></span>
<span class="line"><span>Validation: Loss 0.12822 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12881 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12512</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12339</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11563</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11352</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10493</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09934</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09241</span></span>
<span class="line"><span>Validation: Loss 0.09309 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09132</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08870</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08311</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07948</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07774</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07588</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06982</span></span>
<span class="line"><span>Validation: Loss 0.06885 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06974 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06800</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06540</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06346</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05856</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05578</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05615</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05111</span></span>
<span class="line"><span>Validation: Loss 0.05154 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05246 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04987</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04908</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04749</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04515</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04116</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04168</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03993</span></span>
<span class="line"><span>Validation: Loss 0.03893 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03982 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03657</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03697</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03305</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03498</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03297</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03295</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02973</span></span>
<span class="line"><span>Validation: Loss 0.02986 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03071 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02961</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02808</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02684</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02678</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02343</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02536</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02332</span></span>
<span class="line"><span>Validation: Loss 0.02353 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02429 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02112</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02281</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02194</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02259</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02016</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01894</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01822</span></span>
<span class="line"><span>Validation: Loss 0.01918 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01984 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01802</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01783</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01792</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01692</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01731</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01640</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01745</span></span>
<span class="line"><span>Validation: Loss 0.01613 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01670 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01550</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01419</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01634</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01411</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01315</span></span>
<span class="line"><span>Validation: Loss 0.01392 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01441 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01292</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01289</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01198</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01186</span></span>
<span class="line"><span>Validation: Loss 0.01226 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01270 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01190</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01135</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01078</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01096</span></span>
<span class="line"><span>Validation: Loss 0.01097 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01137 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00995</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01027</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00997</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00985</span></span>
<span class="line"><span>Validation: Loss 0.00993 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01029 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00926</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00945</span></span>
<span class="line"><span>Validation: Loss 0.00907 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00940 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00823</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00824</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00824</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00872</span></span>
<span class="line"><span>Validation: Loss 0.00833 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00864 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00768</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00545</span></span>
<span class="line"><span>Validation: Loss 0.00769 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00798 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00739</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00744</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00716</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00728</span></span>
<span class="line"><span>Validation: Loss 0.00715 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00742 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00720</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00666</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00641</span></span>
<span class="line"><span>Validation: Loss 0.00667 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00692 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00636</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00583</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00664</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00640</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00618</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00611</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00726</span></span>
<span class="line"><span>Validation: Loss 0.00624 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00648 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00601</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00569</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00617</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00590</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00586</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00578</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00579</span></span>
<span class="line"><span>Validation: Loss 0.00585 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00608 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00594</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00514</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00594</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00499</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00480</span></span>
<span class="line"><span>Validation: Loss 0.00550 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00572 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00518</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00557</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00526</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00519</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00516</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00486</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00555</span></span>
<span class="line"><span>Validation: Loss 0.00519 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00540 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56249</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51242</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47692</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45498</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42312</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40721</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38806</span></span>
<span class="line"><span>Validation: Loss 0.36891 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36630 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37219</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34620</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33906</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31536</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30427</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28647</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27251</span></span>
<span class="line"><span>Validation: Loss 0.25773 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25657 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26122</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24823</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23377</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21649</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21036</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19932</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18655</span></span>
<span class="line"><span>Validation: Loss 0.17939 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17917 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18068</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17369</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16113</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15370</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14785</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14184</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13328</span></span>
<span class="line"><span>Validation: Loss 0.12770 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12764 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12986</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12210</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11838</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10997</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10694</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10065</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09506</span></span>
<span class="line"><span>Validation: Loss 0.09266 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09248 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09263</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08941</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08554</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08019</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07866</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07533</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07131</span></span>
<span class="line"><span>Validation: Loss 0.06831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06794 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06963</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06561</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06369</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06005</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05601</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05681</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05265</span></span>
<span class="line"><span>Validation: Loss 0.05099 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05055 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05101</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04778</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04803</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04560</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04460</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04135</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03952</span></span>
<span class="line"><span>Validation: Loss 0.03841 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03794 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03697</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03666</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03547</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03462</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03569</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03195</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02671</span></span>
<span class="line"><span>Validation: Loss 0.02939 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02894 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02954</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02698</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02907</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02604</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02684</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02447</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02214</span></span>
<span class="line"><span>Validation: Loss 0.02316 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02274 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02220</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02252</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02114</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02185</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02174</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02003</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01938</span></span>
<span class="line"><span>Validation: Loss 0.01888 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01851 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01777</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01795</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01907</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01698</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01733</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01709</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01811</span></span>
<span class="line"><span>Validation: Loss 0.01586 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01555 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01570</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01637</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01418</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01456</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01466</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01517</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01355</span></span>
<span class="line"><span>Validation: Loss 0.01368 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01341 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01316</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01334</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01319</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01193</span></span>
<span class="line"><span>Validation: Loss 0.01206 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01182 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01344</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01171</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01078</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01100</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01159</span></span>
<span class="line"><span>Validation: Loss 0.01078 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01057 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01005</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01017</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01070</span></span>
<span class="line"><span>Validation: Loss 0.00975 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00955 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01010</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00976</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00916</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00988</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00749</span></span>
<span class="line"><span>Validation: Loss 0.00890 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00872 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00884</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00757</span></span>
<span class="line"><span>Validation: Loss 0.00817 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00801 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00841</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00833</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00792</span></span>
<span class="line"><span>Validation: Loss 0.00756 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00740 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00712</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00743</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00737</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00731</span></span>
<span class="line"><span>Validation: Loss 0.00701 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00687 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00696</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00697</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00656</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00690</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00637</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00625</span></span>
<span class="line"><span>Validation: Loss 0.00653 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00640 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00627</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00663</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00638</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00630</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00653</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00633</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00619</span></span>
<span class="line"><span>Validation: Loss 0.00611 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00598 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00610</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00634</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00612</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00578</span></span>
<span class="line"><span>Validation: Loss 0.00573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00561 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00596</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00590</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00529</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00541</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00495</span></span>
<span class="line"><span>Validation: Loss 0.00539 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00528 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00524</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00535</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00511</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00535</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00483</span></span>
<span class="line"><span>Validation: Loss 0.00509 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00498 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
