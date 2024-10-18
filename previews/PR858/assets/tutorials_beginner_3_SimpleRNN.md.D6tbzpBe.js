import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.D-WlR71w.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR858/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR858/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR858/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR858/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56330</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50648</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48151</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43888</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43170</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40101</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38161</span></span>
<span class="line"><span>Validation: Loss 0.36452 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36364 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36060</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35483</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33735</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31367</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30301</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28890</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26650</span></span>
<span class="line"><span>Validation: Loss 0.25545 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25456 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25603</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24703</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22736</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22125</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21113</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19970</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18796</span></span>
<span class="line"><span>Validation: Loss 0.17812 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17745 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18064</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16989</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16347</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15284</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14748</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13931</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13605</span></span>
<span class="line"><span>Validation: Loss 0.12674 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12627 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12830</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12032</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11661</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11100</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10458</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10229</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09759</span></span>
<span class="line"><span>Validation: Loss 0.09186 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09149 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09155</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08936</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08562</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08132</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07709</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07436</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06828</span></span>
<span class="line"><span>Validation: Loss 0.06750 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06721 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06789</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06601</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06393</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05972</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05786</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05457</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.04916</span></span>
<span class="line"><span>Validation: Loss 0.05016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04993 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05040</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04908</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04742</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04693</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04231</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04102</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03623</span></span>
<span class="line"><span>Validation: Loss 0.03766 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03747 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03773</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03820</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03493</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03371</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03258</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03209</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03043</span></span>
<span class="line"><span>Validation: Loss 0.02875 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02859 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02815</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02804</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02867</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02756</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02515</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02503</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02078</span></span>
<span class="line"><span>Validation: Loss 0.02260 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02247 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02366</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02189</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02105</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01992</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02050</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01875</span></span>
<span class="line"><span>Validation: Loss 0.01841 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01829 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01907</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01782</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01822</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01763</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01664</span></span>
<span class="line"><span>Validation: Loss 0.01548 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01527</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01537</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01512</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01421</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01386</span></span>
<span class="line"><span>Validation: Loss 0.01337 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01328 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01325</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01370</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01117</span></span>
<span class="line"><span>Validation: Loss 0.01179 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01171 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01237</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01252</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01150</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01056</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01153</span></span>
<span class="line"><span>Validation: Loss 0.01056 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01049 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01108</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01120</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01054</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01074</span></span>
<span class="line"><span>Validation: Loss 0.00956 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00949 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00933</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00905</span></span>
<span class="line"><span>Validation: Loss 0.00872 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00866 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00904</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00840</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00899</span></span>
<span class="line"><span>Validation: Loss 0.00801 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00796 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00798</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00744</span></span>
<span class="line"><span>Validation: Loss 0.00740 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00735 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00722</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00659</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00688</span></span>
<span class="line"><span>Validation: Loss 0.00687 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00682 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00646</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00704</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00661</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00702</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00680</span></span>
<span class="line"><span>Validation: Loss 0.00640 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00636 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00646</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00651</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00607</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00657</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00563</span></span>
<span class="line"><span>Validation: Loss 0.00599 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00595 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00612</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00596</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00572</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00549</span></span>
<span class="line"><span>Validation: Loss 0.00562 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00558 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00619</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00574</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00601</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00526</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00547</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00531</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00548</span></span>
<span class="line"><span>Validation: Loss 0.00529 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00525 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00565</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00516</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00529</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00508</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00530</span></span>
<span class="line"><span>Validation: Loss 0.00499 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00495 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56102</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51407</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47371</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45950</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43227</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39568</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39243</span></span>
<span class="line"><span>Validation: Loss 0.37006 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36856 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36894</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35244</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34054</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31606</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30534</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28842</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26066</span></span>
<span class="line"><span>Validation: Loss 0.25950 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25863 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25513</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24458</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23339</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22326</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21205</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20366</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19624</span></span>
<span class="line"><span>Validation: Loss 0.18147 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18101 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18290</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17034</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16350</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15626</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14916</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14371</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13925</span></span>
<span class="line"><span>Validation: Loss 0.12962 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12933 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12841</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12531</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12029</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11337</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10659</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10255</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09764</span></span>
<span class="line"><span>Validation: Loss 0.09447 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09418 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09481</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08980</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08634</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08237</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08027</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07547</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07452</span></span>
<span class="line"><span>Validation: Loss 0.06974 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06938 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07068</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06666</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06455</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06085</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05896</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05575</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05483</span></span>
<span class="line"><span>Validation: Loss 0.05206 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05167 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05002</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04998</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04879</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04715</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04432</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04207</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04137</span></span>
<span class="line"><span>Validation: Loss 0.03918 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03880 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03648</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03960</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03527</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03338</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03400</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03492</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02973</span></span>
<span class="line"><span>Validation: Loss 0.02994 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02959 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03020</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03075</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02864</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02461</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02517</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02518</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02431</span></span>
<span class="line"><span>Validation: Loss 0.02353 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02323 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02392</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02119</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02198</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02177</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02067</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02095</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02072</span></span>
<span class="line"><span>Validation: Loss 0.01915 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01891 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02051</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01833</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01695</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01761</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01711</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01696</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01728</span></span>
<span class="line"><span>Validation: Loss 0.01609 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01586 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01655</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01590</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01610</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01469</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01402</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01489</span></span>
<span class="line"><span>Validation: Loss 0.01387 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01369 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01403</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01350</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01435</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01233</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01202</span></span>
<span class="line"><span>Validation: Loss 0.01222 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01205 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01192</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01115</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01140</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01198</span></span>
<span class="line"><span>Validation: Loss 0.01094 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01079 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01008</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01056</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01026</span></span>
<span class="line"><span>Validation: Loss 0.00989 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01046</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00804</span></span>
<span class="line"><span>Validation: Loss 0.00904 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00890 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00877</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00815</span></span>
<span class="line"><span>Validation: Loss 0.00830 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00819 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00800</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00792</span></span>
<span class="line"><span>Validation: Loss 0.00768 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00756 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00722</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00729</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00709</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00672</span></span>
<span class="line"><span>Validation: Loss 0.00712 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00702 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00683</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00683</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00694</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00668</span></span>
<span class="line"><span>Validation: Loss 0.00664 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00654 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00672</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00605</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00653</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00621</span></span>
<span class="line"><span>Validation: Loss 0.00621 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00612 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00629</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00613</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00565</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00585</span></span>
<span class="line"><span>Validation: Loss 0.00583 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00574 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00564</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00508</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00660</span></span>
<span class="line"><span>Validation: Loss 0.00548 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00540 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00513</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00546</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00527</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00484</span></span>
<span class="line"><span>Validation: Loss 0.00517 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00509 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
