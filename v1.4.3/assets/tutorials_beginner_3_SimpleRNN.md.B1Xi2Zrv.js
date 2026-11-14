import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.DtydgIfp.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>    403.0 ms  ✓ InvertedIndices</span></span>
<span class="line"><span>  46076.5 ms  ✓ DataFrames</span></span>
<span class="line"><span>  51817.3 ms  ✓ CUDA</span></span>
<span class="line"><span>   5243.0 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8696.7 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5314.1 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  6 dependencies successfully precompiled in 118 seconds. 94 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   4911.4 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   5334.8 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5389.5 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   4984.1 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5230.5 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5231.2 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   5770.6 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 6 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   5013.7 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5384.6 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5053.0 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 107 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   5827.0 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 174 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangDataFramesExt...</span></span>
<span class="line"><span>   1618.6 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling TransducersDataFramesExt...</span></span>
<span class="line"><span>   1423.4 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 61 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.4.3/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.4.3/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.4.3/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##230&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.4.3/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61466</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59222</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56538</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53192</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52945</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50069</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49103</span></span>
<span class="line"><span>Validation: Loss 0.46551 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47201 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46439</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45687</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44415</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42595</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40914</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39567</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39944</span></span>
<span class="line"><span>Validation: Loss 0.36794 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37547 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37290</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36134</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33842</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33959</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32097</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30847</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29734</span></span>
<span class="line"><span>Validation: Loss 0.28265 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29107 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.30058</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27622</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25626</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24315</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24900</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23140</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25694</span></span>
<span class="line"><span>Validation: Loss 0.21271 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22147 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21845</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18657</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19382</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19102</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19010</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18979</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19469</span></span>
<span class="line"><span>Validation: Loss 0.15762 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16596 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16529</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16055</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13431</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13475</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14147</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13922</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11118</span></span>
<span class="line"><span>Validation: Loss 0.11546 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12248 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12262</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11204</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10542</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10456</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09691</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09730</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08353</span></span>
<span class="line"><span>Validation: Loss 0.08286 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08817 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08157</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07897</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07694</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07519</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07263</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06703</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06024</span></span>
<span class="line"><span>Validation: Loss 0.05786 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06142 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06097</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05504</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05171</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05220</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05011</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04811</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04560</span></span>
<span class="line"><span>Validation: Loss 0.04275 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04519 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04774</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04404</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04161</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04061</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03748</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03339</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03148</span></span>
<span class="line"><span>Validation: Loss 0.03443 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03636 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03588</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03527</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03468</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03222</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03144</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03076</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02853</span></span>
<span class="line"><span>Validation: Loss 0.02917 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03085 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03344</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02880</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02919</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02717</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02616</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02728</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02289</span></span>
<span class="line"><span>Validation: Loss 0.02535 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02684 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02693</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02511</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02422</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02572</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02404</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02340</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02421</span></span>
<span class="line"><span>Validation: Loss 0.02241 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02376 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02341</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02184</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02189</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02371</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02139</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02143</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01792</span></span>
<span class="line"><span>Validation: Loss 0.02003 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02127 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01966</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02140</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02016</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02024</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01977</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01755</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02096</span></span>
<span class="line"><span>Validation: Loss 0.01807 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01922 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01791</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02073</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01857</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01715</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01956</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01424</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01647</span></span>
<span class="line"><span>Validation: Loss 0.01640 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01746 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01641</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01715</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01676</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01654</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01603</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01584</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01381</span></span>
<span class="line"><span>Validation: Loss 0.01499 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01598 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01619</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01526</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01433</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01475</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01442</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01540</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01324</span></span>
<span class="line"><span>Validation: Loss 0.01378 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01471 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01502</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01497</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01372</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01236</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01316</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01486</span></span>
<span class="line"><span>Validation: Loss 0.01273 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01359 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01351</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01272</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01258</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01280</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01265</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01232</span></span>
<span class="line"><span>Validation: Loss 0.01177 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01257 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01233</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01204</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01085</span></span>
<span class="line"><span>Validation: Loss 0.01083 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01157 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01096</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01048</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00949</span></span>
<span class="line"><span>Validation: Loss 0.00982 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01049 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01080</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00933</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00846</span></span>
<span class="line"><span>Validation: Loss 0.00872 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00930 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00947</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00817</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00814</span></span>
<span class="line"><span>Validation: Loss 0.00778 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00828 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00774</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00762</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00712</span></span>
<span class="line"><span>Validation: Loss 0.00712 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00755 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62815</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59273</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55839</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54737</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51929</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49971</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47626</span></span>
<span class="line"><span>Validation: Loss 0.46640 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46883 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46898</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46389</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44597</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43167</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40783</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37916</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39570</span></span>
<span class="line"><span>Validation: Loss 0.36861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37164 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37045</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35656</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35852</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32657</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32571</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30801</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28603</span></span>
<span class="line"><span>Validation: Loss 0.28345 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28672 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29379</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28620</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26634</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25464</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23619</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23390</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20474</span></span>
<span class="line"><span>Validation: Loss 0.21358 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21666 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22371</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21711</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19038</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18073</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19395</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17035</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17330</span></span>
<span class="line"><span>Validation: Loss 0.15838 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16096 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17413</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14452</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15736</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13227</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13055</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12925</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14008</span></span>
<span class="line"><span>Validation: Loss 0.11597 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11791 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.13106</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10266</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10650</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10380</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10333</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08646</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09514</span></span>
<span class="line"><span>Validation: Loss 0.08289 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08423 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08447</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08013</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07841</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07234</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06504</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06937</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05756</span></span>
<span class="line"><span>Validation: Loss 0.05774 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05862 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06087</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05691</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05082</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04976</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05170</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04546</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04831</span></span>
<span class="line"><span>Validation: Loss 0.04294 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04362 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04800</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04253</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04092</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03956</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03555</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03826</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03033</span></span>
<span class="line"><span>Validation: Loss 0.03476 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03534 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03568</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03385</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03366</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03423</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03319</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03007</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02974</span></span>
<span class="line"><span>Validation: Loss 0.02954 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03004 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02803</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03045</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02957</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02731</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02797</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02697</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03317</span></span>
<span class="line"><span>Validation: Loss 0.02572 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02618 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02459</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02706</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02679</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02481</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02351</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02417</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02273</span></span>
<span class="line"><span>Validation: Loss 0.02274 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02315 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02335</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02519</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02322</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02088</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02151</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02080</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01727</span></span>
<span class="line"><span>Validation: Loss 0.02035 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02072 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02131</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02079</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02139</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01874</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01938</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01908</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01762</span></span>
<span class="line"><span>Validation: Loss 0.01838 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01873 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01960</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01796</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01856</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01815</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01708</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01718</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01922</span></span>
<span class="line"><span>Validation: Loss 0.01672 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01704 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01786</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01778</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01614</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01669</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01603</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01552</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01337</span></span>
<span class="line"><span>Validation: Loss 0.01528 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01558 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01551</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01673</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01337</span></span>
<span class="line"><span>Validation: Loss 0.01406 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01433 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01476</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01362</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01250</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01496</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01426</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01501</span></span>
<span class="line"><span>Validation: Loss 0.01299 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01325 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01352</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01388</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01305</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01046</span></span>
<span class="line"><span>Validation: Loss 0.01204 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01228 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01301</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01144</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01290</span></span>
<span class="line"><span>Validation: Loss 0.01117 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01138 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01091</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01131</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01189</span></span>
<span class="line"><span>Validation: Loss 0.01029 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01048 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01017</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01037</span></span>
<span class="line"><span>Validation: Loss 0.00930 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00946 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00996</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00860</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00823</span></span>
<span class="line"><span>Validation: Loss 0.00825 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00839 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00840</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00780</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00723</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00710</span></span>
<span class="line"><span>Validation: Loss 0.00741 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00754 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MLDataDevices)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.2</span></span>
<span class="line"><span>Commit 5e9a32e7af2 (2024-12-01 20:02 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LLVM: libLLVM-16.0.6 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.6, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.6</span></span>
<span class="line"><span>NVIDIA driver 560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.6.4</span></span>
<span class="line"><span>- CURAND: 10.3.7</span></span>
<span class="line"><span>- CUFFT: 11.3.0</span></span>
<span class="line"><span>- CUSOLVER: 11.7.1</span></span>
<span class="line"><span>- CUSPARSE: 12.5.4</span></span>
<span class="line"><span>- CUPTI: 2024.3.2 (API 24.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.5.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.10.4+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.15.5+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.11.2</span></span>
<span class="line"><span>- LLVM: 16.0.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.109 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
