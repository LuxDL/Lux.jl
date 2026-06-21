import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.MfBj6Zyp.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>   2766.1 ms  ✓ GPUArrays</span></span>
<span class="line"><span>   1633.5 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>   1753.1 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>  59930.9 ms  ✓ CUDA</span></span>
<span class="line"><span>   5735.1 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5786.6 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   6076.3 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5698.4 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>   5738.0 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>   5876.0 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>   6086.1 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>   9077.5 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5694.5 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>   5901.6 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>   7804.4 ms  ✓ MLUtils</span></span>
<span class="line"><span>   2793.8 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>   6155.1 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>   6392.0 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  10777.6 ms  ✓ Lux</span></span>
<span class="line"><span>   3522.7 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>   4244.8 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  21 dependencies successfully precompiled in 122 seconds. 231 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>   5590.8 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 121 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>  34948.1 ms  ✓ JLD2</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 35 seconds. 31 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.4.0/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.4.0/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.4.0/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.4.0/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61355</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60122</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55937</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53879</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51808</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51605</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47831</span></span>
<span class="line"><span>Validation: Loss 0.46661 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45916 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46628</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44990</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44128</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43255</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40714</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40766</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39937</span></span>
<span class="line"><span>Validation: Loss 0.36971 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36096 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37687</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35752</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34448</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34105</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31895</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31300</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30025</span></span>
<span class="line"><span>Validation: Loss 0.28482 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27534 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29736</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27661</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27365</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25206</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24862</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23127</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20953</span></span>
<span class="line"><span>Validation: Loss 0.21468 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20514 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22380</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20012</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19739</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20192</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19808</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16450</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16835</span></span>
<span class="line"><span>Validation: Loss 0.15907 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15023 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16398</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15826</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16086</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14523</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12280</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12723</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12907</span></span>
<span class="line"><span>Validation: Loss 0.11640 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10915 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12678</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11296</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10677</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09760</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10082</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09412</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09540</span></span>
<span class="line"><span>Validation: Loss 0.08320 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07790 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08385</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08116</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07856</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07251</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06865</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06813</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05979</span></span>
<span class="line"><span>Validation: Loss 0.05788 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05440 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05526</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05986</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05482</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05024</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05068</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04910</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03947</span></span>
<span class="line"><span>Validation: Loss 0.04309 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04062 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04734</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04295</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04133</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03836</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03951</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03558</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03752</span></span>
<span class="line"><span>Validation: Loss 0.03489 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03285 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03542</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03343</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03485</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03239</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03243</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03304</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03280</span></span>
<span class="line"><span>Validation: Loss 0.02961 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02784 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03061</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02998</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02837</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02854</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02796</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02674</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03127</span></span>
<span class="line"><span>Validation: Loss 0.02574 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02417 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02820</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02770</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02764</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02521</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02183</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02022</span></span>
<span class="line"><span>Validation: Loss 0.02272 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02131 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02412</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02214</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02296</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02144</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02131</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02155</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02549</span></span>
<span class="line"><span>Validation: Loss 0.02034 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01905 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02016</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02038</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02045</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02121</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01968</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01939</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01760</span></span>
<span class="line"><span>Validation: Loss 0.01834 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01716 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01903</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01876</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01733</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01890</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01801</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01808</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01462</span></span>
<span class="line"><span>Validation: Loss 0.01666 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01556 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01729</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01612</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01670</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01664</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01657</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01651</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01512</span></span>
<span class="line"><span>Validation: Loss 0.01522 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01420 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01561</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01616</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01452</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01505</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01611</span></span>
<span class="line"><span>Validation: Loss 0.01397 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01302 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01425</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01391</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01348</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01437</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01224</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01583</span></span>
<span class="line"><span>Validation: Loss 0.01285 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01196 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01350</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01313</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01210</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01280</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01120</span></span>
<span class="line"><span>Validation: Loss 0.01178 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01220</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01308</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01087</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01095</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01146</span></span>
<span class="line"><span>Validation: Loss 0.01066 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00994 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01010</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01059</span></span>
<span class="line"><span>Validation: Loss 0.00944 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00883 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00896</span></span>
<span class="line"><span>Validation: Loss 0.00838 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00786 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00738</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00828</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00759</span></span>
<span class="line"><span>Validation: Loss 0.00763 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00716 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00798</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00627</span></span>
<span class="line"><span>Validation: Loss 0.00709 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00666 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.59817</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.61437</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56622</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54189</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51737</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49618</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49495</span></span>
<span class="line"><span>Validation: Loss 0.47899 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47553 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46698</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46119</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45414</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43124</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40566</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38524</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37662</span></span>
<span class="line"><span>Validation: Loss 0.38435 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38085 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37390</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36331</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34397</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32278</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33548</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30345</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33287</span></span>
<span class="line"><span>Validation: Loss 0.30156 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29772 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27521</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28071</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28083</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26859</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24388</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23667</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20741</span></span>
<span class="line"><span>Validation: Loss 0.23261 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22866 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21795</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21692</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20361</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18745</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18196</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18741</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17958</span></span>
<span class="line"><span>Validation: Loss 0.17655 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17258 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16314</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16417</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14852</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15359</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12981</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13258</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13318</span></span>
<span class="line"><span>Validation: Loss 0.13129 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12777 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11685</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12432</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10753</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10374</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10754</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09146</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10040</span></span>
<span class="line"><span>Validation: Loss 0.09435 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09168 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08305</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08316</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07832</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07814</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07008</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06555</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07635</span></span>
<span class="line"><span>Validation: Loss 0.06526 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06347 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06184</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05639</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05512</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05228</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05140</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04724</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04700</span></span>
<span class="line"><span>Validation: Loss 0.04806 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04679 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04574</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04227</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04450</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03964</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03944</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03695</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03498</span></span>
<span class="line"><span>Validation: Loss 0.03876 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03771 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03502</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03347</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03354</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03395</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03252</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03411</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03504</span></span>
<span class="line"><span>Validation: Loss 0.03292 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03200 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03184</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03221</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02706</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02729</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02980</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02648</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02640</span></span>
<span class="line"><span>Validation: Loss 0.02861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02780 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02523</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02739</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02618</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02526</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02435</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02349</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02597</span></span>
<span class="line"><span>Validation: Loss 0.02532 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02459 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02337</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02298</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02154</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02370</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02168</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02094</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02563</span></span>
<span class="line"><span>Validation: Loss 0.02268 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02201 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02049</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01980</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02240</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01997</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01759</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02125</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01877</span></span>
<span class="line"><span>Validation: Loss 0.02048 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01987 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01847</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01965</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01812</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01737</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01818</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01827</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01633</span></span>
<span class="line"><span>Validation: Loss 0.01862 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01806 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01711</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01622</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01781</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01513</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01721</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01667</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01487</span></span>
<span class="line"><span>Validation: Loss 0.01702 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01650 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01609</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01480</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01630</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01499</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01420</span></span>
<span class="line"><span>Validation: Loss 0.01560 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01512 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01473</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01235</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01359</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01329</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01463</span></span>
<span class="line"><span>Validation: Loss 0.01427 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01384 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01423</span></span>
<span class="line"><span>Validation: Loss 0.01291 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01253 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01116</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01088</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01219</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00963</span></span>
<span class="line"><span>Validation: Loss 0.01141 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01109 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01060</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00958</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00924</span></span>
<span class="line"><span>Validation: Loss 0.01005 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00930</span></span>
<span class="line"><span>Validation: Loss 0.00907 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00881 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00911</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00822</span></span>
<span class="line"><span>Validation: Loss 0.00836 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00812 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00707</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00728</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00740</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00823</span></span>
<span class="line"><span>Validation: Loss 0.00781 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00759 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.1</span></span>
<span class="line"><span>Commit 8f5b7ca12ad (2024-10-16 10:53 UTC)</span></span>
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
<span class="line"><span>- Julia: 1.11.1</span></span>
<span class="line"><span>- LLVM: 16.0.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.141 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
