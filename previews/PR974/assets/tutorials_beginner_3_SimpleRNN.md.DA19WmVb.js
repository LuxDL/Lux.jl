import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.BTmvXJuB.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>   5535.5 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 121 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>  35836.0 ms  ✓ JLD2</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 36 seconds. 31 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>   1142.0 ms  ✓ BangBang</span></span>
<span class="line"><span>    778.4 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    783.6 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    992.5 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    762.9 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>   1989.5 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>   1273.5 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   3155.4 ms  ✓ Transducers</span></span>
<span class="line"><span>   1736.9 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>   5420.6 ms  ✓ FLoops</span></span>
<span class="line"><span>   7694.9 ms  ✓ MLUtils</span></span>
<span class="line"><span>  11 dependencies successfully precompiled in 23 seconds. 165 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   2781.1 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 181 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   3806.6 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 259 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR974/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR974/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR974/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR974/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61651</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58957</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57431</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53025</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51596</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51310</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48375</span></span>
<span class="line"><span>Validation: Loss 0.46934 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46484 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46741</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45210</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44446</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42855</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41470</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39528</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38336</span></span>
<span class="line"><span>Validation: Loss 0.37266 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36739 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37954</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35785</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33936</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32412</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32400</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31496</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31474</span></span>
<span class="line"><span>Validation: Loss 0.28776 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28198 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28088</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27802</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25379</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26361</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25084</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23814</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22965</span></span>
<span class="line"><span>Validation: Loss 0.21767 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21189 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21875</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20183</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20327</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18957</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18125</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18313</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16277</span></span>
<span class="line"><span>Validation: Loss 0.16159 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15634 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16591</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16060</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14858</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13375</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12845</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13110</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13163</span></span>
<span class="line"><span>Validation: Loss 0.11834 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11410 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11495</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11149</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10754</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10799</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09440</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09641</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08913</span></span>
<span class="line"><span>Validation: Loss 0.08467 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08155 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08949</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08381</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07608</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06785</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06452</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06631</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06561</span></span>
<span class="line"><span>Validation: Loss 0.05882 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05677 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05925</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05298</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05370</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05252</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04717</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04788</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04955</span></span>
<span class="line"><span>Validation: Loss 0.04355 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04216 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04342</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04334</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04098</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04060</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03699</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03558</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03845</span></span>
<span class="line"><span>Validation: Loss 0.03517 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03408 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03567</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03092</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03447</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03365</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03207</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03265</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02745</span></span>
<span class="line"><span>Validation: Loss 0.02982 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02888 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03102</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02917</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02690</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02796</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02745</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02919</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02099</span></span>
<span class="line"><span>Validation: Loss 0.02595 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02512 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02591</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02502</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02413</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02708</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02499</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02250</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02171</span></span>
<span class="line"><span>Validation: Loss 0.02297 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02222 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02374</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02197</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02221</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02148</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02114</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02250</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01938</span></span>
<span class="line"><span>Validation: Loss 0.02057 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01989 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02128</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02169</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01870</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02035</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01920</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01891</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01539</span></span>
<span class="line"><span>Validation: Loss 0.01858 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01795 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01974</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01680</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01791</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01796</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01774</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01765</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01774</span></span>
<span class="line"><span>Validation: Loss 0.01690 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01631 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01704</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01713</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01738</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01669</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01467</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01802</span></span>
<span class="line"><span>Validation: Loss 0.01545 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01491 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01515</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01570</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01514</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01509</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01380</span></span>
<span class="line"><span>Validation: Loss 0.01421 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01370 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01571</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01240</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01411</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01356</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01407</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01172</span></span>
<span class="line"><span>Validation: Loss 0.01312 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01265 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01171</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01400</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01270</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01269</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01180</span></span>
<span class="line"><span>Validation: Loss 0.01216 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01172 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01191</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01190</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01202</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01255</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01161</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00958</span></span>
<span class="line"><span>Validation: Loss 0.01123 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01082 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01067</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01114</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01088</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01244</span></span>
<span class="line"><span>Validation: Loss 0.01026 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00988 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01095</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01052</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00945</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00953</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00967</span></span>
<span class="line"><span>Validation: Loss 0.00915 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00882 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00877</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00882</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00825</span></span>
<span class="line"><span>Validation: Loss 0.00812 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00784 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00738</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00715</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00695</span></span>
<span class="line"><span>Validation: Loss 0.00737 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00713 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62985</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58713</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57225</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53042</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51957</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50281</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47441</span></span>
<span class="line"><span>Validation: Loss 0.47095 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46864 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47923</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45247</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44540</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43140</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39725</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38615</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39965</span></span>
<span class="line"><span>Validation: Loss 0.37441 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37177 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37090</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35572</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34333</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33831</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31869</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30832</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31510</span></span>
<span class="line"><span>Validation: Loss 0.29017 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28724 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29121</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27627</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25703</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25669</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23990</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24771</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20467</span></span>
<span class="line"><span>Validation: Loss 0.22049 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21746 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21133</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19901</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20756</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19488</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18132</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17825</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18557</span></span>
<span class="line"><span>Validation: Loss 0.16503 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16209 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16037</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14503</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15499</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13627</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13884</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13255</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14353</span></span>
<span class="line"><span>Validation: Loss 0.12165 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11915 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12227</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11004</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11105</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10083</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09786</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09429</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09347</span></span>
<span class="line"><span>Validation: Loss 0.08702 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08515 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08298</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08227</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07826</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07565</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07018</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06299</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05555</span></span>
<span class="line"><span>Validation: Loss 0.06037 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05916 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05665</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05945</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05212</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05365</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05052</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04739</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03490</span></span>
<span class="line"><span>Validation: Loss 0.04480 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04400 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04582</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04046</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04391</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03864</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03813</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03574</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04039</span></span>
<span class="line"><span>Validation: Loss 0.03629 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03566 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03597</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03270</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03391</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03451</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03209</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03220</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02888</span></span>
<span class="line"><span>Validation: Loss 0.03082 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03027 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03004</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03043</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02928</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03120</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02716</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02545</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02213</span></span>
<span class="line"><span>Validation: Loss 0.02683 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02634 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02475</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02495</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02498</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02698</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02520</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02412</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02287</span></span>
<span class="line"><span>Validation: Loss 0.02377 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02333 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02123</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02256</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02115</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02327</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02328</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02193</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02357</span></span>
<span class="line"><span>Validation: Loss 0.02129 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02089 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01878</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02154</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01896</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02092</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01860</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02087</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02199</span></span>
<span class="line"><span>Validation: Loss 0.01923 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01886 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01833</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01825</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01748</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02014</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01648</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01811</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01803</span></span>
<span class="line"><span>Validation: Loss 0.01747 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01713 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01698</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01691</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01631</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01706</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01668</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01583</span></span>
<span class="line"><span>Validation: Loss 0.01598 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01566 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01598</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01531</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01499</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01481</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01375</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01561</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01629</span></span>
<span class="line"><span>Validation: Loss 0.01470 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01440 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01509</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01505</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01267</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01239</span></span>
<span class="line"><span>Validation: Loss 0.01359 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01331 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01409</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01275</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01197</span></span>
<span class="line"><span>Validation: Loss 0.01261 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01235 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01198</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01090</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01254</span></span>
<span class="line"><span>Validation: Loss 0.01172 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01147 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01091</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01057</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01179</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01022</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01204</span></span>
<span class="line"><span>Validation: Loss 0.01085 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01062 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01144</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00983</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00992</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00714</span></span>
<span class="line"><span>Validation: Loss 0.00989 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00968 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00988</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00873</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00911</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00920</span></span>
<span class="line"><span>Validation: Loss 0.00885 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00865 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00916</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00842</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00755</span></span>
<span class="line"><span>Validation: Loss 0.00788 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00772 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.0</span></span>
<span class="line"><span>Commit 501a4f25c2b (2024-10-07 11:40 UTC)</span></span>
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
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.6.3</span></span>
<span class="line"><span>- CURAND: 10.3.7</span></span>
<span class="line"><span>- CUFFT: 11.3.0</span></span>
<span class="line"><span>- CUSOLVER: 11.7.1</span></span>
<span class="line"><span>- CUSPARSE: 12.5.4</span></span>
<span class="line"><span>- CUPTI: 2024.3.2 (API 24.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.5.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.10.3+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.15.3+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.11.0</span></span>
<span class="line"><span>- LLVM: 16.0.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Preferences:</span></span>
<span class="line"><span>- CUDA_Driver_jll.compat: false</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.141 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
