import{_ as a,c as n,o as i,al as e}from"./chunks/framework.BCN3FD2k.js";const d=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"};function t(l,s,c,r,h,k){return i(),n("div",null,s[0]||(s[0]=[e(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><p>Note: If you wish to use AutoZygote() for automatic differentiation, add Zygote to your project dependencies and include <code>using Zygote</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>   4096.2 ms  ✓ FileIO</span></span>
<span class="line"><span>  31399.5 ms  ✓ JLD2</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 36 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>    368.3 ms  ✓ EnumX</span></span>
<span class="line"><span>    610.7 ms  ✓ URIs</span></span>
<span class="line"><span>    490.3 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>   1020.1 ms  ✓ MbedTLS</span></span>
<span class="line"><span>    598.8 ms  ✓ LLVMOpenMP_jll</span></span>
<span class="line"><span>   1162.7 ms  ✓ CUDA_Driver_jll</span></span>
<span class="line"><span>   1945.8 ms  ✓ OpenSSL</span></span>
<span class="line"><span>   2349.1 ms  ✓ Reactant_jll</span></span>
<span class="line"><span>  18352.8 ms  ✓ HTTP</span></span>
<span class="line"><span>  77930.9 ms  ✓ Reactant</span></span>
<span class="line"><span>  10 dependencies successfully precompiled in 100 seconds. 69 already precompiled.</span></span>
<span class="line"><span>Precompiling HTTPExt...</span></span>
<span class="line"><span>   1959.3 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 43 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  13566.9 ms  ✓ LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 84 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  12764.1 ms  ✓ MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 81 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  12861.0 ms  ✓ Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>  13347.4 ms  ✓ Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>  13362.1 ms  ✓ WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 14 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantKernelAbstractionsExt...</span></span>
<span class="line"><span>  13499.4 ms  ✓ Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 91 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantArrayInterfaceExt...</span></span>
<span class="line"><span>  13309.9 ms  ✓ Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  13606.7 ms  ✓ Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  11384.2 ms  ✓ Lux → LuxReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 12 seconds. 178 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L,C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##230&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T,3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T,3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), model))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_loader)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ad </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                ad, lossfn, (x, y), train_state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch (total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples) (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()((train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-6/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>2025-03-31 06:24:07.533248: I external/xla/xla/service/service.cc:152] XLA service 0x8059980 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-31 06:24:07.533567: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1743402247.535771 3227699 se_gpu_pjrt_client.cc:1040] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1743402247.536208 3227699 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743402247.536542 3227699 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743402247.552814 3227699 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>E0000 00:00:1743402299.324434 3227699 buffer_comparator.cc:156] Difference at 32: 0, expected 1.62244</span></span>
<span class="line"><span>E0000 00:00:1743402299.325730 3227699 buffer_comparator.cc:156] Difference at 33: 0, expected 1.87084</span></span>
<span class="line"><span>E0000 00:00:1743402299.325738 3227699 buffer_comparator.cc:156] Difference at 34: 0, expected 1.07351</span></span>
<span class="line"><span>E0000 00:00:1743402299.325745 3227699 buffer_comparator.cc:156] Difference at 35: 0, expected 2.92445</span></span>
<span class="line"><span>E0000 00:00:1743402299.325751 3227699 buffer_comparator.cc:156] Difference at 36: 0, expected 1.98056</span></span>
<span class="line"><span>E0000 00:00:1743402299.325757 3227699 buffer_comparator.cc:156] Difference at 37: 0, expected 2.07715</span></span>
<span class="line"><span>E0000 00:00:1743402299.325764 3227699 buffer_comparator.cc:156] Difference at 38: 0, expected 1.56458</span></span>
<span class="line"><span>E0000 00:00:1743402299.325770 3227699 buffer_comparator.cc:156] Difference at 39: 0, expected 2.27034</span></span>
<span class="line"><span>E0000 00:00:1743402299.325776 3227699 buffer_comparator.cc:156] Difference at 40: 0, expected 2.31795</span></span>
<span class="line"><span>E0000 00:00:1743402299.325782 3227699 buffer_comparator.cc:156] Difference at 41: 0, expected 2.55731</span></span>
<span class="line"><span>2025-03-31 06:24:59.325796: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.331021 3227699 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743402299.331045 3227699 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743402299.331049 3227699 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743402299.331052 3227699 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743402299.331054 3227699 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743402299.331057 3227699 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743402299.331060 3227699 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743402299.331063 3227699 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743402299.331065 3227699 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743402299.331068 3227699 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-03-31 06:24:59.331075: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.333645 3227699 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743402299.333658 3227699 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743402299.333661 3227699 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743402299.333664 3227699 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743402299.333667 3227699 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743402299.333669 3227699 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743402299.333672 3227699 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743402299.333675 3227699 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743402299.333679 3227699 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743402299.333682 3227699 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-03-31 06:24:59.333686: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.336229 3227699 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743402299.336244 3227699 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743402299.336247 3227699 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743402299.336250 3227699 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743402299.336252 3227699 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743402299.336255 3227699 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743402299.336258 3227699 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743402299.336261 3227699 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743402299.336263 3227699 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743402299.336266 3227699 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-03-31 06:24:59.336271: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.338799 3227699 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743402299.338813 3227699 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743402299.338816 3227699 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743402299.338819 3227699 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743402299.338822 3227699 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743402299.338825 3227699 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743402299.338828 3227699 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743402299.338830 3227699 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743402299.338833 3227699 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743402299.338836 3227699 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-03-31 06:24:59.338840: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.341396 3227699 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743402299.341412 3227699 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743402299.341419 3227699 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743402299.341422 3227699 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743402299.341424 3227699 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743402299.341427 3227699 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743402299.341430 3227699 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743402299.341433 3227699 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743402299.341435 3227699 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743402299.341438 3227699 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-03-31 06:24:59.341443: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.344000 3227699 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743402299.344018 3227699 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743402299.344021 3227699 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743402299.344023 3227699 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743402299.344026 3227699 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743402299.344029 3227699 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743402299.344032 3227699 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743402299.344035 3227699 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743402299.344037 3227699 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743402299.344040 3227699 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-03-31 06:24:59.344045: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.346611 3227699 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743402299.346657 3227699 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743402299.346660 3227699 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743402299.346663 3227699 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743402299.346666 3227699 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743402299.346669 3227699 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743402299.346671 3227699 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743402299.346674 3227699 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743402299.346677 3227699 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743402299.346680 3227699 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-03-31 06:24:59.346686: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.349450 3227699 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743402299.349472 3227699 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743402299.349475 3227699 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743402299.349478 3227699 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743402299.349481 3227699 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743402299.349484 3227699 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743402299.349486 3227699 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743402299.349489 3227699 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743402299.349492 3227699 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743402299.349495 3227699 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-03-31 06:24:59.349500: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.352121 3227699 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743402299.352142 3227699 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743402299.352145 3227699 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743402299.352150 3227699 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743402299.352153 3227699 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743402299.352155 3227699 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743402299.352158 3227699 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743402299.352161 3227699 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743402299.352164 3227699 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743402299.352166 3227699 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-03-31 06:24:59.352172: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.354767 3227699 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743402299.354786 3227699 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743402299.354790 3227699 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743402299.354793 3227699 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743402299.354796 3227699 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743402299.354799 3227699 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743402299.354801 3227699 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743402299.354804 3227699 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743402299.354807 3227699 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743402299.354810 3227699 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-03-31 06:24:59.354815: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.357413 3227699 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743402299.357444 3227699 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743402299.357448 3227699 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743402299.357451 3227699 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743402299.357453 3227699 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743402299.357456 3227699 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743402299.357459 3227699 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743402299.357462 3227699 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743402299.357464 3227699 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743402299.357467 3227699 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-03-31 06:24:59.357473: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.360172 3227699 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743402299.360199 3227699 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743402299.360202 3227699 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743402299.360205 3227699 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743402299.360208 3227699 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743402299.360211 3227699 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743402299.360214 3227699 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743402299.360219 3227699 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743402299.360222 3227699 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743402299.360224 3227699 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-03-31 06:24:59.360230: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.362849 3227699 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743402299.362867 3227699 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743402299.362871 3227699 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743402299.362873 3227699 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743402299.362876 3227699 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743402299.362879 3227699 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743402299.362882 3227699 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743402299.362885 3227699 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743402299.362887 3227699 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743402299.362890 3227699 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-03-31 06:24:59.362895: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.365473 3227699 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743402299.365492 3227699 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743402299.365495 3227699 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743402299.365498 3227699 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743402299.365501 3227699 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743402299.365504 3227699 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743402299.365506 3227699 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743402299.365509 3227699 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743402299.365512 3227699 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743402299.365515 3227699 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-03-31 06:24:59.365519: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.368089 3227699 buffer_comparator.cc:156] Difference at 256: 0, expected 0.86224</span></span>
<span class="line"><span>E0000 00:00:1743402299.368106 3227699 buffer_comparator.cc:156] Difference at 257: 0, expected 0.686873</span></span>
<span class="line"><span>E0000 00:00:1743402299.368109 3227699 buffer_comparator.cc:156] Difference at 258: 0, expected 0.252371</span></span>
<span class="line"><span>E0000 00:00:1743402299.368112 3227699 buffer_comparator.cc:156] Difference at 259: 0, expected 0.335927</span></span>
<span class="line"><span>E0000 00:00:1743402299.368115 3227699 buffer_comparator.cc:156] Difference at 260: 0, expected 0.934139</span></span>
<span class="line"><span>E0000 00:00:1743402299.368118 3227699 buffer_comparator.cc:156] Difference at 261: 0, expected 0.274756</span></span>
<span class="line"><span>E0000 00:00:1743402299.368121 3227699 buffer_comparator.cc:156] Difference at 262: 0, expected 0.529946</span></span>
<span class="line"><span>E0000 00:00:1743402299.368123 3227699 buffer_comparator.cc:156] Difference at 263: 0, expected 0.542969</span></span>
<span class="line"><span>E0000 00:00:1743402299.368126 3227699 buffer_comparator.cc:156] Difference at 264: 0, expected 0.895372</span></span>
<span class="line"><span>E0000 00:00:1743402299.368129 3227699 buffer_comparator.cc:156] Difference at 265: 0, expected 0.895664</span></span>
<span class="line"><span>2025-03-31 06:24:59.368133: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402299.370699 3227699 buffer_comparator.cc:156] Difference at 256: 0, expected 0.86224</span></span>
<span class="line"><span>E0000 00:00:1743402299.370715 3227699 buffer_comparator.cc:156] Difference at 257: 0, expected 0.686873</span></span>
<span class="line"><span>E0000 00:00:1743402299.370718 3227699 buffer_comparator.cc:156] Difference at 258: 0, expected 0.252371</span></span>
<span class="line"><span>E0000 00:00:1743402299.370721 3227699 buffer_comparator.cc:156] Difference at 259: 0, expected 0.335927</span></span>
<span class="line"><span>E0000 00:00:1743402299.370724 3227699 buffer_comparator.cc:156] Difference at 260: 0, expected 0.934139</span></span>
<span class="line"><span>E0000 00:00:1743402299.370727 3227699 buffer_comparator.cc:156] Difference at 261: 0, expected 0.274756</span></span>
<span class="line"><span>E0000 00:00:1743402299.370730 3227699 buffer_comparator.cc:156] Difference at 262: 0, expected 0.529946</span></span>
<span class="line"><span>E0000 00:00:1743402299.370732 3227699 buffer_comparator.cc:156] Difference at 263: 0, expected 0.542969</span></span>
<span class="line"><span>E0000 00:00:1743402299.370735 3227699 buffer_comparator.cc:156] Difference at 264: 0, expected 0.895372</span></span>
<span class="line"><span>E0000 00:00:1743402299.370738 3227699 buffer_comparator.cc:156] Difference at 265: 0, expected 0.895664</span></span>
<span class="line"><span>2025-03-31 06:24:59.370743: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.493640 3227699 buffer_comparator.cc:156] Difference at 16: 3.79485, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1743402339.493702 3227699 buffer_comparator.cc:156] Difference at 17: 3.76853, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1743402339.493711 3227699 buffer_comparator.cc:156] Difference at 18: 4.20178, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1743402339.493718 3227699 buffer_comparator.cc:156] Difference at 19: 3.12103, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1743402339.493725 3227699 buffer_comparator.cc:156] Difference at 20: 4.65882, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1743402339.493731 3227699 buffer_comparator.cc:156] Difference at 21: 1.89114, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1743402339.493737 3227699 buffer_comparator.cc:156] Difference at 22: 4.84752, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1743402339.493743 3227699 buffer_comparator.cc:156] Difference at 23: 0.96914, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1743402339.493750 3227699 buffer_comparator.cc:156] Difference at 24: 4.96662, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1743402339.493756 3227699 buffer_comparator.cc:156] Difference at 25: -0.185816, expected 11.3838</span></span>
<span class="line"><span>2025-03-31 06:25:39.493773: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.496230 3227699 buffer_comparator.cc:156] Difference at 16: 3.79485, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1743402339.496242 3227699 buffer_comparator.cc:156] Difference at 17: 3.76853, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1743402339.496245 3227699 buffer_comparator.cc:156] Difference at 18: 4.20178, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1743402339.496248 3227699 buffer_comparator.cc:156] Difference at 19: 3.12103, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1743402339.496251 3227699 buffer_comparator.cc:156] Difference at 20: 4.65882, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1743402339.496254 3227699 buffer_comparator.cc:156] Difference at 21: 1.89114, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1743402339.496256 3227699 buffer_comparator.cc:156] Difference at 22: 4.84752, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1743402339.496259 3227699 buffer_comparator.cc:156] Difference at 23: 0.96914, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1743402339.496262 3227699 buffer_comparator.cc:156] Difference at 24: 4.96662, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1743402339.496265 3227699 buffer_comparator.cc:156] Difference at 25: -0.185816, expected 11.3838</span></span>
<span class="line"><span>2025-03-31 06:25:39.496270: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.498402 3227699 buffer_comparator.cc:156] Difference at 32: 3.25568, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1743402339.498413 3227699 buffer_comparator.cc:156] Difference at 33: -3.24346, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1743402339.498418 3227699 buffer_comparator.cc:156] Difference at 34: 2.69235, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1743402339.498422 3227699 buffer_comparator.cc:156] Difference at 35: -3.80436, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1743402339.498424 3227699 buffer_comparator.cc:156] Difference at 36: 1.7516, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1743402339.498427 3227699 buffer_comparator.cc:156] Difference at 37: -4.36468, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1743402339.498430 3227699 buffer_comparator.cc:156] Difference at 38: 0.805955, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1743402339.498433 3227699 buffer_comparator.cc:156] Difference at 39: -4.31532, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1743402339.498436 3227699 buffer_comparator.cc:156] Difference at 40: 0.152318, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1743402339.498439 3227699 buffer_comparator.cc:156] Difference at 41: -4.39102, expected 8.63119</span></span>
<span class="line"><span>2025-03-31 06:25:39.498444: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.500582 3227699 buffer_comparator.cc:156] Difference at 32: 3.25568, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1743402339.500593 3227699 buffer_comparator.cc:156] Difference at 33: -3.24346, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1743402339.500596 3227699 buffer_comparator.cc:156] Difference at 34: 2.69235, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1743402339.500599 3227699 buffer_comparator.cc:156] Difference at 35: -3.80436, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1743402339.500602 3227699 buffer_comparator.cc:156] Difference at 36: 1.7516, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1743402339.500605 3227699 buffer_comparator.cc:156] Difference at 37: -4.36468, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1743402339.500608 3227699 buffer_comparator.cc:156] Difference at 38: 0.805955, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1743402339.500611 3227699 buffer_comparator.cc:156] Difference at 39: -4.31532, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1743402339.500614 3227699 buffer_comparator.cc:156] Difference at 40: 0.152318, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1743402339.500616 3227699 buffer_comparator.cc:156] Difference at 41: -4.39102, expected 8.63119</span></span>
<span class="line"><span>2025-03-31 06:25:39.500621: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.502766 3227699 buffer_comparator.cc:156] Difference at 64: -2.58896, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743402339.502777 3227699 buffer_comparator.cc:156] Difference at 65: 2.75211, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743402339.502781 3227699 buffer_comparator.cc:156] Difference at 66: -2.07092, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743402339.502784 3227699 buffer_comparator.cc:156] Difference at 67: 3.21954, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743402339.502787 3227699 buffer_comparator.cc:156] Difference at 68: -1.27868, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743402339.502790 3227699 buffer_comparator.cc:156] Difference at 69: 3.28017, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743402339.502792 3227699 buffer_comparator.cc:156] Difference at 70: -0.605668, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743402339.502795 3227699 buffer_comparator.cc:156] Difference at 71: 3.49913, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743402339.502798 3227699 buffer_comparator.cc:156] Difference at 72: 0.0175218, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743402339.502801 3227699 buffer_comparator.cc:156] Difference at 73: 3.61537, expected 8.82565</span></span>
<span class="line"><span>2025-03-31 06:25:39.502806: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.504936 3227699 buffer_comparator.cc:156] Difference at 64: -2.58896, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743402339.504947 3227699 buffer_comparator.cc:156] Difference at 65: 2.75211, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743402339.504950 3227699 buffer_comparator.cc:156] Difference at 66: -2.07092, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743402339.504953 3227699 buffer_comparator.cc:156] Difference at 67: 3.21954, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743402339.504958 3227699 buffer_comparator.cc:156] Difference at 68: -1.27868, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743402339.504961 3227699 buffer_comparator.cc:156] Difference at 69: 3.28017, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743402339.504964 3227699 buffer_comparator.cc:156] Difference at 70: -0.605668, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743402339.504967 3227699 buffer_comparator.cc:156] Difference at 71: 3.49913, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743402339.504969 3227699 buffer_comparator.cc:156] Difference at 72: 0.0175218, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743402339.504972 3227699 buffer_comparator.cc:156] Difference at 73: 3.61537, expected 8.82565</span></span>
<span class="line"><span>2025-03-31 06:25:39.504977: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.507128 3227699 buffer_comparator.cc:156] Difference at 64: -2.58896, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743402339.507141 3227699 buffer_comparator.cc:156] Difference at 65: 2.75211, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743402339.507145 3227699 buffer_comparator.cc:156] Difference at 66: -2.07092, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743402339.507148 3227699 buffer_comparator.cc:156] Difference at 67: 3.21954, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743402339.507150 3227699 buffer_comparator.cc:156] Difference at 68: -1.27868, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743402339.507153 3227699 buffer_comparator.cc:156] Difference at 69: 3.28017, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743402339.507156 3227699 buffer_comparator.cc:156] Difference at 70: -0.605668, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743402339.507159 3227699 buffer_comparator.cc:156] Difference at 71: 3.49913, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743402339.507162 3227699 buffer_comparator.cc:156] Difference at 72: 0.0175218, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743402339.507165 3227699 buffer_comparator.cc:156] Difference at 73: 3.61537, expected 8.82565</span></span>
<span class="line"><span>2025-03-31 06:25:39.507170: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.515720 3227699 buffer_comparator.cc:156] Difference at 16: 9.1023, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1743402339.515733 3227699 buffer_comparator.cc:156] Difference at 17: 8.26259, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1743402339.515737 3227699 buffer_comparator.cc:156] Difference at 18: 10.5875, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1743402339.515740 3227699 buffer_comparator.cc:156] Difference at 19: 9.23707, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1743402339.515743 3227699 buffer_comparator.cc:156] Difference at 20: 8.65902, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1743402339.515746 3227699 buffer_comparator.cc:156] Difference at 21: 10.9934, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1743402339.515749 3227699 buffer_comparator.cc:156] Difference at 22: 10.0339, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1743402339.515752 3227699 buffer_comparator.cc:156] Difference at 23: 8.29374, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1743402339.515755 3227699 buffer_comparator.cc:156] Difference at 24: 7.65491, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1743402339.515758 3227699 buffer_comparator.cc:156] Difference at 25: 8.47378, expected 36.4575</span></span>
<span class="line"><span>2025-03-31 06:25:39.515763: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.517900 3227699 buffer_comparator.cc:156] Difference at 16: 9.1023, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1743402339.517912 3227699 buffer_comparator.cc:156] Difference at 17: 8.26259, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1743402339.517915 3227699 buffer_comparator.cc:156] Difference at 18: 10.5875, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1743402339.517918 3227699 buffer_comparator.cc:156] Difference at 19: 9.23707, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1743402339.517921 3227699 buffer_comparator.cc:156] Difference at 20: 8.65902, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1743402339.517924 3227699 buffer_comparator.cc:156] Difference at 21: 10.9934, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1743402339.517929 3227699 buffer_comparator.cc:156] Difference at 22: 10.0339, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1743402339.517932 3227699 buffer_comparator.cc:156] Difference at 23: 8.29374, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1743402339.517935 3227699 buffer_comparator.cc:156] Difference at 24: 7.65491, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1743402339.517938 3227699 buffer_comparator.cc:156] Difference at 25: 8.47378, expected 36.4575</span></span>
<span class="line"><span>2025-03-31 06:25:39.517942: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.530651 3227699 buffer_comparator.cc:156] Difference at 2: 37.361, expected 33.2434</span></span>
<span class="line"><span>E0000 00:00:1743402339.530666 3227699 buffer_comparator.cc:156] Difference at 8: 32.9182, expected 29.0801</span></span>
<span class="line"><span>E0000 00:00:1743402339.530670 3227699 buffer_comparator.cc:156] Difference at 11: 35.3152, expected 30.7625</span></span>
<span class="line"><span>E0000 00:00:1743402339.530673 3227699 buffer_comparator.cc:156] Difference at 12: 39.5233, expected 34.3637</span></span>
<span class="line"><span>E0000 00:00:1743402339.530676 3227699 buffer_comparator.cc:156] Difference at 20: 38.835, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1743402339.530679 3227699 buffer_comparator.cc:156] Difference at 23: 37.0226, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1743402339.530682 3227699 buffer_comparator.cc:156] Difference at 26: 39.1599, expected 32.4927</span></span>
<span class="line"><span>E0000 00:00:1743402339.530686 3227699 buffer_comparator.cc:156] Difference at 51: 26.8307, expected 33.7879</span></span>
<span class="line"><span>2025-03-31 06:25:39.530692: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.539916 3227699 buffer_comparator.cc:156] Difference at 16: 6.73254, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1743402339.539929 3227699 buffer_comparator.cc:156] Difference at 17: 9.2143, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743402339.539932 3227699 buffer_comparator.cc:156] Difference at 18: 6.85711, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1743402339.539935 3227699 buffer_comparator.cc:156] Difference at 19: 7.1738, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1743402339.539938 3227699 buffer_comparator.cc:156] Difference at 20: 6.9796, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743402339.539941 3227699 buffer_comparator.cc:156] Difference at 21: 7.31867, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1743402339.539944 3227699 buffer_comparator.cc:156] Difference at 22: 8.70739, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1743402339.539947 3227699 buffer_comparator.cc:156] Difference at 23: 7.19731, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1743402339.539950 3227699 buffer_comparator.cc:156] Difference at 24: 8.21217, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1743402339.539953 3227699 buffer_comparator.cc:156] Difference at 25: 9.32022, expected 36.0917</span></span>
<span class="line"><span>2025-03-31 06:25:39.539958: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.542101 3227699 buffer_comparator.cc:156] Difference at 16: 6.73254, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1743402339.542113 3227699 buffer_comparator.cc:156] Difference at 17: 9.2143, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743402339.542116 3227699 buffer_comparator.cc:156] Difference at 18: 6.85711, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1743402339.542119 3227699 buffer_comparator.cc:156] Difference at 19: 7.1738, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1743402339.542122 3227699 buffer_comparator.cc:156] Difference at 20: 6.9796, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743402339.542125 3227699 buffer_comparator.cc:156] Difference at 21: 7.31867, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1743402339.542128 3227699 buffer_comparator.cc:156] Difference at 22: 8.70739, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1743402339.542131 3227699 buffer_comparator.cc:156] Difference at 23: 7.19731, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1743402339.542134 3227699 buffer_comparator.cc:156] Difference at 24: 8.21217, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1743402339.542137 3227699 buffer_comparator.cc:156] Difference at 25: 9.32022, expected 36.0917</span></span>
<span class="line"><span>2025-03-31 06:25:39.542143: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743402339.554834 3227699 buffer_comparator.cc:156] Difference at 2: 38.3457, expected 34.2806</span></span>
<span class="line"><span>E0000 00:00:1743402339.554849 3227699 buffer_comparator.cc:156] Difference at 6: 41.1731, expected 36.7103</span></span>
<span class="line"><span>E0000 00:00:1743402339.554852 3227699 buffer_comparator.cc:156] Difference at 13: 31.5951, expected 35.7459</span></span>
<span class="line"><span>E0000 00:00:1743402339.554856 3227699 buffer_comparator.cc:156] Difference at 17: 37.0853, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743402339.554859 3227699 buffer_comparator.cc:156] Difference at 20: 37.9014, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743402339.554862 3227699 buffer_comparator.cc:156] Difference at 45: 25.9105, expected 32.5352</span></span>
<span class="line"><span>E0000 00:00:1743402339.554865 3227699 buffer_comparator.cc:156] Difference at 75: 24.7101, expected 28.3085</span></span>
<span class="line"><span>E0000 00:00:1743402339.554869 3227699 buffer_comparator.cc:156] Difference at 77: 19.521, expected 27.4887</span></span>
<span class="line"><span>E0000 00:00:1743402339.554872 3227699 buffer_comparator.cc:156] Difference at 94: 24.541, expected 28.5145</span></span>
<span class="line"><span>E0000 00:00:1743402339.554875 3227699 buffer_comparator.cc:156] Difference at 101: 30.6158, expected 26.8436</span></span>
<span class="line"><span>2025-03-31 06:25:39.554880: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.72041</span></span>
<span class="line"><span>Validation:	Loss 0.62017	Accuracy 0.53125</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.57444</span></span>
<span class="line"><span>Validation:	Loss 0.49094	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.46037</span></span>
<span class="line"><span>Validation:	Loss 0.38850	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.33640</span></span>
<span class="line"><span>Validation:	Loss 0.24873	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22493</span></span>
<span class="line"><span>Validation:	Loss 0.17702	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16317</span></span>
<span class="line"><span>Validation:	Loss 0.12982	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12090</span></span>
<span class="line"><span>Validation:	Loss 0.09821	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09324</span></span>
<span class="line"><span>Validation:	Loss 0.07686	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.07354</span></span>
<span class="line"><span>Validation:	Loss 0.06189	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.05976</span></span>
<span class="line"><span>Validation:	Loss 0.05095	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.04950</span></span>
<span class="line"><span>Validation:	Loss 0.04270	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.04170</span></span>
<span class="line"><span>Validation:	Loss 0.03631	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.03608</span></span>
<span class="line"><span>Validation:	Loss 0.03093	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.03066</span></span>
<span class="line"><span>Validation:	Loss 0.02633	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02647</span></span>
<span class="line"><span>Validation:	Loss 0.02309	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02364</span></span>
<span class="line"><span>Validation:	Loss 0.02064	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.02120</span></span>
<span class="line"><span>Validation:	Loss 0.01860	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01925</span></span>
<span class="line"><span>Validation:	Loss 0.01692	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01748</span></span>
<span class="line"><span>Validation:	Loss 0.01549	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01601</span></span>
<span class="line"><span>Validation:	Loss 0.01428	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01483</span></span>
<span class="line"><span>Validation:	Loss 0.01320	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01383</span></span>
<span class="line"><span>Validation:	Loss 0.01226	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01280</span></span>
<span class="line"><span>Validation:	Loss 0.01141	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01199</span></span>
<span class="line"><span>Validation:	Loss 0.01063	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.01110</span></span>
<span class="line"><span>Validation:	Loss 0.00992	Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-6/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.73543</span></span>
<span class="line"><span>Validation:	Loss 0.62151	Accuracy 0.59375</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.57847</span></span>
<span class="line"><span>Validation:	Loss 0.49948	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.44122</span></span>
<span class="line"><span>Validation:	Loss 0.40653	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.34849</span></span>
<span class="line"><span>Validation:	Loss 0.32982	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.26977</span></span>
<span class="line"><span>Validation:	Loss 0.24825	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.18905</span></span>
<span class="line"><span>Validation:	Loss 0.14689	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10057</span></span>
<span class="line"><span>Validation:	Loss 0.06757	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05630</span></span>
<span class="line"><span>Validation:	Loss 0.04480	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04038</span></span>
<span class="line"><span>Validation:	Loss 0.03372	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03148</span></span>
<span class="line"><span>Validation:	Loss 0.02693	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02564</span></span>
<span class="line"><span>Validation:	Loss 0.02240	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02149</span></span>
<span class="line"><span>Validation:	Loss 0.01912	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01848</span></span>
<span class="line"><span>Validation:	Loss 0.01663	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01615</span></span>
<span class="line"><span>Validation:	Loss 0.01463	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01424</span></span>
<span class="line"><span>Validation:	Loss 0.01295	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01265</span></span>
<span class="line"><span>Validation:	Loss 0.01153	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01132</span></span>
<span class="line"><span>Validation:	Loss 0.01032	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01018</span></span>
<span class="line"><span>Validation:	Loss 0.00930	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00924</span></span>
<span class="line"><span>Validation:	Loss 0.00843	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00842</span></span>
<span class="line"><span>Validation:	Loss 0.00770	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00774</span></span>
<span class="line"><span>Validation:	Loss 0.00708	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00713</span></span>
<span class="line"><span>Validation:	Loss 0.00655	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00662</span></span>
<span class="line"><span>Validation:	Loss 0.00609	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00618</span></span>
<span class="line"><span>Validation:	Loss 0.00569	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00578</span></span>
<span class="line"><span>Validation:	Loss 0.00534	Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.4</span></span>
<span class="line"><span>Commit 8561cc3d68d (2025-03-10 11:36 UTC)</span></span>
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
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,47)]))}const f=a(p,[["render",t]]);export{d as __pageData,f as default};
