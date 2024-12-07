import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.BS99Di-t.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>   6765.8 ms  ✓ LLVM</span></span>
<span class="line"><span>   1950.3 ms  ✓ UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   4761.8 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>   1694.8 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>   1742.6 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   6418.6 ms  ✓ NNlib</span></span>
<span class="line"><span>   1794.2 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>   1794.6 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   6508.4 ms  ✓ LuxLib</span></span>
<span class="line"><span>  10043.5 ms  ✓ Lux</span></span>
<span class="line"><span>  10 dependencies successfully precompiled in 40 seconds. 113 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>   1329.4 ms  ✓ LLVM → BFloat16sExt</span></span>
<span class="line"><span>   2162.1 ms  ✓ GPUArrays</span></span>
<span class="line"><span>   1913.6 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>  27316.8 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>  47414.8 ms  ✓ DataFrames</span></span>
<span class="line"><span>  51806.1 ms  ✓ CUDA</span></span>
<span class="line"><span>   5229.3 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8102.7 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5393.5 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  9 dependencies successfully precompiled in 118 seconds. 91 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesGPUArraysExt...</span></span>
<span class="line"><span>   1344.1 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 42 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersGPUArraysExt...</span></span>
<span class="line"><span>   1391.1 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   4936.7 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   5040.9 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5476.4 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   5478.3 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5312.5 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5386.7 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   5904.4 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 6 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   5098.1 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5359.7 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5017.9 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 107 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   5869.2 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 174 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>   4194.9 ms  ✓ FileIO</span></span>
<span class="line"><span>  32669.6 ms  ✓ JLD2</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 37 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    438.7 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>   7085.2 ms  ✓ MLUtils</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 8 seconds. 110 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangDataFramesExt...</span></span>
<span class="line"><span>   1707.8 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling TransducersDataFramesExt...</span></span>
<span class="line"><span>   1441.3 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 61 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   2498.1 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 116 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   3180.3 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 178 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   3543.7 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 162 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62402</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59522</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56964</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53364</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52086</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49861</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47121</span></span>
<span class="line"><span>Validation: Loss 0.46627 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47571 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46712</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45269</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43787</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42902</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41029</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39566</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38302</span></span>
<span class="line"><span>Validation: Loss 0.36816 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37948 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37039</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35525</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35102</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33405</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31028</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30383</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31207</span></span>
<span class="line"><span>Validation: Loss 0.28226 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29448 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28892</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26992</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26028</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25352</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24019</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23345</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22817</span></span>
<span class="line"><span>Validation: Loss 0.21175 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22384 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21413</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19617</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19543</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19682</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18388</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17526</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14524</span></span>
<span class="line"><span>Validation: Loss 0.15605 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16702 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15462</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15657</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14184</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13080</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13508</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13146</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12461</span></span>
<span class="line"><span>Validation: Loss 0.11380 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12278 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11307</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11248</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11567</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10063</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09674</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07919</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09310</span></span>
<span class="line"><span>Validation: Loss 0.08112 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08764 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07844</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07564</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07725</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06845</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07215</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06469</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05620</span></span>
<span class="line"><span>Validation: Loss 0.05658 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06087 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06293</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05309</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05230</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04793</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04758</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04576</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04178</span></span>
<span class="line"><span>Validation: Loss 0.04220 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04527 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04296</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04145</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04111</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03798</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03958</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03522</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03229</span></span>
<span class="line"><span>Validation: Loss 0.03423 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03676 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03646</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03365</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03308</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03048</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03108</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03164</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02915</span></span>
<span class="line"><span>Validation: Loss 0.02908 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03129 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03229</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03064</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02788</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02571</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02679</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02560</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02439</span></span>
<span class="line"><span>Validation: Loss 0.02531 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02728 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02511</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02526</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02610</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02565</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02317</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02194</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02393</span></span>
<span class="line"><span>Validation: Loss 0.02239 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02418 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02279</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02242</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02207</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02152</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02136</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02047</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02219</span></span>
<span class="line"><span>Validation: Loss 0.02004 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02168 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02186</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01831</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01891</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02038</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01886</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01844</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02206</span></span>
<span class="line"><span>Validation: Loss 0.01808 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01959 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01724</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02013</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01749</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01884</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01640</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01447</span></span>
<span class="line"><span>Validation: Loss 0.01641 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01781 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01641</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01674</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01517</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01543</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01613</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01665</span></span>
<span class="line"><span>Validation: Loss 0.01500 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01631 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01458</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01510</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01478</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01600</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01263</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01548</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01477</span></span>
<span class="line"><span>Validation: Loss 0.01379 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01501 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01527</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01345</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01342</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01302</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01413</span></span>
<span class="line"><span>Validation: Loss 0.01273 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01387 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01299</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01320</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01388</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01206</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01310</span></span>
<span class="line"><span>Validation: Loss 0.01178 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01284 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01213</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01188</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01083</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01035</span></span>
<span class="line"><span>Validation: Loss 0.01087 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01185 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01146</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01080</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00871</span></span>
<span class="line"><span>Validation: Loss 0.00993 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01080 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00955</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00968</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00955</span></span>
<span class="line"><span>Validation: Loss 0.00889 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00965 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00908</span></span>
<span class="line"><span>Validation: Loss 0.00792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00857 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00684</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00718</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00742</span></span>
<span class="line"><span>Validation: Loss 0.00720 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00776 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62246</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59001</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56167</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55490</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51723</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49919</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45974</span></span>
<span class="line"><span>Validation: Loss 0.47309 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46329 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47877</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44441</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43693</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42973</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41505</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39439</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40156</span></span>
<span class="line"><span>Validation: Loss 0.37835 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36595 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37451</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36473</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34369</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32642</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32345</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31360</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32194</span></span>
<span class="line"><span>Validation: Loss 0.29478 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28132 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27677</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28748</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26227</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26479</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24051</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24417</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23966</span></span>
<span class="line"><span>Validation: Loss 0.22516 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21185 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21012</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22473</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20520</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18600</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18204</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18286</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17125</span></span>
<span class="line"><span>Validation: Loss 0.16883 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15674 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17415</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16078</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14020</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14261</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14263</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13063</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09975</span></span>
<span class="line"><span>Validation: Loss 0.12449 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11455 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11946</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11732</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10993</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10990</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10187</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08996</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08099</span></span>
<span class="line"><span>Validation: Loss 0.08955 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08210 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08896</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08132</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07317</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07452</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07160</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06821</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06330</span></span>
<span class="line"><span>Validation: Loss 0.06237 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05734 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06038</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05938</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05670</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05363</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04623</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04832</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04185</span></span>
<span class="line"><span>Validation: Loss 0.04631 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04274 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04753</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04515</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04231</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03925</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03944</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03605</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03344</span></span>
<span class="line"><span>Validation: Loss 0.03755 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03464 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03572</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03907</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03396</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03136</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03415</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03082</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03151</span></span>
<span class="line"><span>Validation: Loss 0.03194 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02940 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03222</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02809</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02933</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03006</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02795</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02716</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03137</span></span>
<span class="line"><span>Validation: Loss 0.02783 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02557 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02638</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.03021</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02663</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02335</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02341</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02367</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02595</span></span>
<span class="line"><span>Validation: Loss 0.02461 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02259 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02376</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02381</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02467</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02200</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02329</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02049</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01675</span></span>
<span class="line"><span>Validation: Loss 0.02204 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02020 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02006</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02075</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02105</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02226</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01948</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02032</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01509</span></span>
<span class="line"><span>Validation: Loss 0.01994 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01824 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01958</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01860</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01960</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01651</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01734</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01882</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02118</span></span>
<span class="line"><span>Validation: Loss 0.01817 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01659 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01817</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01706</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01667</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01557</span></span>
<span class="line"><span>Validation: Loss 0.01662 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01515 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01647</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01498</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01404</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01539</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01611</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01608</span></span>
<span class="line"><span>Validation: Loss 0.01529 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01391 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01353</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01643</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01320</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01435</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01461</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01387</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01206</span></span>
<span class="line"><span>Validation: Loss 0.01412 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01283 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01371</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01315</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01315</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01352</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01272</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01322</span></span>
<span class="line"><span>Validation: Loss 0.01308 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01189 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01271</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01270</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01306</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01369</span></span>
<span class="line"><span>Validation: Loss 0.01212 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01101 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01223</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01139</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01252</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01078</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01040</span></span>
<span class="line"><span>Validation: Loss 0.01114 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01012 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01024</span></span>
<span class="line"><span>Validation: Loss 0.01005 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00914 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00910</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00999</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00904</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00907</span></span>
<span class="line"><span>Validation: Loss 0.00890 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00812 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00821</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00840</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00724</span></span>
<span class="line"><span>Validation: Loss 0.00799 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00731 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.141 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
