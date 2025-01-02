import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.bV3h_rQg.js";const E=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    385.3 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    418.3 ms  ✓ ShowCases</span></span>
<span class="line"><span>    326.9 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    339.9 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    853.4 ms  ✓ InitialValues</span></span>
<span class="line"><span>    438.6 ms  ✓ NameResolution</span></span>
<span class="line"><span>    511.5 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>   1033.6 ms  ✓ Baselet</span></span>
<span class="line"><span>   2904.4 ms  ✓ Accessors</span></span>
<span class="line"><span>    650.0 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>    867.1 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>    783.0 ms  ✓ BangBang</span></span>
<span class="line"><span>    697.9 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>    599.2 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    778.7 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    692.2 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    912.4 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2857.3 ms  ✓ Transducers</span></span>
<span class="line"><span>    819.5 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>  17658.5 ms  ✓ MLStyle</span></span>
<span class="line"><span>   4406.8 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   5013.0 ms  ✓ FLoops</span></span>
<span class="line"><span>   6435.1 ms  ✓ MLUtils</span></span>
<span class="line"><span>  23 dependencies successfully precompiled in 35 seconds. 75 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangDataFramesExt...</span></span>
<span class="line"><span>   1743.8 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling TransducersDataFramesExt...</span></span>
<span class="line"><span>   1401.2 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 61 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1637.3 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2202.6 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    350.4 ms  ✓ RealDot</span></span>
<span class="line"><span>    915.8 ms  ✓ FillArrays</span></span>
<span class="line"><span>    394.1 ms  ✓ FillArrays → FillArraysStatisticsExt</span></span>
<span class="line"><span>    673.4 ms  ✓ FillArrays → FillArraysSparseArraysExt</span></span>
<span class="line"><span>   5293.2 ms  ✓ ChainRules</span></span>
<span class="line"><span>  34118.2 ms  ✓ Zygote</span></span>
<span class="line"><span>  6 dependencies successfully precompiled in 40 seconds. 80 already precompiled.</span></span>
<span class="line"><span>Precompiling AccessorsStructArraysExt...</span></span>
<span class="line"><span>    631.2 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 16 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    562.4 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 22 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    893.6 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 39 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    880.0 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    662.6 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1806.4 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   2932.0 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 163 already precompiled.</span></span>
<span class="line"><span>Precompiling ZygoteColorsExt...</span></span>
<span class="line"><span>   1859.4 ms  ✓ Zygote → ZygoteColorsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 89 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62465</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58986</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56698</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53841</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51990</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50342</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47127</span></span>
<span class="line"><span>Validation: Loss 0.46678 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47232 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46317</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45267</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43708</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43490</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41378</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39799</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37406</span></span>
<span class="line"><span>Validation: Loss 0.36924 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37579 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36743</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34190</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35161</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34061</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31952</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31031</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32382</span></span>
<span class="line"><span>Validation: Loss 0.28414 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29142 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27683</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27128</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26858</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25897</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23476</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25121</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21990</span></span>
<span class="line"><span>Validation: Loss 0.21412 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22147 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20933</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21133</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19747</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18728</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17372</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18904</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17825</span></span>
<span class="line"><span>Validation: Loss 0.15828 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16520 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15830</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15860</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14673</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14287</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13011</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12636</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13122</span></span>
<span class="line"><span>Validation: Loss 0.11541 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12118 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11639</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11139</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10860</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10459</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09966</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09078</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07577</span></span>
<span class="line"><span>Validation: Loss 0.08221 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08649 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08716</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08033</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07555</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07487</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06573</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06127</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05912</span></span>
<span class="line"><span>Validation: Loss 0.05725 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06007 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06061</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05405</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05133</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05029</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04662</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04815</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05083</span></span>
<span class="line"><span>Validation: Loss 0.04270 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04468 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04193</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04194</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04123</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04153</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03823</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03678</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03139</span></span>
<span class="line"><span>Validation: Loss 0.03457 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03617 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03567</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03311</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03422</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03264</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03216</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03169</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02645</span></span>
<span class="line"><span>Validation: Loss 0.02935 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03075 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02934</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03021</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02741</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02853</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02565</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02998</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02280</span></span>
<span class="line"><span>Validation: Loss 0.02555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02680 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02488</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02550</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02702</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02376</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02359</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02358</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02668</span></span>
<span class="line"><span>Validation: Loss 0.02261 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02374 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02200</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02340</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02187</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02210</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02195</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02123</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02091</span></span>
<span class="line"><span>Validation: Loss 0.02022 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02125 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02002</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02153</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01962</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02005</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01867</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01894</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01952</span></span>
<span class="line"><span>Validation: Loss 0.01823 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01919 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01788</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01890</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01848</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01789</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01717</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01709</span></span>
<span class="line"><span>Validation: Loss 0.01656 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01745 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01689</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01665</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01536</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01699</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01708</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01536</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01442</span></span>
<span class="line"><span>Validation: Loss 0.01514 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01596 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01528</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01466</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01477</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01475</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01413</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01660</span></span>
<span class="line"><span>Validation: Loss 0.01392 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01469 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01408</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01308</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01319</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01505</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01502</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01263</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01222</span></span>
<span class="line"><span>Validation: Loss 0.01286 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01358 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01209</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01284</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01220</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01276</span></span>
<span class="line"><span>Validation: Loss 0.01192 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01260 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01240</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01180</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01156</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01190</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00917</span></span>
<span class="line"><span>Validation: Loss 0.01105 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01168 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01053</span></span>
<span class="line"><span>Validation: Loss 0.01018 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01076 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00957</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00997</span></span>
<span class="line"><span>Validation: Loss 0.00922 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00974 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00980</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00963</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00882</span></span>
<span class="line"><span>Validation: Loss 0.00820 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00865 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00828</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00706</span></span>
<span class="line"><span>Validation: Loss 0.00737 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00775 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62429</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59917</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57710</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53579</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51832</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49359</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50722</span></span>
<span class="line"><span>Validation: Loss 0.45968 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46715 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47981</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44931</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43441</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42601</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41689</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40442</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38532</span></span>
<span class="line"><span>Validation: Loss 0.36120 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37024 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36973</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35767</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35360</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32973</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33629</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30418</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31509</span></span>
<span class="line"><span>Validation: Loss 0.27486 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28506 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27583</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27923</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26989</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26685</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24275</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24545</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22559</span></span>
<span class="line"><span>Validation: Loss 0.20504 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21539 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22713</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20872</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21017</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18791</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18411</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17533</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16921</span></span>
<span class="line"><span>Validation: Loss 0.15059 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16012 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16174</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16177</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15272</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14987</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14318</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12024</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11964</span></span>
<span class="line"><span>Validation: Loss 0.10962 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11752 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11581</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12519</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10881</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09844</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11385</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08321</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10682</span></span>
<span class="line"><span>Validation: Loss 0.07849 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08441 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08878</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08855</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07305</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06890</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07432</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06967</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05608</span></span>
<span class="line"><span>Validation: Loss 0.05492 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05885 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06083</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05876</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05764</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05034</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04752</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04840</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05055</span></span>
<span class="line"><span>Validation: Loss 0.04084 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04358 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04599</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04338</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04271</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04006</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04113</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03455</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03918</span></span>
<span class="line"><span>Validation: Loss 0.03298 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03518 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03663</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03711</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03364</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03228</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03347</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03054</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03412</span></span>
<span class="line"><span>Validation: Loss 0.02792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02982 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03001</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02963</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03204</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02673</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02912</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02688</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02967</span></span>
<span class="line"><span>Validation: Loss 0.02425 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02594 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02680</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02633</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02674</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02448</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02560</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02281</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02622</span></span>
<span class="line"><span>Validation: Loss 0.02142 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02293 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02600</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02330</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02253</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02239</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01968</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02170</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02424</span></span>
<span class="line"><span>Validation: Loss 0.01914 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02052 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02237</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02036</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02060</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02084</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01929</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01902</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01910</span></span>
<span class="line"><span>Validation: Loss 0.01727 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01853 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02005</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01858</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01713</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01908</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01982</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01644</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01671</span></span>
<span class="line"><span>Validation: Loss 0.01568 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01685 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01665</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01675</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01685</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01726</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01689</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01654</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01649</span></span>
<span class="line"><span>Validation: Loss 0.01431 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01541 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01553</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01563</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01539</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01679</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01402</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01628</span></span>
<span class="line"><span>Validation: Loss 0.01312 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01414 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01467</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01449</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01402</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01332</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01435</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01601</span></span>
<span class="line"><span>Validation: Loss 0.01205 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01301 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01506</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01270</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01280</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01328</span></span>
<span class="line"><span>Validation: Loss 0.01106 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01195 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01250</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01255</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01076</span></span>
<span class="line"><span>Validation: Loss 0.01005 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01086 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01113</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01157</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01003</span></span>
<span class="line"><span>Validation: Loss 0.00897 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00968 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01042</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00883</span></span>
<span class="line"><span>Validation: Loss 0.00796 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00856 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00801</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00827</span></span>
<span class="line"><span>Validation: Loss 0.00722 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00775 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00713</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00786</span></span>
<span class="line"><span>Validation: Loss 0.00669 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00716 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.141 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46)]))}const d=a(l,[["render",e]]);export{E as __pageData,d as default};
