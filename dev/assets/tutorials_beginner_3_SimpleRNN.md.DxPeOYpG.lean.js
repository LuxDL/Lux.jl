import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.BS99Di-t.js";const E=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    443.7 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    412.4 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    342.3 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    795.1 ms  ✓ InitialValues</span></span>
<span class="line"><span>    412.6 ms  ✓ ShowCases</span></span>
<span class="line"><span>    324.6 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    314.7 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>   1046.4 ms  ✓ Baselet</span></span>
<span class="line"><span>    717.4 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>    383.0 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>   2271.4 ms  ✓ StatsBase</span></span>
<span class="line"><span>    427.8 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    380.8 ms  ✓ NameResolution</span></span>
<span class="line"><span>    363.4 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>   2867.9 ms  ✓ Accessors</span></span>
<span class="line"><span>    613.4 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>    813.6 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>    767.5 ms  ✓ BangBang</span></span>
<span class="line"><span>    697.9 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>    509.7 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    766.5 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    479.0 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    853.5 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2781.8 ms  ✓ Transducers</span></span>
<span class="line"><span>    664.2 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>  18339.7 ms  ✓ MLStyle</span></span>
<span class="line"><span>   4178.1 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   5026.9 ms  ✓ FLoops</span></span>
<span class="line"><span>   7177.3 ms  ✓ MLUtils</span></span>
<span class="line"><span>  29 dependencies successfully precompiled in 36 seconds. 83 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangDataFramesExt...</span></span>
<span class="line"><span>   1671.3 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling TransducersDataFramesExt...</span></span>
<span class="line"><span>   1376.4 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 61 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   2475.4 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 116 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   3342.5 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 178 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    324.8 ms  ✓ RealDot</span></span>
<span class="line"><span>    867.5 ms  ✓ FillArrays</span></span>
<span class="line"><span>    380.8 ms  ✓ FillArrays → FillArraysStatisticsExt</span></span>
<span class="line"><span>    647.1 ms  ✓ FillArrays → FillArraysSparseArraysExt</span></span>
<span class="line"><span>   5262.8 ms  ✓ ChainRules</span></span>
<span class="line"><span>  33028.6 ms  ✓ Zygote</span></span>
<span class="line"><span>  6 dependencies successfully precompiled in 39 seconds. 80 already precompiled.</span></span>
<span class="line"><span>Precompiling AccessorsStructArraysExt...</span></span>
<span class="line"><span>    456.4 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 16 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    462.1 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 22 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    780.2 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 39 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    875.2 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    437.9 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 0 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1592.9 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   3493.1 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 162 already precompiled.</span></span>
<span class="line"><span>Precompiling ZygoteColorsExt...</span></span>
<span class="line"><span>   1830.8 ms  ✓ Zygote → ZygoteColorsExt</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62244</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58563</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56945</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54192</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52251</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49668</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48818</span></span>
<span class="line"><span>Validation: Loss 0.47052 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46502 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46987</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46100</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43921</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41588</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42149</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39460</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36600</span></span>
<span class="line"><span>Validation: Loss 0.37324 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36694 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36549</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35573</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33862</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33932</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31978</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31063</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33971</span></span>
<span class="line"><span>Validation: Loss 0.28901 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28181 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28518</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28421</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25942</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25395</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24666</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23445</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23104</span></span>
<span class="line"><span>Validation: Loss 0.21971 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21248 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21918</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19529</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20878</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19330</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18497</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17591</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17313</span></span>
<span class="line"><span>Validation: Loss 0.16406 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15726 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16962</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14637</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14709</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14382</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14049</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12576</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12428</span></span>
<span class="line"><span>Validation: Loss 0.12078 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11506 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11568</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11349</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11329</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10223</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09601</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10109</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07017</span></span>
<span class="line"><span>Validation: Loss 0.08677 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08248 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08431</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09227</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07769</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06996</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06343</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06609</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06266</span></span>
<span class="line"><span>Validation: Loss 0.06051 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05761 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05844</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05279</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05760</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05163</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04816</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04981</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04411</span></span>
<span class="line"><span>Validation: Loss 0.04471 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04265 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04441</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04219</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03909</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03983</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04153</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03753</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03284</span></span>
<span class="line"><span>Validation: Loss 0.03613 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03447 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03615</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03289</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03663</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03218</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03062</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03119</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03502</span></span>
<span class="line"><span>Validation: Loss 0.03066 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02921 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03143</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02876</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02990</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03124</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02560</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02535</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02579</span></span>
<span class="line"><span>Validation: Loss 0.02666 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02598</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02325</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02529</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02658</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02482</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02461</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02275</span></span>
<span class="line"><span>Validation: Loss 0.02359 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02244 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02216</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02239</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02164</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02269</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02155</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02219</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02489</span></span>
<span class="line"><span>Validation: Loss 0.02113 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02007 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02019</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02046</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02253</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01876</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01846</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01944</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01967</span></span>
<span class="line"><span>Validation: Loss 0.01907 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01809 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01803</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01778</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01905</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01862</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01771</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01781</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01530</span></span>
<span class="line"><span>Validation: Loss 0.01733 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01643 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01686</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01599</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01626</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01668</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01724</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01580</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01574</span></span>
<span class="line"><span>Validation: Loss 0.01585 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01502 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01437</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01680</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01588</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01569</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01448</span></span>
<span class="line"><span>Validation: Loss 0.01459 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01380 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01511</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01400</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01492</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01354</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01324</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01391</span></span>
<span class="line"><span>Validation: Loss 0.01348 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01275 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01374</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01268</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01214</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01277</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01388</span></span>
<span class="line"><span>Validation: Loss 0.01249 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01181 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01287</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01213</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01268</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01054</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01237</span></span>
<span class="line"><span>Validation: Loss 0.01155 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01093 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01131</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00985</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00970</span></span>
<span class="line"><span>Validation: Loss 0.01060 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01004 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01041</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01016</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01046</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00964</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00952</span></span>
<span class="line"><span>Validation: Loss 0.00954 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00904 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00838</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01055</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00838</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00782</span></span>
<span class="line"><span>Validation: Loss 0.00846 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00803 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00710</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00798</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00648</span></span>
<span class="line"><span>Validation: Loss 0.00762 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00724 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61938</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59454</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57434</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53039</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52410</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49497</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49160</span></span>
<span class="line"><span>Validation: Loss 0.46947 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46692 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46402</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45123</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43742</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42572</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41527</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40598</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37909</span></span>
<span class="line"><span>Validation: Loss 0.37255 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36950 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36007</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34565</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35685</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33282</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32875</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30929</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31886</span></span>
<span class="line"><span>Validation: Loss 0.28792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28436 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28622</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25804</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27956</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26484</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24611</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23255</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20014</span></span>
<span class="line"><span>Validation: Loss 0.21830 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21459 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21627</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22357</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19682</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17828</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18891</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16946</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17334</span></span>
<span class="line"><span>Validation: Loss 0.16287 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15937 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16394</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16296</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14653</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13824</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14026</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12002</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12332</span></span>
<span class="line"><span>Validation: Loss 0.11975 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11683 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12169</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10801</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11586</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10483</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10103</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08785</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07722</span></span>
<span class="line"><span>Validation: Loss 0.08578 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08360 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07586</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08572</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07828</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07683</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06598</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06680</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06211</span></span>
<span class="line"><span>Validation: Loss 0.05981 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05832 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06136</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05387</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05417</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05180</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04966</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04614</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04715</span></span>
<span class="line"><span>Validation: Loss 0.04433 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04329 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04248</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04239</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04241</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04014</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03735</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03749</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04004</span></span>
<span class="line"><span>Validation: Loss 0.03581 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03498 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03549</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03577</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03344</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03349</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03079</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03145</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03077</span></span>
<span class="line"><span>Validation: Loss 0.03035 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02964 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03111</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03121</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02738</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02803</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02580</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02786</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02751</span></span>
<span class="line"><span>Validation: Loss 0.02639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02576 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02685</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02724</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02277</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02670</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02412</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02266</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02267</span></span>
<span class="line"><span>Validation: Loss 0.02332 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02276 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02414</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02233</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02356</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02203</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02128</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01961</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02250</span></span>
<span class="line"><span>Validation: Loss 0.02087 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02036 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02143</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02112</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02043</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01922</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01807</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01920</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01962</span></span>
<span class="line"><span>Validation: Loss 0.01884 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01836 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01798</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01773</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01881</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01691</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01832</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01840</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01722</span></span>
<span class="line"><span>Validation: Loss 0.01713 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01668 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01680</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01722</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01637</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01615</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01671</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01516</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01630</span></span>
<span class="line"><span>Validation: Loss 0.01565 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01524 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01461</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01618</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01441</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01519</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01558</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01107</span></span>
<span class="line"><span>Validation: Loss 0.01439 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01400 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01298</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01439</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01121</span></span>
<span class="line"><span>Validation: Loss 0.01330 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01293 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01407</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01315</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01299</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01358</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01231</span></span>
<span class="line"><span>Validation: Loss 0.01231 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01197 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01231</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01229</span></span>
<span class="line"><span>Validation: Loss 0.01135 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01104 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01101</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01008</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01208</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01055</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00873</span></span>
<span class="line"><span>Validation: Loss 0.01032 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01003 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01023</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00855</span></span>
<span class="line"><span>Validation: Loss 0.00918 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00893 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00773</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00849</span></span>
<span class="line"><span>Validation: Loss 0.00818 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00796 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00780</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00693</span></span>
<span class="line"><span>Validation: Loss 0.00745 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00726 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
