import{_ as a,c as n,o as i,al as e}from"./chunks/framework.BCN3FD2k.js";const d=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"};function l(t,s,c,r,h,k){return i(),n("div",null,s[0]||(s[0]=[e(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><p>Note: If you wish to use AutoZygote() for automatic differentiation, add Zygote to your project dependencies and include <code>using Zygote</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    467.1 ms  ✓ OpenLibm_jll</span></span>
<span class="line"><span>    509.7 ms  ✓ Statistics</span></span>
<span class="line"><span>    512.7 ms  ✓ DocStringExtensions</span></span>
<span class="line"><span>    420.9 ms  ✓ ConstructionBase</span></span>
<span class="line"><span>    575.3 ms  ✓ NaNMath</span></span>
<span class="line"><span>    362.5 ms  ✓ ConstructionBase → ConstructionBaseLinearAlgebraExt</span></span>
<span class="line"><span>   1125.1 ms  ✓ IrrationalConstants</span></span>
<span class="line"><span>    733.4 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    373.6 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>    624.7 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    616.3 ms  ✓ Functors</span></span>
<span class="line"><span>    653.5 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>   1446.7 ms  ✓ Setfield</span></span>
<span class="line"><span>    833.4 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>    459.3 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>   1249.0 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>   1315.4 ms  ✓ Optimisers</span></span>
<span class="line"><span>    430.4 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    744.1 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>    463.9 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    428.0 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>   2458.9 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    424.0 ms  ✓ Optimisers → OptimisersAdaptExt</span></span>
<span class="line"><span>   5658.6 ms  ✓ NNlib</span></span>
<span class="line"><span>    634.6 ms  ✓ DiffRules</span></span>
<span class="line"><span>   1605.3 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    840.3 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    832.2 ms  ✓ NNlib → NNlibSpecialFunctionsExt</span></span>
<span class="line"><span>   2601.4 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    904.8 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>   3557.6 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    791.5 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>    891.2 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5430.3 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9295.5 ms  ✓ Lux</span></span>
<span class="line"><span>  35 dependencies successfully precompiled in 29 seconds. 70 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>   4217.0 ms  ✓ FileIO</span></span>
<span class="line"><span>  31810.7 ms  ✓ JLD2</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 36 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    391.2 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    640.1 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>   1317.4 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>   2468.0 ms  ✓ Accessors</span></span>
<span class="line"><span>   2307.2 ms  ✓ StatsBase</span></span>
<span class="line"><span>    670.0 ms  ✓ Accessors → TestExt</span></span>
<span class="line"><span>    900.3 ms  ✓ Accessors → LinearAlgebraExt</span></span>
<span class="line"><span>    679.5 ms  ✓ Accessors → StaticArraysExt</span></span>
<span class="line"><span>    749.2 ms  ✓ BangBang</span></span>
<span class="line"><span>    503.3 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    504.9 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    700.9 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>   1045.2 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2707.1 ms  ✓ Transducers</span></span>
<span class="line"><span>    669.0 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   5244.7 ms  ✓ FLoops</span></span>
<span class="line"><span>   5807.8 ms  ✓ MLUtils</span></span>
<span class="line"><span>  17 dependencies successfully precompiled in 20 seconds. 80 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    671.3 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1509.4 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   1994.5 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 164 already precompiled.</span></span>
<span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>   2334.1 ms  ✓ Reactant_jll</span></span>
<span class="line"><span>  75318.2 ms  ✓ Reactant</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 78 seconds. 77 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibEnzymeExt...</span></span>
<span class="line"><span>   1229.8 ms  ✓ LuxLib → LuxLibEnzymeExt</span></span>
<span class="line"><span>   5604.9 ms  ✓ Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>   5803.1 ms  ✓ Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 6 seconds. 130 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   6584.4 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 148 already precompiled.</span></span>
<span class="line"><span>Precompiling HTTPExt...</span></span>
<span class="line"><span>   1764.1 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 43 already precompiled.</span></span>
<span class="line"><span>Precompiling OptimisersReactantExt...</span></span>
<span class="line"><span>  12776.1 ms  ✓ Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>  15554.6 ms  ✓ Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 16 seconds. 87 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  13130.4 ms  ✓ LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 84 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  12763.6 ms  ✓ MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 81 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  12712.0 ms  ✓ Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>  12748.0 ms  ✓ WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 13 seconds. 94 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantKernelAbstractionsExt...</span></span>
<span class="line"><span>  13177.4 ms  ✓ Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 91 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantArrayInterfaceExt...</span></span>
<span class="line"><span>  12644.5 ms  ✓ Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  13241.3 ms  ✓ Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  11078.9 ms  ✓ Lux → LuxReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 12 seconds. 179 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-13/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>2025-04-05 14:53:34.026423: I external/xla/xla/service/service.cc:152] XLA service 0x8ab4970 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-04-05 14:53:34.026760: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1743864814.029613   60461 se_gpu_pjrt_client.cc:1040] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1743864814.029911   60461 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743864814.030145   60461 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743864814.046415   60461 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>E0000 00:00:1743864871.141822   60461 buffer_comparator.cc:156] Difference at 32: 0, expected 1.62244</span></span>
<span class="line"><span>E0000 00:00:1743864871.142998   60461 buffer_comparator.cc:156] Difference at 33: 0, expected 1.87084</span></span>
<span class="line"><span>E0000 00:00:1743864871.143007   60461 buffer_comparator.cc:156] Difference at 34: 0, expected 1.07351</span></span>
<span class="line"><span>E0000 00:00:1743864871.143014   60461 buffer_comparator.cc:156] Difference at 35: 0, expected 2.92445</span></span>
<span class="line"><span>E0000 00:00:1743864871.143021   60461 buffer_comparator.cc:156] Difference at 36: 0, expected 1.98056</span></span>
<span class="line"><span>E0000 00:00:1743864871.143028   60461 buffer_comparator.cc:156] Difference at 37: 0, expected 2.07715</span></span>
<span class="line"><span>E0000 00:00:1743864871.143035   60461 buffer_comparator.cc:156] Difference at 38: 0, expected 1.56458</span></span>
<span class="line"><span>E0000 00:00:1743864871.143041   60461 buffer_comparator.cc:156] Difference at 39: 0, expected 2.27034</span></span>
<span class="line"><span>E0000 00:00:1743864871.143048   60461 buffer_comparator.cc:156] Difference at 40: 0, expected 2.31795</span></span>
<span class="line"><span>E0000 00:00:1743864871.143054   60461 buffer_comparator.cc:156] Difference at 41: 0, expected 2.55731</span></span>
<span class="line"><span>2025-04-05 14:54:31.143070: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.148086   60461 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743864871.148116   60461 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743864871.148124   60461 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743864871.148130   60461 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743864871.148136   60461 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743864871.148143   60461 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743864871.148149   60461 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743864871.148155   60461 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743864871.148161   60461 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743864871.148167   60461 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-05 14:54:31.148178: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.151056   60461 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743864871.151070   60461 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743864871.151073   60461 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743864871.151076   60461 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743864871.151079   60461 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743864871.151082   60461 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743864871.151084   60461 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743864871.151087   60461 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743864871.151091   60461 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743864871.151094   60461 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-05 14:54:31.151098: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.153648   60461 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743864871.153661   60461 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743864871.153664   60461 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743864871.153667   60461 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743864871.153670   60461 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743864871.153672   60461 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743864871.153675   60461 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743864871.153678   60461 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743864871.153681   60461 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743864871.153683   60461 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-05 14:54:31.153688: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.156252   60461 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743864871.156266   60461 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743864871.156269   60461 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743864871.156272   60461 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743864871.156275   60461 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743864871.156277   60461 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743864871.156280   60461 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743864871.156283   60461 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743864871.156286   60461 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743864871.156288   60461 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-05 14:54:31.156293: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.158843   60461 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743864871.158856   60461 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743864871.158859   60461 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743864871.158862   60461 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743864871.158865   60461 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743864871.158868   60461 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743864871.158871   60461 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743864871.158873   60461 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743864871.158876   60461 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743864871.158879   60461 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-05 14:54:31.158883: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.161440   60461 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743864871.161453   60461 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743864871.161456   60461 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743864871.161459   60461 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743864871.161462   60461 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743864871.161465   60461 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743864871.161468   60461 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743864871.161470   60461 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743864871.161473   60461 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743864871.161476   60461 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-05 14:54:31.161480: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.164060   60461 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743864871.164074   60461 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743864871.164077   60461 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743864871.164080   60461 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743864871.164082   60461 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743864871.164085   60461 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743864871.164088   60461 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743864871.164091   60461 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743864871.164094   60461 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743864871.164096   60461 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-05 14:54:31.164101: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.166646   60461 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743864871.166662   60461 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743864871.166665   60461 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743864871.166668   60461 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743864871.166671   60461 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743864871.166674   60461 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743864871.166676   60461 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743864871.166679   60461 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743864871.166682   60461 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743864871.166685   60461 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-05 14:54:31.166689: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.169235   60461 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743864871.169250   60461 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743864871.169253   60461 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743864871.169257   60461 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743864871.169260   60461 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743864871.169263   60461 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743864871.169265   60461 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743864871.169268   60461 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743864871.169271   60461 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743864871.169274   60461 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-05 14:54:31.169278: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.171826   60461 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743864871.171841   60461 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743864871.171844   60461 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743864871.171847   60461 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743864871.171850   60461 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743864871.171853   60461 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743864871.171856   60461 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743864871.171858   60461 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743864871.171861   60461 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743864871.171864   60461 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-05 14:54:31.171868: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.174438   60461 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743864871.174452   60461 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743864871.174455   60461 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743864871.174458   60461 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743864871.174461   60461 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743864871.174464   60461 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743864871.174466   60461 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743864871.174469   60461 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743864871.174472   60461 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743864871.174475   60461 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-05 14:54:31.174479: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.177030   60461 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743864871.177046   60461 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743864871.177049   60461 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743864871.177052   60461 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743864871.177055   60461 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743864871.177057   60461 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743864871.177060   60461 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743864871.177064   60461 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743864871.177067   60461 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743864871.177070   60461 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-05 14:54:31.177074: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.179620   60461 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743864871.179635   60461 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743864871.179638   60461 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743864871.179641   60461 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743864871.179644   60461 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743864871.179647   60461 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743864871.179649   60461 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743864871.179652   60461 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743864871.179655   60461 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743864871.179658   60461 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-05 14:54:31.179662: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.182269   60461 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743864871.182283   60461 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743864871.182286   60461 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743864871.182289   60461 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743864871.182292   60461 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743864871.182295   60461 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743864871.182297   60461 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743864871.182300   60461 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743864871.182303   60461 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743864871.182306   60461 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-05 14:54:31.182310: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.184898   60461 buffer_comparator.cc:156] Difference at 256: 0, expected 0.86224</span></span>
<span class="line"><span>E0000 00:00:1743864871.184915   60461 buffer_comparator.cc:156] Difference at 257: 0, expected 0.686873</span></span>
<span class="line"><span>E0000 00:00:1743864871.184918   60461 buffer_comparator.cc:156] Difference at 258: 0, expected 0.252371</span></span>
<span class="line"><span>E0000 00:00:1743864871.184921   60461 buffer_comparator.cc:156] Difference at 259: 0, expected 0.335927</span></span>
<span class="line"><span>E0000 00:00:1743864871.184924   60461 buffer_comparator.cc:156] Difference at 260: 0, expected 0.934139</span></span>
<span class="line"><span>E0000 00:00:1743864871.184927   60461 buffer_comparator.cc:156] Difference at 261: 0, expected 0.274756</span></span>
<span class="line"><span>E0000 00:00:1743864871.184930   60461 buffer_comparator.cc:156] Difference at 262: 0, expected 0.529946</span></span>
<span class="line"><span>E0000 00:00:1743864871.184932   60461 buffer_comparator.cc:156] Difference at 263: 0, expected 0.542969</span></span>
<span class="line"><span>E0000 00:00:1743864871.184935   60461 buffer_comparator.cc:156] Difference at 264: 0, expected 0.895372</span></span>
<span class="line"><span>E0000 00:00:1743864871.184938   60461 buffer_comparator.cc:156] Difference at 265: 0, expected 0.895664</span></span>
<span class="line"><span>2025-04-05 14:54:31.184942: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864871.187559   60461 buffer_comparator.cc:156] Difference at 256: 0, expected 0.86224</span></span>
<span class="line"><span>E0000 00:00:1743864871.187575   60461 buffer_comparator.cc:156] Difference at 257: 0, expected 0.686873</span></span>
<span class="line"><span>E0000 00:00:1743864871.187578   60461 buffer_comparator.cc:156] Difference at 258: 0, expected 0.252371</span></span>
<span class="line"><span>E0000 00:00:1743864871.187581   60461 buffer_comparator.cc:156] Difference at 259: 0, expected 0.335927</span></span>
<span class="line"><span>E0000 00:00:1743864871.187584   60461 buffer_comparator.cc:156] Difference at 260: 0, expected 0.934139</span></span>
<span class="line"><span>E0000 00:00:1743864871.187586   60461 buffer_comparator.cc:156] Difference at 261: 0, expected 0.274756</span></span>
<span class="line"><span>E0000 00:00:1743864871.187589   60461 buffer_comparator.cc:156] Difference at 262: 0, expected 0.529946</span></span>
<span class="line"><span>E0000 00:00:1743864871.187592   60461 buffer_comparator.cc:156] Difference at 263: 0, expected 0.542969</span></span>
<span class="line"><span>E0000 00:00:1743864871.187595   60461 buffer_comparator.cc:156] Difference at 264: 0, expected 0.895372</span></span>
<span class="line"><span>E0000 00:00:1743864871.187597   60461 buffer_comparator.cc:156] Difference at 265: 0, expected 0.895664</span></span>
<span class="line"><span>2025-04-05 14:54:31.187602: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.631886   60461 buffer_comparator.cc:156] Difference at 16: -6.07288, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1743864908.631935   60461 buffer_comparator.cc:156] Difference at 17: -0.034427, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1743864908.631943   60461 buffer_comparator.cc:156] Difference at 18: -5.91177, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1743864908.631951   60461 buffer_comparator.cc:156] Difference at 19: 1.09204, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1743864908.631958   60461 buffer_comparator.cc:156] Difference at 20: -5.39417, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1743864908.631964   60461 buffer_comparator.cc:156] Difference at 21: 2.47815, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1743864908.631971   60461 buffer_comparator.cc:156] Difference at 22: -4.79055, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1743864908.631978   60461 buffer_comparator.cc:156] Difference at 23: 3.22506, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1743864908.631984   60461 buffer_comparator.cc:156] Difference at 24: -4.25869, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1743864908.631991   60461 buffer_comparator.cc:156] Difference at 25: 4.04895, expected 36.4575</span></span>
<span class="line"><span>2025-04-05 14:55:08.632005: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.634420   60461 buffer_comparator.cc:156] Difference at 16: -6.07288, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1743864908.634446   60461 buffer_comparator.cc:156] Difference at 17: -0.034427, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1743864908.634454   60461 buffer_comparator.cc:156] Difference at 18: -5.91177, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1743864908.634461   60461 buffer_comparator.cc:156] Difference at 19: 1.09204, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1743864908.634468   60461 buffer_comparator.cc:156] Difference at 20: -5.39417, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1743864908.634475   60461 buffer_comparator.cc:156] Difference at 21: 2.47815, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1743864908.634481   60461 buffer_comparator.cc:156] Difference at 22: -4.79055, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1743864908.634488   60461 buffer_comparator.cc:156] Difference at 23: 3.22506, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1743864908.634494   60461 buffer_comparator.cc:156] Difference at 24: -4.25869, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1743864908.634501   60461 buffer_comparator.cc:156] Difference at 25: 4.04895, expected 36.4575</span></span>
<span class="line"><span>2025-04-05 14:55:08.634511: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.647773   60461 buffer_comparator.cc:156] Difference at 2: 37.361, expected 33.2434</span></span>
<span class="line"><span>E0000 00:00:1743864908.647788   60461 buffer_comparator.cc:156] Difference at 8: 32.9182, expected 29.0801</span></span>
<span class="line"><span>E0000 00:00:1743864908.647793   60461 buffer_comparator.cc:156] Difference at 11: 35.3152, expected 30.7625</span></span>
<span class="line"><span>E0000 00:00:1743864908.647796   60461 buffer_comparator.cc:156] Difference at 12: 39.5233, expected 34.3637</span></span>
<span class="line"><span>E0000 00:00:1743864908.647799   60461 buffer_comparator.cc:156] Difference at 20: 38.835, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1743864908.647802   60461 buffer_comparator.cc:156] Difference at 23: 37.0226, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1743864908.647805   60461 buffer_comparator.cc:156] Difference at 26: 39.1599, expected 32.4927</span></span>
<span class="line"><span>E0000 00:00:1743864908.647809   60461 buffer_comparator.cc:156] Difference at 51: 26.8307, expected 33.7879</span></span>
<span class="line"><span>2025-04-05 14:55:08.647814: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.656933   60461 buffer_comparator.cc:156] Difference at 16: 3.03813, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1743864908.656946   60461 buffer_comparator.cc:156] Difference at 17: 2.97279, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743864908.656949   60461 buffer_comparator.cc:156] Difference at 18: 3.35354, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1743864908.656953   60461 buffer_comparator.cc:156] Difference at 19: 2.30754, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1743864908.656956   60461 buffer_comparator.cc:156] Difference at 20: 3.71941, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743864908.656959   60461 buffer_comparator.cc:156] Difference at 21: 1.62894, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1743864908.656962   60461 buffer_comparator.cc:156] Difference at 22: 3.92903, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1743864908.656964   60461 buffer_comparator.cc:156] Difference at 23: 0.799771, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1743864908.656968   60461 buffer_comparator.cc:156] Difference at 24: 4.01028, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1743864908.656971   60461 buffer_comparator.cc:156] Difference at 25: -0.0661373, expected 36.0917</span></span>
<span class="line"><span>2025-04-05 14:55:08.656976: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.659107   60461 buffer_comparator.cc:156] Difference at 16: 3.03813, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1743864908.659119   60461 buffer_comparator.cc:156] Difference at 17: 2.97279, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743864908.659122   60461 buffer_comparator.cc:156] Difference at 18: 3.35354, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1743864908.659125   60461 buffer_comparator.cc:156] Difference at 19: 2.30754, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1743864908.659128   60461 buffer_comparator.cc:156] Difference at 20: 3.71941, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743864908.659131   60461 buffer_comparator.cc:156] Difference at 21: 1.62894, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1743864908.659134   60461 buffer_comparator.cc:156] Difference at 22: 3.92903, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1743864908.659137   60461 buffer_comparator.cc:156] Difference at 23: 0.799771, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1743864908.659140   60461 buffer_comparator.cc:156] Difference at 24: 4.01028, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1743864908.659143   60461 buffer_comparator.cc:156] Difference at 25: -0.0661373, expected 36.0917</span></span>
<span class="line"><span>2025-04-05 14:55:08.659148: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.671753   60461 buffer_comparator.cc:156] Difference at 2: 38.3457, expected 34.2806</span></span>
<span class="line"><span>E0000 00:00:1743864908.671767   60461 buffer_comparator.cc:156] Difference at 6: 41.1731, expected 36.7103</span></span>
<span class="line"><span>E0000 00:00:1743864908.671770   60461 buffer_comparator.cc:156] Difference at 13: 31.5951, expected 35.7459</span></span>
<span class="line"><span>E0000 00:00:1743864908.671774   60461 buffer_comparator.cc:156] Difference at 17: 37.0853, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743864908.671777   60461 buffer_comparator.cc:156] Difference at 20: 37.9014, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743864908.671780   60461 buffer_comparator.cc:156] Difference at 45: 25.9105, expected 32.5352</span></span>
<span class="line"><span>E0000 00:00:1743864908.671785   60461 buffer_comparator.cc:156] Difference at 75: 24.7101, expected 28.3085</span></span>
<span class="line"><span>E0000 00:00:1743864908.671788   60461 buffer_comparator.cc:156] Difference at 77: 19.521, expected 27.4887</span></span>
<span class="line"><span>E0000 00:00:1743864908.671791   60461 buffer_comparator.cc:156] Difference at 94: 24.541, expected 28.5145</span></span>
<span class="line"><span>E0000 00:00:1743864908.671794   60461 buffer_comparator.cc:156] Difference at 101: 30.6158, expected 26.8436</span></span>
<span class="line"><span>2025-04-05 14:55:08.671799: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.683681   60461 buffer_comparator.cc:156] Difference at 16: -3.78258, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1743864908.683695   60461 buffer_comparator.cc:156] Difference at 17: -3.72381, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1743864908.683699   60461 buffer_comparator.cc:156] Difference at 18: -4.1847, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1743864908.683702   60461 buffer_comparator.cc:156] Difference at 19: -2.87769, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1743864908.683705   60461 buffer_comparator.cc:156] Difference at 20: -4.75822, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1743864908.683708   60461 buffer_comparator.cc:156] Difference at 21: -1.89209, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1743864908.683711   60461 buffer_comparator.cc:156] Difference at 22: -4.87747, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1743864908.683714   60461 buffer_comparator.cc:156] Difference at 23: -1.13161, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1743864908.683716   60461 buffer_comparator.cc:156] Difference at 24: -4.98629, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1743864908.683719   60461 buffer_comparator.cc:156] Difference at 25: 0.00145702, expected 11.3838</span></span>
<span class="line"><span>2025-04-05 14:55:08.683725: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.685853   60461 buffer_comparator.cc:156] Difference at 16: -3.78258, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1743864908.685863   60461 buffer_comparator.cc:156] Difference at 17: -3.72381, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1743864908.685867   60461 buffer_comparator.cc:156] Difference at 18: -4.1847, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1743864908.685870   60461 buffer_comparator.cc:156] Difference at 19: -2.87769, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1743864908.685873   60461 buffer_comparator.cc:156] Difference at 20: -4.75822, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1743864908.685876   60461 buffer_comparator.cc:156] Difference at 21: -1.89209, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1743864908.685879   60461 buffer_comparator.cc:156] Difference at 22: -4.87747, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1743864908.685881   60461 buffer_comparator.cc:156] Difference at 23: -1.13161, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1743864908.685884   60461 buffer_comparator.cc:156] Difference at 24: -4.98629, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1743864908.685887   60461 buffer_comparator.cc:156] Difference at 25: 0.00145702, expected 11.3838</span></span>
<span class="line"><span>2025-04-05 14:55:08.685892: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.688021   60461 buffer_comparator.cc:156] Difference at 32: -3.25765, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1743864908.688033   60461 buffer_comparator.cc:156] Difference at 33: 3.46504, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1743864908.688036   60461 buffer_comparator.cc:156] Difference at 34: -2.68665, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1743864908.688039   60461 buffer_comparator.cc:156] Difference at 35: 3.96033, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1743864908.688042   60461 buffer_comparator.cc:156] Difference at 36: -1.79596, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1743864908.688045   60461 buffer_comparator.cc:156] Difference at 37: 4.35839, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1743864908.688048   60461 buffer_comparator.cc:156] Difference at 38: -0.895182, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1743864908.688051   60461 buffer_comparator.cc:156] Difference at 39: 4.53373, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1743864908.688056   60461 buffer_comparator.cc:156] Difference at 40: -0.0940509, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1743864908.688059   60461 buffer_comparator.cc:156] Difference at 41: 4.304, expected 8.63119</span></span>
<span class="line"><span>2025-04-05 14:55:08.688064: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.690183   60461 buffer_comparator.cc:156] Difference at 32: -3.25765, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1743864908.690194   60461 buffer_comparator.cc:156] Difference at 33: 3.46504, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1743864908.690197   60461 buffer_comparator.cc:156] Difference at 34: -2.68665, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1743864908.690201   60461 buffer_comparator.cc:156] Difference at 35: 3.96033, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1743864908.690203   60461 buffer_comparator.cc:156] Difference at 36: -1.79596, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1743864908.690206   60461 buffer_comparator.cc:156] Difference at 37: 4.35839, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1743864908.690209   60461 buffer_comparator.cc:156] Difference at 38: -0.895182, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1743864908.690212   60461 buffer_comparator.cc:156] Difference at 39: 4.53373, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1743864908.690215   60461 buffer_comparator.cc:156] Difference at 40: -0.0940509, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1743864908.690218   60461 buffer_comparator.cc:156] Difference at 41: 4.304, expected 8.63119</span></span>
<span class="line"><span>2025-04-05 14:55:08.690223: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.692357   60461 buffer_comparator.cc:156] Difference at 64: 2.65118, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743864908.692368   60461 buffer_comparator.cc:156] Difference at 65: -2.59211, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743864908.692372   60461 buffer_comparator.cc:156] Difference at 66: 2.15995, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743864908.692375   60461 buffer_comparator.cc:156] Difference at 67: -3.05104, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743864908.692378   60461 buffer_comparator.cc:156] Difference at 68: 1.52622, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743864908.692381   60461 buffer_comparator.cc:156] Difference at 69: -3.35745, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743864908.692383   60461 buffer_comparator.cc:156] Difference at 70: 0.753905, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743864908.692387   60461 buffer_comparator.cc:156] Difference at 71: -3.69781, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743864908.692389   60461 buffer_comparator.cc:156] Difference at 72: 0.0912027, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743864908.692392   60461 buffer_comparator.cc:156] Difference at 73: -3.42791, expected 8.82565</span></span>
<span class="line"><span>2025-04-05 14:55:08.692397: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.694525   60461 buffer_comparator.cc:156] Difference at 64: 2.65118, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743864908.694536   60461 buffer_comparator.cc:156] Difference at 65: -2.59211, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743864908.694540   60461 buffer_comparator.cc:156] Difference at 66: 2.15995, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743864908.694543   60461 buffer_comparator.cc:156] Difference at 67: -3.05104, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743864908.694546   60461 buffer_comparator.cc:156] Difference at 68: 1.52622, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743864908.694549   60461 buffer_comparator.cc:156] Difference at 69: -3.35745, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743864908.694552   60461 buffer_comparator.cc:156] Difference at 70: 0.753905, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743864908.694555   60461 buffer_comparator.cc:156] Difference at 71: -3.69781, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743864908.694557   60461 buffer_comparator.cc:156] Difference at 72: 0.0912027, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743864908.694560   60461 buffer_comparator.cc:156] Difference at 73: -3.42791, expected 8.82565</span></span>
<span class="line"><span>2025-04-05 14:55:08.694567: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743864908.696697   60461 buffer_comparator.cc:156] Difference at 64: 2.65118, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743864908.696708   60461 buffer_comparator.cc:156] Difference at 65: -2.59211, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743864908.696711   60461 buffer_comparator.cc:156] Difference at 66: 2.15995, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743864908.696714   60461 buffer_comparator.cc:156] Difference at 67: -3.05104, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743864908.696717   60461 buffer_comparator.cc:156] Difference at 68: 1.52622, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743864908.696720   60461 buffer_comparator.cc:156] Difference at 69: -3.35745, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743864908.696723   60461 buffer_comparator.cc:156] Difference at 70: 0.753905, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743864908.696726   60461 buffer_comparator.cc:156] Difference at 71: -3.69781, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743864908.696729   60461 buffer_comparator.cc:156] Difference at 72: 0.0912027, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743864908.696732   60461 buffer_comparator.cc:156] Difference at 73: -3.42791, expected 8.82565</span></span>
<span class="line"><span>2025-04-05 14:55:08.696736: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.61968</span></span>
<span class="line"><span>Validation:	Loss 0.55674	Accuracy 0.53125</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.53674</span></span>
<span class="line"><span>Validation:	Loss 0.48199	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.46159</span></span>
<span class="line"><span>Validation:	Loss 0.40361	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.37379</span></span>
<span class="line"><span>Validation:	Loss 0.30891	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.27057</span></span>
<span class="line"><span>Validation:	Loss 0.20310	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.18326</span></span>
<span class="line"><span>Validation:	Loss 0.14140	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12735</span></span>
<span class="line"><span>Validation:	Loss 0.09593	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08783</span></span>
<span class="line"><span>Validation:	Loss 0.07059	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06661</span></span>
<span class="line"><span>Validation:	Loss 0.05547	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.05230</span></span>
<span class="line"><span>Validation:	Loss 0.04354	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.04072</span></span>
<span class="line"><span>Validation:	Loss 0.03442	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03231</span></span>
<span class="line"><span>Validation:	Loss 0.02763	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02605</span></span>
<span class="line"><span>Validation:	Loss 0.02262	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02148</span></span>
<span class="line"><span>Validation:	Loss 0.01899	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01814</span></span>
<span class="line"><span>Validation:	Loss 0.01635	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01567</span></span>
<span class="line"><span>Validation:	Loss 0.01433	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01381</span></span>
<span class="line"><span>Validation:	Loss 0.01275	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01233</span></span>
<span class="line"><span>Validation:	Loss 0.01145	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01111</span></span>
<span class="line"><span>Validation:	Loss 0.01037	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01007</span></span>
<span class="line"><span>Validation:	Loss 0.00943	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00914</span></span>
<span class="line"><span>Validation:	Loss 0.00859	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00831</span></span>
<span class="line"><span>Validation:	Loss 0.00781	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00754</span></span>
<span class="line"><span>Validation:	Loss 0.00710	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00685</span></span>
<span class="line"><span>Validation:	Loss 0.00647	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00625</span></span>
<span class="line"><span>Validation:	Loss 0.00594	Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-13/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.61025</span></span>
<span class="line"><span>Validation:	Loss 0.51165	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47878</span></span>
<span class="line"><span>Validation:	Loss 0.42448	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.40036</span></span>
<span class="line"><span>Validation:	Loss 0.35541	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.33625</span></span>
<span class="line"><span>Validation:	Loss 0.29383	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.27710</span></span>
<span class="line"><span>Validation:	Loss 0.23943	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.22312</span></span>
<span class="line"><span>Validation:	Loss 0.18986	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.17228</span></span>
<span class="line"><span>Validation:	Loss 0.13945	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.12189</span></span>
<span class="line"><span>Validation:	Loss 0.09457	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.08294</span></span>
<span class="line"><span>Validation:	Loss 0.06499	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.05836</span></span>
<span class="line"><span>Validation:	Loss 0.04882	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.04551</span></span>
<span class="line"><span>Validation:	Loss 0.03959	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03736</span></span>
<span class="line"><span>Validation:	Loss 0.03312	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.03135</span></span>
<span class="line"><span>Validation:	Loss 0.02820	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02686</span></span>
<span class="line"><span>Validation:	Loss 0.02429	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02323</span></span>
<span class="line"><span>Validation:	Loss 0.02111	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02030</span></span>
<span class="line"><span>Validation:	Loss 0.01849	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01786</span></span>
<span class="line"><span>Validation:	Loss 0.01628	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01568</span></span>
<span class="line"><span>Validation:	Loss 0.01434	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01372</span></span>
<span class="line"><span>Validation:	Loss 0.01259	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01199</span></span>
<span class="line"><span>Validation:	Loss 0.01087	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01029</span></span>
<span class="line"><span>Validation:	Loss 0.00928	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00879</span></span>
<span class="line"><span>Validation:	Loss 0.00810	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00778</span></span>
<span class="line"><span>Validation:	Loss 0.00734	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00711</span></span>
<span class="line"><span>Validation:	Loss 0.00679	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00661</span></span>
<span class="line"><span>Validation:	Loss 0.00635	Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,47)]))}const E=a(p,[["render",l]]);export{d as __pageData,E as default};
