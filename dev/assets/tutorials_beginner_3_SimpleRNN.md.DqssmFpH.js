import{_ as a,c as n,o as i,al as e}from"./chunks/framework.BCN3FD2k.js";const d=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"};function l(t,s,c,r,h,k){return i(),n("div",null,s[0]||(s[0]=[e(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><p>Note: If you wish to use AutoZygote() for automatic differentiation, add Zygote to your project dependencies and include <code>using Zygote</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    680.2 ms  ✓ NaNMath</span></span>
<span class="line"><span>   3719.1 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    858.7 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   1023.0 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5932.3 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9668.4 ms  ✓ Lux</span></span>
<span class="line"><span>  6 dependencies successfully precompiled in 21 seconds. 99 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>    528.7 ms  ✓ TranscodingStreams</span></span>
<span class="line"><span>  33412.8 ms  ✓ JLD2</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 34 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    411.3 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    451.3 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    400.4 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    800.3 ms  ✓ InitialValues</span></span>
<span class="line"><span>    438.3 ms  ✓ ShowCases</span></span>
<span class="line"><span>    370.1 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    389.5 ms  ✓ PtrArrays</span></span>
<span class="line"><span>    371.0 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    526.9 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    650.5 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>   1126.6 ms  ✓ Baselet</span></span>
<span class="line"><span>    443.0 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>    415.4 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    439.1 ms  ✓ NameResolution</span></span>
<span class="line"><span>    424.6 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>    501.9 ms  ✓ AliasTables</span></span>
<span class="line"><span>   2470.4 ms  ✓ Accessors</span></span>
<span class="line"><span>   2365.2 ms  ✓ StatsBase</span></span>
<span class="line"><span>    998.5 ms  ✓ Accessors → LinearAlgebraExt</span></span>
<span class="line"><span>    683.7 ms  ✓ Accessors → TestExt</span></span>
<span class="line"><span>    714.8 ms  ✓ Accessors → StaticArraysExt</span></span>
<span class="line"><span>    808.8 ms  ✓ BangBang</span></span>
<span class="line"><span>    573.5 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    727.7 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    514.5 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>   1101.7 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2867.8 ms  ✓ Transducers</span></span>
<span class="line"><span>    799.7 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>  18781.7 ms  ✓ MLStyle</span></span>
<span class="line"><span>   4894.3 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   5761.0 ms  ✓ FLoops</span></span>
<span class="line"><span>   6046.3 ms  ✓ MLUtils</span></span>
<span class="line"><span>  32 dependencies successfully precompiled in 37 seconds. 65 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1550.4 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2153.2 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 164 already precompiled.</span></span>
<span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>    469.2 ms  ✓ CodecZlib</span></span>
<span class="line"><span>  19277.8 ms  ✓ HTTP</span></span>
<span class="line"><span>  78260.1 ms  ✓ Reactant</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 98 seconds. 76 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibEnzymeExt...</span></span>
<span class="line"><span>   1241.4 ms  ✓ LuxLib → LuxLibEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 132 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   6868.1 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 148 already precompiled.</span></span>
<span class="line"><span>Precompiling HTTPExt...</span></span>
<span class="line"><span>   1913.1 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 43 already precompiled.</span></span>
<span class="line"><span>Precompiling OptimisersReactantExt...</span></span>
<span class="line"><span>  13719.1 ms  ✓ Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>  15881.5 ms  ✓ Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 16 seconds. 87 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  13082.0 ms  ✓ LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 84 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  12912.7 ms  ✓ MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 81 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  13325.3 ms  ✓ Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>  14046.5 ms  ✓ WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 14 seconds. 94 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantKernelAbstractionsExt...</span></span>
<span class="line"><span>  13460.5 ms  ✓ Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 91 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantArrayInterfaceExt...</span></span>
<span class="line"><span>  13397.8 ms  ✓ Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  13910.4 ms  ✓ Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  11519.2 ms  ✓ Lux → LuxReactantExt</span></span>
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
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-11/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>2025-04-07 01:57:04.729736: I external/xla/xla/service/service.cc:152] XLA service 0xab15e20 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-04-07 01:57:04.730079: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1743991024.732114 3752797 se_gpu_pjrt_client.cc:1040] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1743991024.732425 3752797 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743991024.732658 3752797 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743991024.748663 3752797 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>E0000 00:00:1743991084.318045 3752797 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743991084.318903 3752797 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743991084.318911 3752797 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743991084.318918 3752797 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743991084.318924 3752797 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743991084.318930 3752797 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743991084.318936 3752797 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743991084.318942 3752797 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743991084.318948 3752797 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743991084.318954 3752797 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-07 01:58:04.318971: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.322170 3752797 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743991084.322203 3752797 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743991084.322210 3752797 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743991084.322216 3752797 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743991084.322223 3752797 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743991084.322229 3752797 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743991084.322235 3752797 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743991084.322241 3752797 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743991084.322247 3752797 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743991084.322253 3752797 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-07 01:58:04.322265: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.325158 3752797 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743991084.325188 3752797 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743991084.325195 3752797 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743991084.325201 3752797 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743991084.325207 3752797 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743991084.325213 3752797 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743991084.325219 3752797 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743991084.325225 3752797 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743991084.325234 3752797 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743991084.325240 3752797 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-07 01:58:04.325251: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.328095 3752797 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743991084.328115 3752797 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743991084.328118 3752797 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743991084.328120 3752797 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743991084.328123 3752797 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743991084.328126 3752797 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743991084.328129 3752797 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743991084.328131 3752797 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743991084.328134 3752797 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743991084.328137 3752797 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-07 01:58:04.328142: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.330735 3752797 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743991084.330761 3752797 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743991084.330764 3752797 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743991084.330767 3752797 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743991084.330769 3752797 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743991084.330772 3752797 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743991084.330775 3752797 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743991084.330778 3752797 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743991084.330780 3752797 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743991084.330783 3752797 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-07 01:58:04.330790: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.333386 3752797 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743991084.333401 3752797 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743991084.333404 3752797 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743991084.333407 3752797 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743991084.333410 3752797 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743991084.333413 3752797 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743991084.333416 3752797 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743991084.333418 3752797 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743991084.333421 3752797 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743991084.333424 3752797 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-07 01:58:04.333429: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.336009 3752797 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743991084.336023 3752797 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743991084.336026 3752797 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743991084.336029 3752797 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743991084.336032 3752797 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743991084.336035 3752797 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743991084.336037 3752797 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743991084.336040 3752797 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743991084.336043 3752797 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743991084.336046 3752797 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-07 01:58:04.336050: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.338604 3752797 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743991084.338620 3752797 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743991084.338623 3752797 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743991084.338626 3752797 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743991084.338629 3752797 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743991084.338632 3752797 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743991084.338634 3752797 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743991084.338637 3752797 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743991084.338640 3752797 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743991084.338643 3752797 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-07 01:58:04.338647: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.341240 3752797 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743991084.341255 3752797 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743991084.341258 3752797 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743991084.341261 3752797 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743991084.341263 3752797 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743991084.341266 3752797 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743991084.341269 3752797 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743991084.341272 3752797 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743991084.341274 3752797 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743991084.341277 3752797 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-07 01:58:04.341282: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.343874 3752797 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743991084.343893 3752797 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743991084.343897 3752797 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743991084.343901 3752797 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743991084.343904 3752797 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743991084.343907 3752797 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743991084.343909 3752797 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743991084.343912 3752797 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743991084.343915 3752797 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743991084.343917 3752797 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-07 01:58:04.343923: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.346777 3752797 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743991084.346794 3752797 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743991084.346797 3752797 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743991084.346800 3752797 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743991084.346803 3752797 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743991084.346806 3752797 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743991084.346808 3752797 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743991084.346811 3752797 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743991084.346814 3752797 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743991084.346816 3752797 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-07 01:58:04.346821: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.349396 3752797 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743991084.349417 3752797 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743991084.349421 3752797 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743991084.349423 3752797 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743991084.349426 3752797 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743991084.349429 3752797 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743991084.349432 3752797 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743991084.349434 3752797 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743991084.349437 3752797 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743991084.349440 3752797 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-07 01:58:04.349445: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.352171 3752797 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743991084.352190 3752797 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743991084.352193 3752797 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743991084.352196 3752797 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743991084.352199 3752797 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743991084.352202 3752797 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743991084.352204 3752797 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743991084.352209 3752797 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743991084.352211 3752797 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743991084.352214 3752797 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-07 01:58:04.352219: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.354770 3752797 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743991084.354792 3752797 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743991084.354796 3752797 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743991084.354798 3752797 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743991084.354801 3752797 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743991084.354804 3752797 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743991084.354807 3752797 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743991084.354809 3752797 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743991084.354812 3752797 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743991084.354815 3752797 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-07 01:58:04.354820: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.357376 3752797 buffer_comparator.cc:156] Difference at 256: 0, expected 0.86224</span></span>
<span class="line"><span>E0000 00:00:1743991084.357399 3752797 buffer_comparator.cc:156] Difference at 257: 0, expected 0.686873</span></span>
<span class="line"><span>E0000 00:00:1743991084.357402 3752797 buffer_comparator.cc:156] Difference at 258: 0, expected 0.252371</span></span>
<span class="line"><span>E0000 00:00:1743991084.357405 3752797 buffer_comparator.cc:156] Difference at 259: 0, expected 0.335927</span></span>
<span class="line"><span>E0000 00:00:1743991084.357408 3752797 buffer_comparator.cc:156] Difference at 260: 0, expected 0.934139</span></span>
<span class="line"><span>E0000 00:00:1743991084.357411 3752797 buffer_comparator.cc:156] Difference at 261: 0, expected 0.274756</span></span>
<span class="line"><span>E0000 00:00:1743991084.357413 3752797 buffer_comparator.cc:156] Difference at 262: 0, expected 0.529946</span></span>
<span class="line"><span>E0000 00:00:1743991084.357416 3752797 buffer_comparator.cc:156] Difference at 263: 0, expected 0.542969</span></span>
<span class="line"><span>E0000 00:00:1743991084.357419 3752797 buffer_comparator.cc:156] Difference at 264: 0, expected 0.895372</span></span>
<span class="line"><span>E0000 00:00:1743991084.357422 3752797 buffer_comparator.cc:156] Difference at 265: 0, expected 0.895664</span></span>
<span class="line"><span>2025-04-07 01:58:04.357426: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.359997 3752797 buffer_comparator.cc:156] Difference at 256: 0, expected 0.86224</span></span>
<span class="line"><span>E0000 00:00:1743991084.360020 3752797 buffer_comparator.cc:156] Difference at 257: 0, expected 0.686873</span></span>
<span class="line"><span>E0000 00:00:1743991084.360023 3752797 buffer_comparator.cc:156] Difference at 258: 0, expected 0.252371</span></span>
<span class="line"><span>E0000 00:00:1743991084.360026 3752797 buffer_comparator.cc:156] Difference at 259: 0, expected 0.335927</span></span>
<span class="line"><span>E0000 00:00:1743991084.360029 3752797 buffer_comparator.cc:156] Difference at 260: 0, expected 0.934139</span></span>
<span class="line"><span>E0000 00:00:1743991084.360032 3752797 buffer_comparator.cc:156] Difference at 261: 0, expected 0.274756</span></span>
<span class="line"><span>E0000 00:00:1743991084.360035 3752797 buffer_comparator.cc:156] Difference at 262: 0, expected 0.529946</span></span>
<span class="line"><span>E0000 00:00:1743991084.360037 3752797 buffer_comparator.cc:156] Difference at 263: 0, expected 0.542969</span></span>
<span class="line"><span>E0000 00:00:1743991084.360040 3752797 buffer_comparator.cc:156] Difference at 264: 0, expected 0.895372</span></span>
<span class="line"><span>E0000 00:00:1743991084.360043 3752797 buffer_comparator.cc:156] Difference at 265: 0, expected 0.895664</span></span>
<span class="line"><span>2025-04-07 01:58:04.360048: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991084.363113 3752797 buffer_comparator.cc:156] Difference at 32: -nan, expected 1.62244</span></span>
<span class="line"><span>E0000 00:00:1743991084.363135 3752797 buffer_comparator.cc:156] Difference at 33: -nan, expected 1.87084</span></span>
<span class="line"><span>E0000 00:00:1743991084.363138 3752797 buffer_comparator.cc:156] Difference at 34: -nan, expected 1.07351</span></span>
<span class="line"><span>E0000 00:00:1743991084.363141 3752797 buffer_comparator.cc:156] Difference at 35: -nan, expected 2.92445</span></span>
<span class="line"><span>E0000 00:00:1743991084.363143 3752797 buffer_comparator.cc:156] Difference at 36: -nan, expected 1.98056</span></span>
<span class="line"><span>E0000 00:00:1743991084.363146 3752797 buffer_comparator.cc:156] Difference at 37: -nan, expected 2.07715</span></span>
<span class="line"><span>E0000 00:00:1743991084.363148 3752797 buffer_comparator.cc:156] Difference at 38: -nan, expected 1.56458</span></span>
<span class="line"><span>E0000 00:00:1743991084.363151 3752797 buffer_comparator.cc:156] Difference at 39: -nan, expected 2.27034</span></span>
<span class="line"><span>E0000 00:00:1743991084.363154 3752797 buffer_comparator.cc:156] Difference at 40: -nan, expected 2.31795</span></span>
<span class="line"><span>E0000 00:00:1743991084.363156 3752797 buffer_comparator.cc:156] Difference at 41: -nan, expected 2.55731</span></span>
<span class="line"><span>2025-04-07 01:58:04.363163: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.280951 3752797 buffer_comparator.cc:156] Difference at 16: -3.64346, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1743991123.281003 3752797 buffer_comparator.cc:156] Difference at 17: -3.6928, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1743991123.281007 3752797 buffer_comparator.cc:156] Difference at 18: -4.10613, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1743991123.281010 3752797 buffer_comparator.cc:156] Difference at 19: -2.88055, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1743991123.281014 3752797 buffer_comparator.cc:156] Difference at 20: -4.5112, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1743991123.281016 3752797 buffer_comparator.cc:156] Difference at 21: -1.76331, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1743991123.281019 3752797 buffer_comparator.cc:156] Difference at 22: -4.97911, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1743991123.281022 3752797 buffer_comparator.cc:156] Difference at 23: -1.24709, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1743991123.281025 3752797 buffer_comparator.cc:156] Difference at 24: -4.94225, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1743991123.281028 3752797 buffer_comparator.cc:156] Difference at 25: 0.10301, expected 11.3838</span></span>
<span class="line"><span>2025-04-07 01:58:43.281041: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.283223 3752797 buffer_comparator.cc:156] Difference at 16: -3.64346, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1743991123.283234 3752797 buffer_comparator.cc:156] Difference at 17: -3.6928, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1743991123.283237 3752797 buffer_comparator.cc:156] Difference at 18: -4.10613, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1743991123.283240 3752797 buffer_comparator.cc:156] Difference at 19: -2.88055, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1743991123.283243 3752797 buffer_comparator.cc:156] Difference at 20: -4.5112, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1743991123.283246 3752797 buffer_comparator.cc:156] Difference at 21: -1.76331, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1743991123.283249 3752797 buffer_comparator.cc:156] Difference at 22: -4.97911, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1743991123.283252 3752797 buffer_comparator.cc:156] Difference at 23: -1.24709, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1743991123.283254 3752797 buffer_comparator.cc:156] Difference at 24: -4.94225, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1743991123.283257 3752797 buffer_comparator.cc:156] Difference at 25: 0.10301, expected 11.3838</span></span>
<span class="line"><span>2025-04-07 01:58:43.283263: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.285436 3752797 buffer_comparator.cc:156] Difference at 32: -3.3021, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1743991123.285447 3752797 buffer_comparator.cc:156] Difference at 33: 3.51003, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1743991123.285452 3752797 buffer_comparator.cc:156] Difference at 34: -2.56291, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1743991123.285455 3752797 buffer_comparator.cc:156] Difference at 35: 4.03451, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1743991123.285458 3752797 buffer_comparator.cc:156] Difference at 36: -1.58068, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1743991123.285461 3752797 buffer_comparator.cc:156] Difference at 37: 4.28225, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1743991123.285464 3752797 buffer_comparator.cc:156] Difference at 38: -0.773736, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1743991123.285467 3752797 buffer_comparator.cc:156] Difference at 39: 4.54621, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1743991123.285470 3752797 buffer_comparator.cc:156] Difference at 40: -0.0555389, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1743991123.285473 3752797 buffer_comparator.cc:156] Difference at 41: 4.57256, expected 8.63119</span></span>
<span class="line"><span>2025-04-07 01:58:43.285478: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.287647 3752797 buffer_comparator.cc:156] Difference at 32: -3.3021, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1743991123.287658 3752797 buffer_comparator.cc:156] Difference at 33: 3.51003, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1743991123.287661 3752797 buffer_comparator.cc:156] Difference at 34: -2.56291, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1743991123.287665 3752797 buffer_comparator.cc:156] Difference at 35: 4.03451, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1743991123.287667 3752797 buffer_comparator.cc:156] Difference at 36: -1.58068, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1743991123.287670 3752797 buffer_comparator.cc:156] Difference at 37: 4.28225, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1743991123.287673 3752797 buffer_comparator.cc:156] Difference at 38: -0.773736, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1743991123.287676 3752797 buffer_comparator.cc:156] Difference at 39: 4.54621, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1743991123.287679 3752797 buffer_comparator.cc:156] Difference at 40: -0.0555389, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1743991123.287682 3752797 buffer_comparator.cc:156] Difference at 41: 4.57256, expected 8.63119</span></span>
<span class="line"><span>2025-04-07 01:58:43.287687: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.289855 3752797 buffer_comparator.cc:156] Difference at 64: 2.76847, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743991123.289868 3752797 buffer_comparator.cc:156] Difference at 65: -2.68847, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743991123.289872 3752797 buffer_comparator.cc:156] Difference at 66: 1.91627, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743991123.289875 3752797 buffer_comparator.cc:156] Difference at 67: -2.98143, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743991123.289877 3752797 buffer_comparator.cc:156] Difference at 68: 1.34898, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743991123.289880 3752797 buffer_comparator.cc:156] Difference at 69: -3.40215, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743991123.289883 3752797 buffer_comparator.cc:156] Difference at 70: 0.678766, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743991123.289886 3752797 buffer_comparator.cc:156] Difference at 71: -3.41605, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743991123.289889 3752797 buffer_comparator.cc:156] Difference at 72: -0.13915, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743991123.289892 3752797 buffer_comparator.cc:156] Difference at 73: -3.35055, expected 8.82565</span></span>
<span class="line"><span>2025-04-07 01:58:43.289897: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.292058 3752797 buffer_comparator.cc:156] Difference at 64: 2.76847, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743991123.292071 3752797 buffer_comparator.cc:156] Difference at 65: -2.68847, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743991123.292075 3752797 buffer_comparator.cc:156] Difference at 66: 1.91627, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743991123.292078 3752797 buffer_comparator.cc:156] Difference at 67: -2.98143, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743991123.292083 3752797 buffer_comparator.cc:156] Difference at 68: 1.34898, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743991123.292086 3752797 buffer_comparator.cc:156] Difference at 69: -3.40215, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743991123.292088 3752797 buffer_comparator.cc:156] Difference at 70: 0.678766, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743991123.292091 3752797 buffer_comparator.cc:156] Difference at 71: -3.41605, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743991123.292094 3752797 buffer_comparator.cc:156] Difference at 72: -0.13915, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743991123.292097 3752797 buffer_comparator.cc:156] Difference at 73: -3.35055, expected 8.82565</span></span>
<span class="line"><span>2025-04-07 01:58:43.292102: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.294260 3752797 buffer_comparator.cc:156] Difference at 64: 2.76847, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743991123.294272 3752797 buffer_comparator.cc:156] Difference at 65: -2.68847, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743991123.294276 3752797 buffer_comparator.cc:156] Difference at 66: 1.91627, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743991123.294279 3752797 buffer_comparator.cc:156] Difference at 67: -2.98143, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743991123.294281 3752797 buffer_comparator.cc:156] Difference at 68: 1.34898, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743991123.294284 3752797 buffer_comparator.cc:156] Difference at 69: -3.40215, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743991123.294287 3752797 buffer_comparator.cc:156] Difference at 70: 0.678766, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743991123.294290 3752797 buffer_comparator.cc:156] Difference at 71: -3.41605, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743991123.294293 3752797 buffer_comparator.cc:156] Difference at 72: -0.13915, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743991123.294296 3752797 buffer_comparator.cc:156] Difference at 73: -3.35055, expected 8.82565</span></span>
<span class="line"><span>2025-04-07 01:58:43.294300: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.303723 3752797 buffer_comparator.cc:156] Difference at 16: 6.73254, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1743991123.303764 3752797 buffer_comparator.cc:156] Difference at 17: 9.2143, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743991123.303767 3752797 buffer_comparator.cc:156] Difference at 18: 6.85711, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1743991123.303770 3752797 buffer_comparator.cc:156] Difference at 19: 7.1738, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1743991123.303773 3752797 buffer_comparator.cc:156] Difference at 20: 6.9796, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743991123.303776 3752797 buffer_comparator.cc:156] Difference at 21: 7.31867, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1743991123.303779 3752797 buffer_comparator.cc:156] Difference at 22: 8.70739, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1743991123.303782 3752797 buffer_comparator.cc:156] Difference at 23: 7.19731, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1743991123.303785 3752797 buffer_comparator.cc:156] Difference at 24: 8.21217, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1743991123.303788 3752797 buffer_comparator.cc:156] Difference at 25: 9.32022, expected 36.0917</span></span>
<span class="line"><span>2025-04-07 01:58:43.303796: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.305948 3752797 buffer_comparator.cc:156] Difference at 16: 6.73254, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1743991123.305961 3752797 buffer_comparator.cc:156] Difference at 17: 9.2143, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743991123.305964 3752797 buffer_comparator.cc:156] Difference at 18: 6.85711, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1743991123.305967 3752797 buffer_comparator.cc:156] Difference at 19: 7.1738, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1743991123.305970 3752797 buffer_comparator.cc:156] Difference at 20: 6.9796, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743991123.305973 3752797 buffer_comparator.cc:156] Difference at 21: 7.31867, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1743991123.305977 3752797 buffer_comparator.cc:156] Difference at 22: 8.70739, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1743991123.305980 3752797 buffer_comparator.cc:156] Difference at 23: 7.19731, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1743991123.305983 3752797 buffer_comparator.cc:156] Difference at 24: 8.21217, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1743991123.305986 3752797 buffer_comparator.cc:156] Difference at 25: 9.32022, expected 36.0917</span></span>
<span class="line"><span>2025-04-07 01:58:43.305991: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.318775 3752797 buffer_comparator.cc:156] Difference at 2: 38.3457, expected 34.2806</span></span>
<span class="line"><span>E0000 00:00:1743991123.318813 3752797 buffer_comparator.cc:156] Difference at 6: 41.1731, expected 36.7103</span></span>
<span class="line"><span>E0000 00:00:1743991123.318817 3752797 buffer_comparator.cc:156] Difference at 13: 31.5951, expected 35.7459</span></span>
<span class="line"><span>E0000 00:00:1743991123.318820 3752797 buffer_comparator.cc:156] Difference at 17: 37.0853, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743991123.318823 3752797 buffer_comparator.cc:156] Difference at 20: 37.9014, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743991123.318827 3752797 buffer_comparator.cc:156] Difference at 45: 25.9105, expected 32.5352</span></span>
<span class="line"><span>E0000 00:00:1743991123.318830 3752797 buffer_comparator.cc:156] Difference at 75: 24.7101, expected 28.3085</span></span>
<span class="line"><span>E0000 00:00:1743991123.318833 3752797 buffer_comparator.cc:156] Difference at 77: 19.521, expected 27.4887</span></span>
<span class="line"><span>E0000 00:00:1743991123.318836 3752797 buffer_comparator.cc:156] Difference at 94: 24.541, expected 28.5145</span></span>
<span class="line"><span>E0000 00:00:1743991123.318839 3752797 buffer_comparator.cc:156] Difference at 101: 30.6158, expected 26.8436</span></span>
<span class="line"><span>2025-04-07 01:58:43.318847: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.328225 3752797 buffer_comparator.cc:156] Difference at 19: 38.0823, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1743991123.328238 3752797 buffer_comparator.cc:156] Difference at 21: 37.818, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1743991123.328242 3752797 buffer_comparator.cc:156] Difference at 22: 35.4896, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1743991123.328245 3752797 buffer_comparator.cc:156] Difference at 24: 37.6513, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1743991123.328248 3752797 buffer_comparator.cc:156] Difference at 28: 35.9018, expected 31.9598</span></span>
<span class="line"><span>E0000 00:00:1743991123.328251 3752797 buffer_comparator.cc:156] Difference at 50: 31.5173, expected 38.8675</span></span>
<span class="line"><span>E0000 00:00:1743991123.328254 3752797 buffer_comparator.cc:156] Difference at 59: 32.2662, expected 36.5446</span></span>
<span class="line"><span>E0000 00:00:1743991123.328257 3752797 buffer_comparator.cc:156] Difference at 62: 33.1071, expected 39.0134</span></span>
<span class="line"><span>E0000 00:00:1743991123.328260 3752797 buffer_comparator.cc:156] Difference at 63: 31.6033, expected 37.4517</span></span>
<span class="line"><span>2025-04-07 01:58:43.328265: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.330421 3752797 buffer_comparator.cc:156] Difference at 19: 38.0823, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1743991123.330434 3752797 buffer_comparator.cc:156] Difference at 21: 37.818, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1743991123.330437 3752797 buffer_comparator.cc:156] Difference at 22: 35.4896, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1743991123.330440 3752797 buffer_comparator.cc:156] Difference at 24: 37.6513, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1743991123.330443 3752797 buffer_comparator.cc:156] Difference at 28: 35.9018, expected 31.9598</span></span>
<span class="line"><span>E0000 00:00:1743991123.330446 3752797 buffer_comparator.cc:156] Difference at 50: 31.5173, expected 38.8675</span></span>
<span class="line"><span>E0000 00:00:1743991123.330450 3752797 buffer_comparator.cc:156] Difference at 59: 32.2662, expected 36.5446</span></span>
<span class="line"><span>E0000 00:00:1743991123.330453 3752797 buffer_comparator.cc:156] Difference at 62: 33.1071, expected 39.0134</span></span>
<span class="line"><span>E0000 00:00:1743991123.330456 3752797 buffer_comparator.cc:156] Difference at 63: 31.6033, expected 37.4517</span></span>
<span class="line"><span>2025-04-07 01:58:43.330462: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743991123.343160 3752797 buffer_comparator.cc:156] Difference at 2: 37.361, expected 33.2434</span></span>
<span class="line"><span>E0000 00:00:1743991123.343173 3752797 buffer_comparator.cc:156] Difference at 8: 32.9182, expected 29.0801</span></span>
<span class="line"><span>E0000 00:00:1743991123.343176 3752797 buffer_comparator.cc:156] Difference at 11: 35.3152, expected 30.7625</span></span>
<span class="line"><span>E0000 00:00:1743991123.343179 3752797 buffer_comparator.cc:156] Difference at 12: 39.5233, expected 34.3637</span></span>
<span class="line"><span>E0000 00:00:1743991123.343182 3752797 buffer_comparator.cc:156] Difference at 20: 38.835, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1743991123.343185 3752797 buffer_comparator.cc:156] Difference at 23: 37.0226, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1743991123.343188 3752797 buffer_comparator.cc:156] Difference at 26: 39.1599, expected 32.4927</span></span>
<span class="line"><span>E0000 00:00:1743991123.343192 3752797 buffer_comparator.cc:156] Difference at 51: 26.8307, expected 33.7879</span></span>
<span class="line"><span>2025-04-07 01:58:43.343197: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.99490</span></span>
<span class="line"><span>Validation:	Loss 0.87817	Accuracy 0.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.82729</span></span>
<span class="line"><span>Validation:	Loss 0.76100	Accuracy 0.50000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.72064</span></span>
<span class="line"><span>Validation:	Loss 0.66514	Accuracy 0.50000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.62898</span></span>
<span class="line"><span>Validation:	Loss 0.58230	Accuracy 0.82031</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.55090</span></span>
<span class="line"><span>Validation:	Loss 0.51343	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.49045</span></span>
<span class="line"><span>Validation:	Loss 0.45703	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.43174</span></span>
<span class="line"><span>Validation:	Loss 0.39954	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.37304</span></span>
<span class="line"><span>Validation:	Loss 0.34015	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.32047</span></span>
<span class="line"><span>Validation:	Loss 0.29535	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.27577</span></span>
<span class="line"><span>Validation:	Loss 0.25348	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.23811</span></span>
<span class="line"><span>Validation:	Loss 0.21354	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.19657</span></span>
<span class="line"><span>Validation:	Loss 0.17366	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.15817</span></span>
<span class="line"><span>Validation:	Loss 0.13802	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.12495</span></span>
<span class="line"><span>Validation:	Loss 0.10930	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.09985</span></span>
<span class="line"><span>Validation:	Loss 0.08733	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.08003</span></span>
<span class="line"><span>Validation:	Loss 0.07077	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.06448</span></span>
<span class="line"><span>Validation:	Loss 0.05662	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.05096</span></span>
<span class="line"><span>Validation:	Loss 0.04375	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.03965</span></span>
<span class="line"><span>Validation:	Loss 0.03469	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.03247</span></span>
<span class="line"><span>Validation:	Loss 0.02958	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.02813</span></span>
<span class="line"><span>Validation:	Loss 0.02627	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.02515</span></span>
<span class="line"><span>Validation:	Loss 0.02381	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.02299</span></span>
<span class="line"><span>Validation:	Loss 0.02184	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.02110</span></span>
<span class="line"><span>Validation:	Loss 0.02020	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.01956</span></span>
<span class="line"><span>Validation:	Loss 0.01879	Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-11/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.79457</span></span>
<span class="line"><span>Validation:	Loss 0.66628	Accuracy 0.48438</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.61656</span></span>
<span class="line"><span>Validation:	Loss 0.56646	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.53688</span></span>
<span class="line"><span>Validation:	Loss 0.49988	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.47171</span></span>
<span class="line"><span>Validation:	Loss 0.43655	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.40865</span></span>
<span class="line"><span>Validation:	Loss 0.37074	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.34266</span></span>
<span class="line"><span>Validation:	Loss 0.30445	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.27837</span></span>
<span class="line"><span>Validation:	Loss 0.24427	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.22251</span></span>
<span class="line"><span>Validation:	Loss 0.19388	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.17647</span></span>
<span class="line"><span>Validation:	Loss 0.15339	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.14038</span></span>
<span class="line"><span>Validation:	Loss 0.12386	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.11518</span></span>
<span class="line"><span>Validation:	Loss 0.10355	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.09704</span></span>
<span class="line"><span>Validation:	Loss 0.08802	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.08280</span></span>
<span class="line"><span>Validation:	Loss 0.07572	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.07156</span></span>
<span class="line"><span>Validation:	Loss 0.06572	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.06231</span></span>
<span class="line"><span>Validation:	Loss 0.05741	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.05456</span></span>
<span class="line"><span>Validation:	Loss 0.05035	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.04789</span></span>
<span class="line"><span>Validation:	Loss 0.04426	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.04222</span></span>
<span class="line"><span>Validation:	Loss 0.03900	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.03719</span></span>
<span class="line"><span>Validation:	Loss 0.03452	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.03315</span></span>
<span class="line"><span>Validation:	Loss 0.03073	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.02949</span></span>
<span class="line"><span>Validation:	Loss 0.02751	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.02649</span></span>
<span class="line"><span>Validation:	Loss 0.02476	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.02386</span></span>
<span class="line"><span>Validation:	Loss 0.02236	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.02159</span></span>
<span class="line"><span>Validation:	Loss 0.02020	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.01950</span></span>
<span class="line"><span>Validation:	Loss 0.01835	Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,47)]))}const f=a(p,[["render",l]]);export{d as __pageData,f as default};
