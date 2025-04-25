import{_ as a,c as n,o as e,al as p}from"./chunks/framework.BCN3FD2k.js";const f=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),i={name:"tutorials/beginner/3_SimpleRNN.md"};function c(t,s,l,r,d,o){return e(),n("div",null,s[0]||(s[0]=[p(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><p>Note: If you wish to use AutoZygote() for automatic differentiation, add Zygote to your project dependencies and include <code>using Zygote</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    316.0 ms  ✓ SIMDTypes</span></span>
<span class="line"><span>    314.0 ms  ✓ FastClosures</span></span>
<span class="line"><span>    425.1 ms  ✓ ManualMemory</span></span>
<span class="line"><span>    389.7 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>    474.6 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>   1011.2 ms  ✓ CPUSummary</span></span>
<span class="line"><span>   1213.8 ms  ✓ LuxCore</span></span>
<span class="line"><span>    892.9 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    583.0 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    809.7 ms  ✓ ThreadingUtilities</span></span>
<span class="line"><span>    425.5 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    437.0 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    601.5 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>    442.5 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    458.8 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    640.0 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>    871.3 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>    704.9 ms  ✓ Polyester</span></span>
<span class="line"><span>   5284.6 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9066.5 ms  ✓ Lux</span></span>
<span class="line"><span>  20 dependencies successfully precompiled in 20 seconds. 85 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>   4115.9 ms  ✓ FileIO</span></span>
<span class="line"><span>  33097.7 ms  ✓ JLD2</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 37 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    339.9 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    394.8 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    440.6 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    317.3 ms  ✓ PtrArrays</span></span>
<span class="line"><span>    456.0 ms  ✓ Missings</span></span>
<span class="line"><span>    519.4 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    996.5 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>   1158.5 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    596.7 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>    412.9 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>   1670.6 ms  ✓ DataStructures</span></span>
<span class="line"><span>    400.9 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    404.1 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>    480.7 ms  ✓ AliasTables</span></span>
<span class="line"><span>    534.1 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>   1003.4 ms  ✓ MLCore</span></span>
<span class="line"><span>   2374.7 ms  ✓ Accessors</span></span>
<span class="line"><span>   2259.7 ms  ✓ StatsBase</span></span>
<span class="line"><span>    630.9 ms  ✓ Accessors → TestExt</span></span>
<span class="line"><span>    807.4 ms  ✓ Accessors → LinearAlgebraExt</span></span>
<span class="line"><span>    686.3 ms  ✓ Accessors → StaticArraysExt</span></span>
<span class="line"><span>    739.7 ms  ✓ BangBang</span></span>
<span class="line"><span>    479.8 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    532.8 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    689.7 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    900.3 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2649.6 ms  ✓ Transducers</span></span>
<span class="line"><span>    641.0 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   4947.6 ms  ✓ FLoops</span></span>
<span class="line"><span>   5707.7 ms  ✓ MLUtils</span></span>
<span class="line"><span>  30 dependencies successfully precompiled in 22 seconds. 67 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1448.4 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   1968.2 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 164 already precompiled.</span></span>
<span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>    477.0 ms  ✓ CodecZlib</span></span>
<span class="line"><span>    487.4 ms  ✓ EnumX</span></span>
<span class="line"><span>    743.8 ms  ✓ ConcurrentUtilities</span></span>
<span class="line"><span>    510.0 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>    619.6 ms  ✓ ReactantCore</span></span>
<span class="line"><span>  18432.2 ms  ✓ HTTP</span></span>
<span class="line"><span>  91022.3 ms  ✓ Reactant</span></span>
<span class="line"><span>  7 dependencies successfully precompiled in 111 seconds. 73 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibEnzymeExt...</span></span>
<span class="line"><span>   1342.9 ms  ✓ LuxLib → LuxLibEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 133 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   7666.1 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 8 seconds. 149 already precompiled.</span></span>
<span class="line"><span>Precompiling HTTPExt...</span></span>
<span class="line"><span>   1738.6 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 43 already precompiled.</span></span>
<span class="line"><span>Precompiling OptimisersReactantExt...</span></span>
<span class="line"><span>  21090.6 ms  ✓ Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>  24904.8 ms  ✓ Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 25 seconds. 88 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  20963.1 ms  ✓ LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 21 seconds. 85 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  21162.0 ms  ✓ MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 22 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  21954.2 ms  ✓ Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>  22087.7 ms  ✓ WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 22 seconds. 95 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantKernelAbstractionsExt...</span></span>
<span class="line"><span>  21346.0 ms  ✓ Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 22 seconds. 92 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantArrayInterfaceExt...</span></span>
<span class="line"><span>  21129.1 ms  ✓ Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 21 seconds. 83 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  21351.8 ms  ✓ Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 22 seconds. 103 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  11830.3 ms  ✓ Lux → LuxReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 12 seconds. 180 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span>2025-04-25 01:17:48.272474: I external/xla/xla/service/service.cc:152] XLA service 0x13661a50 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-04-25 01:17:48.272845: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1745543868.274491 1243516 se_gpu_pjrt_client.cc:999] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1745543868.274817 1243516 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1745543868.275204 1243516 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1745543868.292780 1243516 cuda_dnn.cc:527] Loaded cuDNN version 90400</span></span>
<span class="line"><span>E0000 00:00:1745543926.099366 1243516 buffer_comparator.cc:145] Difference at 32: 0, expected 1.62244</span></span>
<span class="line"><span>E0000 00:00:1745543926.100181 1243516 buffer_comparator.cc:145] Difference at 33: 0, expected 1.87084</span></span>
<span class="line"><span>E0000 00:00:1745543926.100190 1243516 buffer_comparator.cc:145] Difference at 34: 0, expected 1.07351</span></span>
<span class="line"><span>E0000 00:00:1745543926.100196 1243516 buffer_comparator.cc:145] Difference at 35: 0, expected 2.92445</span></span>
<span class="line"><span>E0000 00:00:1745543926.100202 1243516 buffer_comparator.cc:145] Difference at 36: 0, expected 1.98056</span></span>
<span class="line"><span>E0000 00:00:1745543926.100209 1243516 buffer_comparator.cc:145] Difference at 37: 0, expected 2.07715</span></span>
<span class="line"><span>E0000 00:00:1745543926.100215 1243516 buffer_comparator.cc:145] Difference at 38: 0, expected 1.56458</span></span>
<span class="line"><span>E0000 00:00:1745543926.100221 1243516 buffer_comparator.cc:145] Difference at 39: 0, expected 2.27034</span></span>
<span class="line"><span>E0000 00:00:1745543926.100227 1243516 buffer_comparator.cc:145] Difference at 40: 0, expected 2.31795</span></span>
<span class="line"><span>E0000 00:00:1745543926.100233 1243516 buffer_comparator.cc:145] Difference at 41: 0, expected 2.55731</span></span>
<span class="line"><span>2025-04-25 01:18:46.100249: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.104915 1243516 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1745543926.104947 1243516 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1745543926.104954 1243516 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1745543926.104961 1243516 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1745543926.104967 1243516 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1745543926.104973 1243516 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1745543926.104980 1243516 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1745543926.104986 1243516 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1745543926.104992 1243516 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1745543926.104998 1243516 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-25 01:18:46.105010: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.107779 1243516 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1745543926.107794 1243516 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1745543926.107797 1243516 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1745543926.107799 1243516 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1745543926.107802 1243516 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1745543926.107805 1243516 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1745543926.107808 1243516 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1745543926.107810 1243516 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1745543926.107815 1243516 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1745543926.107818 1243516 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-25 01:18:46.107824: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.110379 1243516 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1745543926.110392 1243516 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1745543926.110395 1243516 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1745543926.110398 1243516 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1745543926.110400 1243516 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1745543926.110403 1243516 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1745543926.110406 1243516 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1745543926.110409 1243516 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1745543926.110411 1243516 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1745543926.110414 1243516 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-25 01:18:46.110419: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.112943 1243516 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1745543926.112954 1243516 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1745543926.112957 1243516 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1745543926.112960 1243516 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1745543926.112963 1243516 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1745543926.112966 1243516 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1745543926.112968 1243516 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1745543926.112971 1243516 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1745543926.112974 1243516 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1745543926.112977 1243516 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-25 01:18:46.112982: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.115496 1243516 buffer_comparator.cc:145] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1745543926.115508 1243516 buffer_comparator.cc:145] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1745543926.115512 1243516 buffer_comparator.cc:145] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1745543926.115515 1243516 buffer_comparator.cc:145] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1745543926.115518 1243516 buffer_comparator.cc:145] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1745543926.115520 1243516 buffer_comparator.cc:145] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1745543926.115523 1243516 buffer_comparator.cc:145] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1745543926.115526 1243516 buffer_comparator.cc:145] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1745543926.115529 1243516 buffer_comparator.cc:145] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1745543926.115531 1243516 buffer_comparator.cc:145] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-25 01:18:46.115536: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.118088 1243516 buffer_comparator.cc:145] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1745543926.118110 1243516 buffer_comparator.cc:145] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1745543926.118113 1243516 buffer_comparator.cc:145] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1745543926.118116 1243516 buffer_comparator.cc:145] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1745543926.118119 1243516 buffer_comparator.cc:145] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1745543926.118122 1243516 buffer_comparator.cc:145] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1745543926.118125 1243516 buffer_comparator.cc:145] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1745543926.118127 1243516 buffer_comparator.cc:145] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1745543926.118130 1243516 buffer_comparator.cc:145] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1745543926.118133 1243516 buffer_comparator.cc:145] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-25 01:18:46.118138: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.120690 1243516 buffer_comparator.cc:145] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1745543926.120705 1243516 buffer_comparator.cc:145] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1745543926.120708 1243516 buffer_comparator.cc:145] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1745543926.120711 1243516 buffer_comparator.cc:145] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1745543926.120714 1243516 buffer_comparator.cc:145] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1745543926.120717 1243516 buffer_comparator.cc:145] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1745543926.120719 1243516 buffer_comparator.cc:145] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1745543926.120722 1243516 buffer_comparator.cc:145] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1745543926.120725 1243516 buffer_comparator.cc:145] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1745543926.120728 1243516 buffer_comparator.cc:145] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-25 01:18:46.120733: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.123262 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.123275 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.123278 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.123281 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.123284 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.123286 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.123289 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.123292 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.123295 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.123298 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.123302: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.125816 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.125828 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.125831 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.125836 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.125839 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.125842 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.125845 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.125848 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.125850 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.125853 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.125858: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.128386 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.128398 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.128401 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.128404 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.128407 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.128410 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.128412 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.128415 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.128418 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.128421 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.128426: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.130939 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.130950 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.130953 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.130956 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.130959 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.130962 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.130964 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.130967 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.130970 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.130973 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.130977: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.133499 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.133512 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.133515 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.133518 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.133521 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.133524 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.133526 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.133547 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.133550 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.133552 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.133557: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.136127 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.136143 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.136146 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.136148 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.136151 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.136154 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.136157 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.136160 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.136162 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.136165 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.136171: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.138984 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.139000 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.139004 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.139006 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.139009 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.139012 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.139015 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.139018 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.139020 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.139023 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.139028: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.141663 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.141676 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.141679 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.141681 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.141684 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.141687 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.141690 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.141693 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.141695 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.141698 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.141703: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.144356 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.144369 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.144372 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.144375 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.144378 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.144380 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.144383 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.144386 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.144389 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.144392 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.144396: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.146991 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.147011 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.147014 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.147017 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.147020 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.147023 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.147026 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.147028 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.147031 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.147034 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.147040: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.149650 1243516 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1745543926.149668 1243516 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1745543926.149671 1243516 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1745543926.149674 1243516 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1745543926.149677 1243516 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1745543926.149680 1243516 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1745543926.149682 1243516 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1745543926.149685 1243516 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1745543926.149688 1243516 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1745543926.149691 1243516 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-25 01:18:46.149697: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.152347 1243516 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1745543926.152359 1243516 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1745543926.152362 1243516 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1745543926.152369 1243516 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1745543926.152371 1243516 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1745543926.152375 1243516 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1745543926.152377 1243516 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1745543926.152380 1243516 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1745543926.152383 1243516 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1745543926.152386 1243516 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-25 01:18:46.152390: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.154961 1243516 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1745543926.154972 1243516 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1745543926.154976 1243516 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1745543926.154978 1243516 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1745543926.154981 1243516 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1745543926.154984 1243516 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1745543926.154987 1243516 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1745543926.154990 1243516 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1745543926.154992 1243516 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1745543926.154995 1243516 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-25 01:18:46.155000: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.157562 1243516 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1745543926.157576 1243516 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1745543926.157579 1243516 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1745543926.157582 1243516 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1745543926.157585 1243516 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1745543926.157588 1243516 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1745543926.157590 1243516 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1745543926.157593 1243516 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1745543926.157596 1243516 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1745543926.157599 1243516 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-25 01:18:46.157604: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.160199 1243516 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1745543926.160211 1243516 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1745543926.160215 1243516 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1745543926.160217 1243516 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1745543926.160220 1243516 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1745543926.160223 1243516 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1745543926.160226 1243516 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1745543926.160231 1243516 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1745543926.160234 1243516 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1745543926.160236 1243516 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-25 01:18:46.160241: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543926.162801 1243516 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1745543926.162813 1243516 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1745543926.162816 1243516 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1745543926.162819 1243516 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1745543926.162822 1243516 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1745543926.162824 1243516 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1745543926.162827 1243516 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1745543926.162830 1243516 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1745543926.162833 1243516 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1745543926.162835 1243516 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-25 01:18:46.162840: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.072596 1243516 buffer_comparator.cc:145] Difference at 16: 6.09308, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1745543971.072651 1243516 buffer_comparator.cc:145] Difference at 17: 0.116254, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1745543971.072660 1243516 buffer_comparator.cc:145] Difference at 18: 5.86847, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1745543971.072667 1243516 buffer_comparator.cc:145] Difference at 19: -1.02751, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1745543971.072674 1243516 buffer_comparator.cc:145] Difference at 20: 5.44934, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1745543971.072680 1243516 buffer_comparator.cc:145] Difference at 21: -2.19791, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1745543971.072687 1243516 buffer_comparator.cc:145] Difference at 22: 4.85714, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1745543971.072694 1243516 buffer_comparator.cc:145] Difference at 23: -3.20712, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1745543971.072700 1243516 buffer_comparator.cc:145] Difference at 24: 4.00776, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1745543971.072707 1243516 buffer_comparator.cc:145] Difference at 25: -4.11396, expected 36.4575</span></span>
<span class="line"><span>2025-04-25 01:19:31.072720: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.075143 1243516 buffer_comparator.cc:145] Difference at 16: 6.09308, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1745543971.075168 1243516 buffer_comparator.cc:145] Difference at 17: 0.116254, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1745543971.075176 1243516 buffer_comparator.cc:145] Difference at 18: 5.86847, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1745543971.075183 1243516 buffer_comparator.cc:145] Difference at 19: -1.02751, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1745543971.075190 1243516 buffer_comparator.cc:145] Difference at 20: 5.44934, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1745543971.075196 1243516 buffer_comparator.cc:145] Difference at 21: -2.19791, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1745543971.075203 1243516 buffer_comparator.cc:145] Difference at 22: 4.85714, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1745543971.075210 1243516 buffer_comparator.cc:145] Difference at 23: -3.20712, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1745543971.075216 1243516 buffer_comparator.cc:145] Difference at 24: 4.00776, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1745543971.075223 1243516 buffer_comparator.cc:145] Difference at 25: -4.11396, expected 36.4575</span></span>
<span class="line"><span>2025-04-25 01:19:31.075236: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.077660 1243516 buffer_comparator.cc:145] Difference at 16: 6.09308, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1745543971.077683 1243516 buffer_comparator.cc:145] Difference at 17: 0.116254, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1745543971.077691 1243516 buffer_comparator.cc:145] Difference at 18: 5.86847, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1745543971.077698 1243516 buffer_comparator.cc:145] Difference at 19: -1.02751, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1745543971.077705 1243516 buffer_comparator.cc:145] Difference at 20: 5.44934, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1745543971.077711 1243516 buffer_comparator.cc:145] Difference at 21: -2.19791, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1745543971.077718 1243516 buffer_comparator.cc:145] Difference at 22: 4.85714, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1745543971.077724 1243516 buffer_comparator.cc:145] Difference at 23: -3.20712, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1745543971.077731 1243516 buffer_comparator.cc:145] Difference at 24: 4.00776, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1745543971.077737 1243516 buffer_comparator.cc:145] Difference at 25: -4.11396, expected 36.4575</span></span>
<span class="line"><span>2025-04-25 01:19:31.077747: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.080171 1243516 buffer_comparator.cc:145] Difference at 16: 6.09308, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1745543971.080195 1243516 buffer_comparator.cc:145] Difference at 17: 0.116254, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1745543971.080203 1243516 buffer_comparator.cc:145] Difference at 18: 5.86847, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1745543971.080210 1243516 buffer_comparator.cc:145] Difference at 19: -1.02751, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1745543971.080216 1243516 buffer_comparator.cc:145] Difference at 20: 5.44934, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1745543971.080223 1243516 buffer_comparator.cc:145] Difference at 21: -2.19791, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1745543971.080229 1243516 buffer_comparator.cc:145] Difference at 22: 4.85714, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1745543971.080236 1243516 buffer_comparator.cc:145] Difference at 23: -3.20712, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1745543971.080243 1243516 buffer_comparator.cc:145] Difference at 24: 4.00776, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1745543971.080249 1243516 buffer_comparator.cc:145] Difference at 25: -4.11396, expected 36.4575</span></span>
<span class="line"><span>2025-04-25 01:19:31.080259: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.082643 1243516 buffer_comparator.cc:145] Difference at 16: 6.09308, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1745543971.082655 1243516 buffer_comparator.cc:145] Difference at 17: 0.116254, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1745543971.082658 1243516 buffer_comparator.cc:145] Difference at 18: 5.86847, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1745543971.082661 1243516 buffer_comparator.cc:145] Difference at 19: -1.02751, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1745543971.082664 1243516 buffer_comparator.cc:145] Difference at 20: 5.44934, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1745543971.082667 1243516 buffer_comparator.cc:145] Difference at 21: -2.19791, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1745543971.082670 1243516 buffer_comparator.cc:145] Difference at 22: 4.85714, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1745543971.082673 1243516 buffer_comparator.cc:145] Difference at 23: -3.20712, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1745543971.082676 1243516 buffer_comparator.cc:145] Difference at 24: 4.00776, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1745543971.082679 1243516 buffer_comparator.cc:145] Difference at 25: -4.11396, expected 36.4575</span></span>
<span class="line"><span>2025-04-25 01:19:31.082683: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.105178 1243516 buffer_comparator.cc:145] Difference at 16: -2.9902, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1745543971.105222 1243516 buffer_comparator.cc:145] Difference at 17: -3.08035, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1745543971.105226 1243516 buffer_comparator.cc:145] Difference at 18: -3.46422, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1745543971.105229 1243516 buffer_comparator.cc:145] Difference at 19: -2.42414, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1745543971.105232 1243516 buffer_comparator.cc:145] Difference at 20: -3.75915, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1745543971.105235 1243516 buffer_comparator.cc:145] Difference at 21: -1.68308, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1745543971.105238 1243516 buffer_comparator.cc:145] Difference at 22: -3.98516, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1745543971.105241 1243516 buffer_comparator.cc:145] Difference at 23: -0.886093, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1745543971.105244 1243516 buffer_comparator.cc:145] Difference at 24: -4.01655, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1745543971.105247 1243516 buffer_comparator.cc:145] Difference at 25: -0.0175289, expected 36.0917</span></span>
<span class="line"><span>2025-04-25 01:19:31.105257: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.107487 1243516 buffer_comparator.cc:145] Difference at 16: -2.9902, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1745543971.107520 1243516 buffer_comparator.cc:145] Difference at 17: -3.08035, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1745543971.107524 1243516 buffer_comparator.cc:145] Difference at 18: -3.46422, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1745543971.107527 1243516 buffer_comparator.cc:145] Difference at 19: -2.42414, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1745543971.107529 1243516 buffer_comparator.cc:145] Difference at 20: -3.75915, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1745543971.107532 1243516 buffer_comparator.cc:145] Difference at 21: -1.68308, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1745543971.107535 1243516 buffer_comparator.cc:145] Difference at 22: -3.98516, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1745543971.107538 1243516 buffer_comparator.cc:145] Difference at 23: -0.886093, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1745543971.107541 1243516 buffer_comparator.cc:145] Difference at 24: -4.01655, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1745543971.107544 1243516 buffer_comparator.cc:145] Difference at 25: -0.0175289, expected 36.0917</span></span>
<span class="line"><span>2025-04-25 01:19:31.107552: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.109861 1243516 buffer_comparator.cc:145] Difference at 16: -2.9902, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1745543971.109893 1243516 buffer_comparator.cc:145] Difference at 17: -3.08035, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1745543971.109897 1243516 buffer_comparator.cc:145] Difference at 18: -3.46422, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1745543971.109900 1243516 buffer_comparator.cc:145] Difference at 19: -2.42414, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1745543971.109902 1243516 buffer_comparator.cc:145] Difference at 20: -3.75915, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1745543971.109905 1243516 buffer_comparator.cc:145] Difference at 21: -1.68308, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1745543971.109908 1243516 buffer_comparator.cc:145] Difference at 22: -3.98516, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1745543971.109911 1243516 buffer_comparator.cc:145] Difference at 23: -0.886093, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1745543971.109914 1243516 buffer_comparator.cc:145] Difference at 24: -4.01655, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1745543971.109917 1243516 buffer_comparator.cc:145] Difference at 25: -0.0175289, expected 36.0917</span></span>
<span class="line"><span>2025-04-25 01:19:31.109926: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.112245 1243516 buffer_comparator.cc:145] Difference at 16: -2.9902, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1745543971.112284 1243516 buffer_comparator.cc:145] Difference at 17: -3.08035, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1745543971.112287 1243516 buffer_comparator.cc:145] Difference at 18: -3.46422, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1745543971.112290 1243516 buffer_comparator.cc:145] Difference at 19: -2.42414, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1745543971.112293 1243516 buffer_comparator.cc:145] Difference at 20: -3.75915, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1745543971.112296 1243516 buffer_comparator.cc:145] Difference at 21: -1.68308, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1745543971.112299 1243516 buffer_comparator.cc:145] Difference at 22: -3.98516, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1745543971.112302 1243516 buffer_comparator.cc:145] Difference at 23: -0.886093, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1745543971.112305 1243516 buffer_comparator.cc:145] Difference at 24: -4.01655, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1745543971.112308 1243516 buffer_comparator.cc:145] Difference at 25: -0.0175289, expected 36.0917</span></span>
<span class="line"><span>2025-04-25 01:19:31.112317: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.114635 1243516 buffer_comparator.cc:145] Difference at 16: -2.9902, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1745543971.114669 1243516 buffer_comparator.cc:145] Difference at 17: -3.08035, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1745543971.114672 1243516 buffer_comparator.cc:145] Difference at 18: -3.46422, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1745543971.114675 1243516 buffer_comparator.cc:145] Difference at 19: -2.42414, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1745543971.114678 1243516 buffer_comparator.cc:145] Difference at 20: -3.75915, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1745543971.114681 1243516 buffer_comparator.cc:145] Difference at 21: -1.68308, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1745543971.114684 1243516 buffer_comparator.cc:145] Difference at 22: -3.98516, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1745543971.114687 1243516 buffer_comparator.cc:145] Difference at 23: -0.886093, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1745543971.114690 1243516 buffer_comparator.cc:145] Difference at 24: -4.01655, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1745543971.114693 1243516 buffer_comparator.cc:145] Difference at 25: -0.0175289, expected 36.0917</span></span>
<span class="line"><span>2025-04-25 01:19:31.114700: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.139920 1243516 buffer_comparator.cc:145] Difference at 16: 3.70289, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1745543971.139965 1243516 buffer_comparator.cc:145] Difference at 17: 3.76075, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1745543971.139969 1243516 buffer_comparator.cc:145] Difference at 18: 4.29592, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1745543971.139972 1243516 buffer_comparator.cc:145] Difference at 19: 2.9263, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1745543971.139975 1243516 buffer_comparator.cc:145] Difference at 20: 4.77082, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1745543971.139978 1243516 buffer_comparator.cc:145] Difference at 21: 1.81923, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1745543971.139981 1243516 buffer_comparator.cc:145] Difference at 22: 4.9023, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1745543971.139984 1243516 buffer_comparator.cc:145] Difference at 23: 0.977103, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1745543971.139987 1243516 buffer_comparator.cc:145] Difference at 24: 4.84085, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1745543971.139990 1243516 buffer_comparator.cc:145] Difference at 25: -0.101811, expected 11.3838</span></span>
<span class="line"><span>2025-04-25 01:19:31.140000: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.142275 1243516 buffer_comparator.cc:145] Difference at 16: 3.70289, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1745543971.142307 1243516 buffer_comparator.cc:145] Difference at 17: 3.76075, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1745543971.142311 1243516 buffer_comparator.cc:145] Difference at 18: 4.29592, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1745543971.142317 1243516 buffer_comparator.cc:145] Difference at 19: 2.9263, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1745543971.142320 1243516 buffer_comparator.cc:145] Difference at 20: 4.77082, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1745543971.142323 1243516 buffer_comparator.cc:145] Difference at 21: 1.81923, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1745543971.142326 1243516 buffer_comparator.cc:145] Difference at 22: 4.9023, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1745543971.142329 1243516 buffer_comparator.cc:145] Difference at 23: 0.977103, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1745543971.142332 1243516 buffer_comparator.cc:145] Difference at 24: 4.84085, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1745543971.142335 1243516 buffer_comparator.cc:145] Difference at 25: -0.101811, expected 11.3838</span></span>
<span class="line"><span>2025-04-25 01:19:31.142342: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.144648 1243516 buffer_comparator.cc:145] Difference at 16: 3.70289, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1745543971.144672 1243516 buffer_comparator.cc:145] Difference at 17: 3.76075, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1745543971.144676 1243516 buffer_comparator.cc:145] Difference at 18: 4.29592, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1745543971.144678 1243516 buffer_comparator.cc:145] Difference at 19: 2.9263, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1745543971.144682 1243516 buffer_comparator.cc:145] Difference at 20: 4.77082, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1745543971.144684 1243516 buffer_comparator.cc:145] Difference at 21: 1.81923, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1745543971.144687 1243516 buffer_comparator.cc:145] Difference at 22: 4.9023, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1745543971.144690 1243516 buffer_comparator.cc:145] Difference at 23: 0.977103, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1745543971.144693 1243516 buffer_comparator.cc:145] Difference at 24: 4.84085, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1745543971.144696 1243516 buffer_comparator.cc:145] Difference at 25: -0.101811, expected 11.3838</span></span>
<span class="line"><span>2025-04-25 01:19:31.144702: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.146887 1243516 buffer_comparator.cc:145] Difference at 32: 3.32287, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1745543971.146899 1243516 buffer_comparator.cc:145] Difference at 33: -3.26778, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1745543971.146902 1243516 buffer_comparator.cc:145] Difference at 34: 2.67801, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1745543971.146905 1243516 buffer_comparator.cc:145] Difference at 35: -4.07239, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1745543971.146908 1243516 buffer_comparator.cc:145] Difference at 36: 1.81365, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1745543971.146911 1243516 buffer_comparator.cc:145] Difference at 37: -4.28642, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1745543971.146914 1243516 buffer_comparator.cc:145] Difference at 38: 0.834827, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1745543971.146917 1243516 buffer_comparator.cc:145] Difference at 39: -4.46678, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1745543971.146920 1243516 buffer_comparator.cc:145] Difference at 40: -0.0272737, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1745543971.146923 1243516 buffer_comparator.cc:145] Difference at 41: -4.41948, expected 8.63119</span></span>
<span class="line"><span>2025-04-25 01:19:31.146928: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.149095 1243516 buffer_comparator.cc:145] Difference at 64: -2.80717, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1745543971.149106 1243516 buffer_comparator.cc:145] Difference at 65: 2.6368, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1745543971.149110 1243516 buffer_comparator.cc:145] Difference at 66: -2.23367, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1745543971.149113 1243516 buffer_comparator.cc:145] Difference at 67: 3.10709, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1745543971.149115 1243516 buffer_comparator.cc:145] Difference at 68: -1.37759, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1745543971.149120 1243516 buffer_comparator.cc:145] Difference at 69: 3.56963, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1745543971.149123 1243516 buffer_comparator.cc:145] Difference at 70: -0.691076, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1745543971.149126 1243516 buffer_comparator.cc:145] Difference at 71: 3.62855, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1745543971.149128 1243516 buffer_comparator.cc:145] Difference at 72: 0.0269013, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1745543971.149131 1243516 buffer_comparator.cc:145] Difference at 73: 3.4102, expected 8.82565</span></span>
<span class="line"><span>2025-04-25 01:19:31.149136: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.151279 1243516 buffer_comparator.cc:145] Difference at 64: -2.80717, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1745543971.151291 1243516 buffer_comparator.cc:145] Difference at 65: 2.6368, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1745543971.151294 1243516 buffer_comparator.cc:145] Difference at 66: -2.23367, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1745543971.151297 1243516 buffer_comparator.cc:145] Difference at 67: 3.10709, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1745543971.151300 1243516 buffer_comparator.cc:145] Difference at 68: -1.37759, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1745543971.151303 1243516 buffer_comparator.cc:145] Difference at 69: 3.56963, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1745543971.151306 1243516 buffer_comparator.cc:145] Difference at 70: -0.691076, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1745543971.151309 1243516 buffer_comparator.cc:145] Difference at 71: 3.62855, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1745543971.151311 1243516 buffer_comparator.cc:145] Difference at 72: 0.0269013, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1745543971.151314 1243516 buffer_comparator.cc:145] Difference at 73: 3.4102, expected 8.82565</span></span>
<span class="line"><span>2025-04-25 01:19:31.151319: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.153460 1243516 buffer_comparator.cc:145] Difference at 64: -2.80717, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1745543971.153471 1243516 buffer_comparator.cc:145] Difference at 65: 2.6368, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1745543971.153475 1243516 buffer_comparator.cc:145] Difference at 66: -2.23367, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1745543971.153478 1243516 buffer_comparator.cc:145] Difference at 67: 3.10709, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1745543971.153481 1243516 buffer_comparator.cc:145] Difference at 68: -1.37759, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1745543971.153483 1243516 buffer_comparator.cc:145] Difference at 69: 3.56963, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1745543971.153486 1243516 buffer_comparator.cc:145] Difference at 70: -0.691076, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1745543971.153489 1243516 buffer_comparator.cc:145] Difference at 71: 3.62855, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1745543971.153492 1243516 buffer_comparator.cc:145] Difference at 72: 0.0269013, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1745543971.153495 1243516 buffer_comparator.cc:145] Difference at 73: 3.4102, expected 8.82565</span></span>
<span class="line"><span>2025-04-25 01:19:31.153499: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.155637 1243516 buffer_comparator.cc:145] Difference at 64: -2.80717, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1745543971.155648 1243516 buffer_comparator.cc:145] Difference at 65: 2.6368, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1745543971.155652 1243516 buffer_comparator.cc:145] Difference at 66: -2.23367, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1745543971.155655 1243516 buffer_comparator.cc:145] Difference at 67: 3.10709, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1745543971.155658 1243516 buffer_comparator.cc:145] Difference at 68: -1.37759, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1745543971.155660 1243516 buffer_comparator.cc:145] Difference at 69: 3.56963, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1745543971.155663 1243516 buffer_comparator.cc:145] Difference at 70: -0.691076, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1745543971.155668 1243516 buffer_comparator.cc:145] Difference at 71: 3.62855, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1745543971.155671 1243516 buffer_comparator.cc:145] Difference at 72: 0.0269013, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1745543971.155674 1243516 buffer_comparator.cc:145] Difference at 73: 3.4102, expected 8.82565</span></span>
<span class="line"><span>2025-04-25 01:19:31.155679: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.157827 1243516 buffer_comparator.cc:145] Difference at 64: -2.80717, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1745543971.157838 1243516 buffer_comparator.cc:145] Difference at 65: 2.6368, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1745543971.157841 1243516 buffer_comparator.cc:145] Difference at 66: -2.23367, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1745543971.157844 1243516 buffer_comparator.cc:145] Difference at 67: 3.10709, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1745543971.157847 1243516 buffer_comparator.cc:145] Difference at 68: -1.37759, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1745543971.157850 1243516 buffer_comparator.cc:145] Difference at 69: 3.56963, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1745543971.157853 1243516 buffer_comparator.cc:145] Difference at 70: -0.691076, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1745543971.157856 1243516 buffer_comparator.cc:145] Difference at 71: 3.62855, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1745543971.157858 1243516 buffer_comparator.cc:145] Difference at 72: 0.0269013, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1745543971.157861 1243516 buffer_comparator.cc:145] Difference at 73: 3.4102, expected 8.82565</span></span>
<span class="line"><span>2025-04-25 01:19:31.157866: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745543971.160008 1243516 buffer_comparator.cc:145] Difference at 64: -2.80717, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1745543971.160019 1243516 buffer_comparator.cc:145] Difference at 65: 2.6368, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1745543971.160022 1243516 buffer_comparator.cc:145] Difference at 66: -2.23367, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1745543971.160025 1243516 buffer_comparator.cc:145] Difference at 67: 3.10709, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1745543971.160028 1243516 buffer_comparator.cc:145] Difference at 68: -1.37759, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1745543971.160031 1243516 buffer_comparator.cc:145] Difference at 69: 3.56963, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1745543971.160034 1243516 buffer_comparator.cc:145] Difference at 70: -0.691076, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1745543971.160037 1243516 buffer_comparator.cc:145] Difference at 71: 3.62855, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1745543971.160039 1243516 buffer_comparator.cc:145] Difference at 72: 0.0269013, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1745543971.160042 1243516 buffer_comparator.cc:145] Difference at 73: 3.4102, expected 8.82565</span></span>
<span class="line"><span>2025-04-25 01:19:31.160047: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.69160</span></span>
<span class="line"><span>Validation:	Loss 0.52610	Accuracy 0.56250</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.52191</span></span>
<span class="line"><span>Validation:	Loss 0.43853	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.43407</span></span>
<span class="line"><span>Validation:	Loss 0.34492	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.33474</span></span>
<span class="line"><span>Validation:	Loss 0.25899	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.25559</span></span>
<span class="line"><span>Validation:	Loss 0.20521	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.20695</span></span>
<span class="line"><span>Validation:	Loss 0.16638	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.17174</span></span>
<span class="line"><span>Validation:	Loss 0.13772	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.14419</span></span>
<span class="line"><span>Validation:	Loss 0.11695	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.12464</span></span>
<span class="line"><span>Validation:	Loss 0.10098	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.10760</span></span>
<span class="line"><span>Validation:	Loss 0.08781	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.09339</span></span>
<span class="line"><span>Validation:	Loss 0.07668	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.08130</span></span>
<span class="line"><span>Validation:	Loss 0.06706	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.07190</span></span>
<span class="line"><span>Validation:	Loss 0.05846	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.06222</span></span>
<span class="line"><span>Validation:	Loss 0.05037	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.05212</span></span>
<span class="line"><span>Validation:	Loss 0.04124	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.04157</span></span>
<span class="line"><span>Validation:	Loss 0.03238	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.03226</span></span>
<span class="line"><span>Validation:	Loss 0.02585	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.02566</span></span>
<span class="line"><span>Validation:	Loss 0.02087	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.02061</span></span>
<span class="line"><span>Validation:	Loss 0.01738	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01712</span></span>
<span class="line"><span>Validation:	Loss 0.01484	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01447</span></span>
<span class="line"><span>Validation:	Loss 0.01260	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01219</span></span>
<span class="line"><span>Validation:	Loss 0.01094	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01075</span></span>
<span class="line"><span>Validation:	Loss 0.00983	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00975</span></span>
<span class="line"><span>Validation:	Loss 0.00902	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00898</span></span>
<span class="line"><span>Validation:	Loss 0.00836	Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-6/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span> todo inst:   %11 = addrspacecast { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }* %10 to { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.2.0.0.gep = getelementptr inbounds { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }* %10, i64 0, i32 2, i64 0, i32 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.0.0.extract, i64* %12, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %.fca.2.0.1.gep = getelementptr inbounds { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }* %10, i64 0, i32 2, i64 0, i32 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.0.1.extract, i64* %13, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %.fca.2.1.0.gep = getelementptr inbounds { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }* %10, i64 0, i32 2, i64 1, i32 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.1.0.extract, i64* %14, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %.fca.2.1.1.gep = getelementptr inbounds { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }* %10, i64 0, i32 2, i64 1, i32 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.1.1.extract, i64* %15, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %17 = addrspacecast { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }* %16 to { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.0.0.gep6 = getelementptr inbounds { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }* %16, i64 0, i32 0, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.0.extract5, {} addrspace(10)** %18, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %.fca.0.1.gep = getelementptr inbounds { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }* %16, i64 0, i32 0, i64 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.1.extract, {} addrspace(10)** %19, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %.fca.0.2.gep = getelementptr inbounds { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }* %16, i64 0, i32 0, i64 2, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.2.extract, {} addrspace(10)** %20, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %.fca.0.3.gep = getelementptr inbounds { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }* %16, i64 0, i32 0, i64 3, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.3.extract, {} addrspace(10)** %21, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %.fca.1.0.gep = getelementptr inbounds { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }* %16, i64 0, i32 1, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.1.0.extract, {} addrspace(10)** %22, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %.fca.1.1.gep = getelementptr inbounds { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }* %16, i64 0, i32 1, i64 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.1.1.extract, {} addrspace(10)** %23, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %23 = addrspacecast { [1 x {} addrspace(10)*] }* %22 to { [1 x {} addrspace(10)*] } addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.0.0.gep = getelementptr inbounds { [1 x {} addrspace(10)*] }, { [1 x {} addrspace(10)*] }* %22, i64 0, i32 0, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.0.extract, {} addrspace(10)** %24, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %29 = addrspacecast [2 x {} addrspace(10)*]* %28 to [2 x {} addrspace(10)*] addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.0.gep = getelementptr inbounds [2 x {} addrspace(10)*], [2 x {} addrspace(10)*]* %28, i64 0, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.extract, {} addrspace(10)** %30, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %.fca.1.gep = getelementptr inbounds [2 x {} addrspace(10)*], [2 x {} addrspace(10)*]* %28, i64 0, i64 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.1.extract, {} addrspace(10)** %31, align 8, !dbg !32, !noalias !45</span></span>
<span class="line"><span> todo inst:   %.fca.0.0.gep = getelementptr inbounds { [1 x {} addrspace(10)*], [1 x i64] }, { [1 x {} addrspace(10)*], [1 x i64] }* %10, i64 0, i32 0, i64 0, !dbg !93</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %1, {} addrspace(10)** %53, align 8, !dbg !93, !noalias !99</span></span>
<span class="line"><span> todo inst:   %.fca.1.0.gep = getelementptr inbounds { [1 x {} addrspace(10)*], [1 x i64] }, { [1 x {} addrspace(10)*], [1 x i64] }* %10, i64 0, i32 1, i64 0, !dbg !93</span></span>
<span class="line"><span> todo inst:   store i64 %.sroa.1.0.copyload, i64* %54, align 8, !dbg !93, !noalias !99</span></span>
<span class="line"><span> todo inst:   %55 = addrspacecast { [1 x {} addrspace(10)*], [1 x i64] }* %10 to { [1 x {} addrspace(10)*], [1 x i64] } addrspace(11)*, !dbg !93</span></span>
<span class="line"><span> todo inst:   call fastcc void @julia__selectdim_121986({ {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }* noalias nocapture nofree noundef nonnull writeonly sret({ {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }) align 8 dereferenceable(48) %7, [1 x {} addrspace(10)*]* noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) &quot;enzymejl_returnRoots&quot; %10, {} addrspace(10)* noundef nonnull align 8 dereferenceable(40) %29, { [1 x [1 x i64]], i64, [1 x [1 x i64]] } addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(24) %37) #48, !dbg !132</span></span>
<span class="line"><span> todo inst:   %39 = getelementptr inbounds { {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }, { {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }* %7, i64 0, i32 0, !dbg !144</span></span>
<span class="line"><span> todo inst:   %41 = load atomic {} addrspace(10)*, {} addrspace(10)** %40 unordered, align 8, !dbg !144, !tbaa !140, !alias.scope !142, !noalias !151, !nonnull !0, !dereferenceable !123, !align !40, !enzyme_inactive !0, !enzyme_type !124, !enzymejl_byref_MUT_REF !0, !enzymejl_source_type_Reactant.ConcretePJRTArray\\7BFloat32\\2C\\203\\2C\\201\\2C\\20Reactant.Sharding.ShardInfo\\7BReactant.Sharding.NoSharding\\2C\\20Nothing\\7D\\7D !0</span></span>
<span class="line"><span> todo inst:   %42 = getelementptr inbounds { {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }, { {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }* %7, i64 0, i32 1, i32 0, !dbg !145</span></span>
<span class="line"><span> todo inst:   %46 = addrspacecast [1 x [1 x i64]]* %43 to [1 x [1 x i64]] addrspace(11)*, !dbg !145</span></span>
<span class="line"><span> todo inst:   %44 = getelementptr inbounds { {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }, { {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }* %7, i64 0, i32 1, i32 1, !dbg !145</span></span>
<span class="line"><span> todo inst:   %48 = load i64, i64* %45, align 8, !dbg !145, !tbaa !140, !alias.scope !142, !noalias !151, !enzyme_inactive !0, !enzyme_type !75, !enzymejl_source_type_Int64 !0, !enzymejl_byref_BITS_VALUE !0</span></span>
<span class="line"><span> todo inst:   %46 = getelementptr inbounds { {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }, { {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }* %7, i64 0, i32 1, i32 2, !dbg !145</span></span>
<span class="line"><span> todo inst:   %50 = addrspacecast [1 x [1 x i64]]* %47 to [1 x [1 x i64]] addrspace(11)*, !dbg !145</span></span>
<span class="line"><span> todo inst:   call fastcc void @julia__selectdim_121986({ {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }* noalias nocapture nofree noundef nonnull writeonly sret({ {} addrspace(10)*, { [1 x [1 x i64]], i64, [1 x [1 x i64]] }, i64, i64 }) align 8 dereferenceable(48) %43, [1 x {} addrspace(10)*]* noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) &quot;enzymejl_returnRoots&quot; %15, {} addrspace(10)* noundef nonnull align 8 dereferenceable(40) %34, { [1 x [1 x i64]], i64, [1 x [1 x i64]] } addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(24) %42) #48, !dbg !132</span></span>
<span class="line"><span> todo inst:   call void @zeroType.27({} addrspace(10)* %30, i8 0, i64 8), !enzyme_zerostack !0</span></span>
<span class="line"><span> todo inst:   %46 = bitcast {} addrspace(10)* %30 to [1 x {} addrspace(10)*] addrspace(10)*, !enzyme_caststack !0</span></span>
<span class="line"><span> todo inst:   %98 = addrspacecast [1 x {} addrspace(10)*] addrspace(10)* %47 to [1 x {} addrspace(10)*]*, !dbg !152</span></span>
<span class="line"><span> todo inst:   call void @zeroType.24({} addrspace(10)* %156, i8 0, i64 16), !dbg !32, !enzyme_zerostack !0</span></span>
<span class="line"><span> todo inst:   %186 = bitcast {} addrspace(10)* %156 to [2 x {} addrspace(10)*] addrspace(10)*, !dbg !32, !enzyme_caststack !0</span></span>
<span class="line"><span> todo inst:   %188 = addrspacecast [2 x {} addrspace(10)*] addrspace(10)* %187 to [2 x {} addrspace(10)*] addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.0.gep = getelementptr [2 x {} addrspace(10)*], [2 x {} addrspace(10)*] addrspace(10)* %187, i64 0, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.extract, {} addrspace(10)* addrspace(10)* %189, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.1.gep = getelementptr [2 x {} addrspace(10)*], [2 x {} addrspace(10)*] addrspace(10)* %187, i64 0, i64 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.1.extract, {} addrspace(10)* addrspace(10)* %190, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %156, {} addrspace(10)* %.fca.0.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %156, {} addrspace(10)* %.fca.1.extract)</span></span>
<span class="line"><span> todo inst:   call void @zeroType.23({} addrspace(10)* %158, i8 0, i64 8), !dbg !32, !enzyme_zerostack !0</span></span>
<span class="line"><span> todo inst:   %181 = bitcast {} addrspace(10)* %158 to { [1 x {} addrspace(10)*] } addrspace(10)*, !dbg !32, !enzyme_caststack !0</span></span>
<span class="line"><span> todo inst:   %183 = addrspacecast { [1 x {} addrspace(10)*] } addrspace(10)* %182 to { [1 x {} addrspace(10)*] } addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.0.0.gep = getelementptr { [1 x {} addrspace(10)*] }, { [1 x {} addrspace(10)*] } addrspace(10)* %182, i64 0, i32 0, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.0.extract, {} addrspace(10)* addrspace(10)* %184, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %158, {} addrspace(10)* %.fca.0.0.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void @zeroType.22({} addrspace(10)* %160, i8 0, i64 48), !dbg !32, !enzyme_zerostack !0</span></span>
<span class="line"><span> todo inst:   %176 = bitcast {} addrspace(10)* %160 to { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)*, !dbg !32, !enzyme_caststack !0</span></span>
<span class="line"><span> todo inst:   %178 = addrspacecast { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %177 to { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.0.0.gep6 = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %177, i64 0, i32 0, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.0.extract5, {} addrspace(10)* addrspace(10)* %179, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.0.1.gep = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %177, i64 0, i32 0, i64 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.1.extract, {} addrspace(10)* addrspace(10)* %180, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.0.2.gep = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %177, i64 0, i32 0, i64 2, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.2.extract, {} addrspace(10)* addrspace(10)* %181, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.0.3.gep = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %177, i64 0, i32 0, i64 3, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.3.extract, {} addrspace(10)* addrspace(10)* %182, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.1.0.gep = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %177, i64 0, i32 1, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.1.0.extract, {} addrspace(10)* addrspace(10)* %183, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.1.1.gep = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %177, i64 0, i32 1, i64 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.1.1.extract, {} addrspace(10)* addrspace(10)* %184, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %160, {} addrspace(10)* %.fca.0.0.extract5), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %160, {} addrspace(10)* %.fca.0.1.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %160, {} addrspace(10)* %.fca.0.2.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %160, {} addrspace(10)* %.fca.0.3.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %160, {} addrspace(10)* %.fca.1.0.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %160, {} addrspace(10)* %.fca.1.1.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void @zeroType({} addrspace(10)* %162, i8 0, i64 64), !dbg !32, !enzyme_zerostack !0</span></span>
<span class="line"><span> todo inst:   %171 = bitcast {} addrspace(10)* %162 to { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)*, !dbg !32, !enzyme_caststack !0</span></span>
<span class="line"><span> todo inst:   %173 = addrspacecast { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)* %172 to { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.2.0.0.gep = getelementptr { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)* %172, i64 0, i32 2, i64 0, i32 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.0.0.extract, i64 addrspace(10)* %174, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.2.0.1.gep = getelementptr { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)* %172, i64 0, i32 2, i64 0, i32 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.0.1.extract, i64 addrspace(10)* %175, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.2.1.0.gep = getelementptr { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)* %172, i64 0, i32 2, i64 1, i32 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.1.0.extract, i64 addrspace(10)* %176, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.2.1.1.gep = getelementptr { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)* %172, i64 0, i32 2, i64 1, i32 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.1.1.extract, i64 addrspace(10)* %177, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %7, {} addrspace(10)* %.fca.1.extract)</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %7, {} addrspace(10)* %.fca.0.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   %36 = bitcast {} addrspace(10)* %7 to [2 x {} addrspace(10)*] addrspace(10)*, !dbg !32, !enzyme_caststack !0</span></span>
<span class="line"><span> todo inst:   %38 = addrspacecast [2 x {} addrspace(10)*] addrspace(10)* %37 to [2 x {} addrspace(10)*] addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.0.gep = getelementptr [2 x {} addrspace(10)*], [2 x {} addrspace(10)*] addrspace(10)* %37, i64 0, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.extract, {} addrspace(10)* addrspace(10)* %39, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.1.gep = getelementptr [2 x {} addrspace(10)*], [2 x {} addrspace(10)*] addrspace(10)* %37, i64 0, i64 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.1.extract, {} addrspace(10)* addrspace(10)* %40, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %9, {} addrspace(10)* %.fca.0.0.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   %31 = bitcast {} addrspace(10)* %9 to { [1 x {} addrspace(10)*] } addrspace(10)*, !dbg !32, !enzyme_caststack !0</span></span>
<span class="line"><span> todo inst:   %33 = addrspacecast { [1 x {} addrspace(10)*] } addrspace(10)* %32 to { [1 x {} addrspace(10)*] } addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.0.0.gep = getelementptr { [1 x {} addrspace(10)*] }, { [1 x {} addrspace(10)*] } addrspace(10)* %32, i64 0, i32 0, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.0.extract, {} addrspace(10)* addrspace(10)* %34, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %11, {} addrspace(10)* %.fca.1.1.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %11, {} addrspace(10)* %.fca.1.0.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %11, {} addrspace(10)* %.fca.0.3.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %11, {} addrspace(10)* %.fca.0.2.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %11, {} addrspace(10)* %.fca.0.1.extract), !dbg !32</span></span>
<span class="line"><span> todo inst:   call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* %11, {} addrspace(10)* %.fca.0.0.extract5), !dbg !32</span></span>
<span class="line"><span> todo inst:   %26 = bitcast {} addrspace(10)* %11 to { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)*, !dbg !32, !enzyme_caststack !0</span></span>
<span class="line"><span> todo inst:   %28 = addrspacecast { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %27 to { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.0.0.gep6 = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %27, i64 0, i32 0, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.0.extract5, {} addrspace(10)* addrspace(10)* %29, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.0.1.gep = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %27, i64 0, i32 0, i64 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.1.extract, {} addrspace(10)* addrspace(10)* %30, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.0.2.gep = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %27, i64 0, i32 0, i64 2, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.2.extract, {} addrspace(10)* addrspace(10)* %31, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.0.3.gep = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %27, i64 0, i32 0, i64 3, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.0.3.extract, {} addrspace(10)* addrspace(10)* %32, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.1.0.gep = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %27, i64 0, i32 1, i64 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.1.0.extract, {} addrspace(10)* addrspace(10)* %33, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.1.1.gep = getelementptr { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] }, { [4 x {} addrspace(10)*], [2 x {} addrspace(10)*] } addrspace(10)* %27, i64 0, i32 1, i64 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store {} addrspace(10)* %.fca.1.1.extract, {} addrspace(10)* addrspace(10)* %34, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %21 = bitcast {} addrspace(10)* %13 to { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)*, !dbg !32, !enzyme_caststack !0</span></span>
<span class="line"><span> todo inst:   %23 = addrspacecast { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)* %22 to { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(11)*, !dbg !32</span></span>
<span class="line"><span> todo inst:   %.fca.2.0.0.gep = getelementptr { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)* %22, i64 0, i32 2, i64 0, i32 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.0.0.extract, i64 addrspace(10)* %24, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.2.0.1.gep = getelementptr { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)* %22, i64 0, i32 2, i64 0, i32 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.0.1.extract, i64 addrspace(10)* %25, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.2.1.0.gep = getelementptr { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)* %22, i64 0, i32 2, i64 1, i32 0, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.1.0.extract, i64 addrspace(10)* %26, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span> todo inst:   %.fca.2.1.1.gep = getelementptr { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] }, { [3 x {} addrspace(10)*], {} addrspace(10)*, [2 x { i64, i64 }] } addrspace(10)* %22, i64 0, i32 2, i64 1, i32 1, !dbg !32</span></span>
<span class="line"><span> todo inst:   store i64 %.fca.2.1.1.extract, i64 addrspace(10)* %27, align 8, !dbg !32, !noalias !51</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.62062</span></span>
<span class="line"><span>Validation:	Loss 0.54368	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.50304</span></span>
<span class="line"><span>Validation:	Loss 0.46212	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.42818</span></span>
<span class="line"><span>Validation:	Loss 0.39294	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.35640</span></span>
<span class="line"><span>Validation:	Loss 0.32424	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.29423</span></span>
<span class="line"><span>Validation:	Loss 0.26726	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.23952</span></span>
<span class="line"><span>Validation:	Loss 0.22031	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.20051</span></span>
<span class="line"><span>Validation:	Loss 0.18527	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.16967</span></span>
<span class="line"><span>Validation:	Loss 0.15681	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.14284</span></span>
<span class="line"><span>Validation:	Loss 0.13358	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.12323</span></span>
<span class="line"><span>Validation:	Loss 0.11630	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.10771</span></span>
<span class="line"><span>Validation:	Loss 0.10331	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.09620</span></span>
<span class="line"><span>Validation:	Loss 0.09221	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.08490</span></span>
<span class="line"><span>Validation:	Loss 0.08249	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.07662</span></span>
<span class="line"><span>Validation:	Loss 0.07403	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.06885</span></span>
<span class="line"><span>Validation:	Loss 0.06663	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.06218</span></span>
<span class="line"><span>Validation:	Loss 0.05960	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.05400</span></span>
<span class="line"><span>Validation:	Loss 0.05248	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.04754</span></span>
<span class="line"><span>Validation:	Loss 0.04442	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.03889</span></span>
<span class="line"><span>Validation:	Loss 0.03400	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.02869</span></span>
<span class="line"><span>Validation:	Loss 0.02336	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01921</span></span>
<span class="line"><span>Validation:	Loss 0.01593	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01361</span></span>
<span class="line"><span>Validation:	Loss 0.01193	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01067</span></span>
<span class="line"><span>Validation:	Loss 0.00966	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00879</span></span>
<span class="line"><span>Validation:	Loss 0.00818	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00754</span></span>
<span class="line"><span>Validation:	Loss 0.00711	Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.5</span></span>
<span class="line"><span>Commit 760b2e5b739 (2025-04-14 06:53 UTC)</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,47)]))}const k=a(i,[["render",c]]);export{f as __pageData,k as default};
