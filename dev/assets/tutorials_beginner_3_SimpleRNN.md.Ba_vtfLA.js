import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.bV3h_rQg.js";const E=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>   7252.5 ms  ✓ StaticArrays</span></span>
<span class="line"><span>    614.2 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    633.0 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    825.6 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    658.7 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    703.9 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    931.4 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   3900.7 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    686.3 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    742.7 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5223.1 ms  ✓ NNlib</span></span>
<span class="line"><span>    851.3 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    954.7 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5751.2 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9296.0 ms  ✓ Lux</span></span>
<span class="line"><span>  15 dependencies successfully precompiled in 35 seconds. 105 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>    960.4 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>  52293.7 ms  ✓ CUDA</span></span>
<span class="line"><span>   5089.7 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8149.8 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5327.9 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  5 dependencies successfully precompiled in 72 seconds. 95 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   4946.0 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   5091.6 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5414.5 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   5035.1 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5199.1 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5244.8 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   6266.1 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 7 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   5094.1 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5348.0 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5276.7 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 107 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   5876.9 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 174 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>  32917.3 ms  ✓ JLD2</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 33 seconds. 31 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    365.9 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    420.9 ms  ✓ ShowCases</span></span>
<span class="line"><span>    342.3 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    324.1 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    794.0 ms  ✓ InitialValues</span></span>
<span class="line"><span>    390.7 ms  ✓ NameResolution</span></span>
<span class="line"><span>    382.7 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>   1028.3 ms  ✓ Baselet</span></span>
<span class="line"><span>   2831.3 ms  ✓ Accessors</span></span>
<span class="line"><span>    635.5 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>    804.4 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>    839.3 ms  ✓ BangBang</span></span>
<span class="line"><span>    706.2 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>    520.7 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    754.8 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    507.4 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    861.1 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2759.7 ms  ✓ Transducers</span></span>
<span class="line"><span>    674.7 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>  17574.9 ms  ✓ MLStyle</span></span>
<span class="line"><span>   4235.4 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   5058.0 ms  ✓ FLoops</span></span>
<span class="line"><span>   6081.0 ms  ✓ MLUtils</span></span>
<span class="line"><span>  23 dependencies successfully precompiled in 34 seconds. 75 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangDataFramesExt...</span></span>
<span class="line"><span>   1627.0 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling TransducersDataFramesExt...</span></span>
<span class="line"><span>   1371.3 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 61 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1862.9 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2186.3 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 177 already precompiled.</span></span>
<span class="line"><span>Precompiling AccessorsStructArraysExt...</span></span>
<span class="line"><span>    484.9 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 16 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    497.7 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 22 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysStaticArraysExt...</span></span>
<span class="line"><span>    650.0 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   2820.1 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 163 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61662</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59586</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56661</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53892</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50615</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50638</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47986</span></span>
<span class="line"><span>Validation: Loss 0.47207 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47750 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47306</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44507</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43348</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42229</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41654</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39818</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38418</span></span>
<span class="line"><span>Validation: Loss 0.37573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38207 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35373</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36714</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34572</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33052</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31568</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31465</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30486</span></span>
<span class="line"><span>Validation: Loss 0.29141 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29848 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29594</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27336</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25462</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23812</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24490</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24399</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22616</span></span>
<span class="line"><span>Validation: Loss 0.22145 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22866 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21947</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20865</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18867</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19873</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17792</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16953</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16778</span></span>
<span class="line"><span>Validation: Loss 0.16523 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17209 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15598</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15897</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13422</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13523</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13988</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13363</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12350</span></span>
<span class="line"><span>Validation: Loss 0.12151 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12737 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12085</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11197</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10594</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10151</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08907</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09151</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10476</span></span>
<span class="line"><span>Validation: Loss 0.08685 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09123 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08662</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07372</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07408</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07657</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06411</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06624</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05938</span></span>
<span class="line"><span>Validation: Loss 0.06017 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06309 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05718</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05483</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05499</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04740</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05065</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04557</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04344</span></span>
<span class="line"><span>Validation: Loss 0.04454 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04659 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04545</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04161</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03897</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03899</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03664</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03706</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03370</span></span>
<span class="line"><span>Validation: Loss 0.03599 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03765 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03537</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03503</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03280</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03096</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03156</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03134</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02637</span></span>
<span class="line"><span>Validation: Loss 0.03055 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03199 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02902</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02841</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02775</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02744</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02577</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02460</span></span>
<span class="line"><span>Validation: Loss 0.02660 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02790 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02414</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02522</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02046</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02818</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02391</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02515</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02251</span></span>
<span class="line"><span>Validation: Loss 0.02356 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02473 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02253</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02236</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02295</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02019</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01937</span></span>
<span class="line"><span>Validation: Loss 0.02108 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02216 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01904</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02076</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02044</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01907</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01844</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01904</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02018</span></span>
<span class="line"><span>Validation: Loss 0.01904 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02003 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01782</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01942</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01717</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01775</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01783</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01447</span></span>
<span class="line"><span>Validation: Loss 0.01730 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01823 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01572</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01832</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01607</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01587</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01673</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01436</span></span>
<span class="line"><span>Validation: Loss 0.01584 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01670 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01569</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01569</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01394</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01265</span></span>
<span class="line"><span>Validation: Loss 0.01458 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01539 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01413</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01357</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01409</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01353</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01365</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01063</span></span>
<span class="line"><span>Validation: Loss 0.01349 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01424 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01413</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01217</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01316</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01168</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01276</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01224</span></span>
<span class="line"><span>Validation: Loss 0.01249 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01320 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01308</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01116</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01181</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01043</span></span>
<span class="line"><span>Validation: Loss 0.01152 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01217 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01157</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01088</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01113</span></span>
<span class="line"><span>Validation: Loss 0.01048 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01107 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00933</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01026</span></span>
<span class="line"><span>Validation: Loss 0.00932 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00982 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00833</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00909</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00816</span></span>
<span class="line"><span>Validation: Loss 0.00829 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00872 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00821</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00737</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00709</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00823</span></span>
<span class="line"><span>Validation: Loss 0.00755 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00793 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61151</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59402</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56505</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54626</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50964</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50409</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49972</span></span>
<span class="line"><span>Validation: Loss 0.47252 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46509 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46365</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43761</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43112</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41816</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39819</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38012</span></span>
<span class="line"><span>Validation: Loss 0.37679 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36851 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36692</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36168</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35449</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33718</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32050</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30121</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29236</span></span>
<span class="line"><span>Validation: Loss 0.29273 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28375 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.30340</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27049</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25643</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26207</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23676</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24401</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.19486</span></span>
<span class="line"><span>Validation: Loss 0.22328 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21406 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20817</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22702</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20086</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18178</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17449</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17589</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21664</span></span>
<span class="line"><span>Validation: Loss 0.16786 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15902 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16218</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14533</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14456</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14696</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14561</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13701</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10850</span></span>
<span class="line"><span>Validation: Loss 0.12422 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11681 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12470</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10950</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10675</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10214</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10569</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09539</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08749</span></span>
<span class="line"><span>Validation: Loss 0.08957 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08396 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08577</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08121</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07610</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07024</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07189</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07102</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06599</span></span>
<span class="line"><span>Validation: Loss 0.06239 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05855 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06237</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05293</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05512</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05126</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04863</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05108</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04457</span></span>
<span class="line"><span>Validation: Loss 0.04579 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04310 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04260</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04271</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04033</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04004</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04013</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03835</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03673</span></span>
<span class="line"><span>Validation: Loss 0.03684 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03467 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03533</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03504</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03191</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03226</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03147</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03370</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03401</span></span>
<span class="line"><span>Validation: Loss 0.03121 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02932 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03096</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03004</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02895</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02810</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02705</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02678</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02574</span></span>
<span class="line"><span>Validation: Loss 0.02711 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02543 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02643</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02416</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02658</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02524</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02353</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02351</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02520</span></span>
<span class="line"><span>Validation: Loss 0.02396 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02244 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02315</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02337</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02100</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02226</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02095</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02100</span></span>
<span class="line"><span>Validation: Loss 0.02143 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02005 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02106</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01890</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01952</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01941</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02058</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01973</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01897</span></span>
<span class="line"><span>Validation: Loss 0.01935 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01868</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01858</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01864</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01806</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01748</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01692</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01523</span></span>
<span class="line"><span>Validation: Loss 0.01759 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01640 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01744</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01593</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01699</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01600</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01499</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01691</span></span>
<span class="line"><span>Validation: Loss 0.01610 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01499 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01595</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01620</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01512</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01552</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01391</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01417</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01104</span></span>
<span class="line"><span>Validation: Loss 0.01481 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01377 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01482</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01299</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01366</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01446</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01348</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01283</span></span>
<span class="line"><span>Validation: Loss 0.01371 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01273 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01502</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01213</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01444</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01353</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01090</span></span>
<span class="line"><span>Validation: Loss 0.01270 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01179 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01209</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01010</span></span>
<span class="line"><span>Validation: Loss 0.01176 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01092 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01255</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01086</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01082</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01190</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01015</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01010</span></span>
<span class="line"><span>Validation: Loss 0.01077 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01000 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01017</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01001</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00988</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00946</span></span>
<span class="line"><span>Validation: Loss 0.00964 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00897 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00897</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00934</span></span>
<span class="line"><span>Validation: Loss 0.00854 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00797 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00877</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00833</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00751</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00701</span></span>
<span class="line"><span>Validation: Loss 0.00770 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00721 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
