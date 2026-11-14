import{_ as a,c as n,a2 as p,o as i}from"./chunks/framework.B-8EJUGo.js";const d=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(t,s,h,c,k,o){return i(),n("div",null,s[0]||(s[0]=[p(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    399.0 ms  ✓ Future</span></span>
<span class="line"><span>    331.7 ms  ✓ CEnum</span></span>
<span class="line"><span>    493.0 ms  ✓ Statistics</span></span>
<span class="line"><span>    326.3 ms  ✓ Reexport</span></span>
<span class="line"><span>    362.5 ms  ✓ Zlib_jll</span></span>
<span class="line"><span>    364.8 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>    379.3 ms  ✓ DiffResults</span></span>
<span class="line"><span>    477.9 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>   2138.5 ms  ✓ Hwloc</span></span>
<span class="line"><span>   2644.5 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>   1524.4 ms  ✓ Setfield</span></span>
<span class="line"><span>    954.3 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    450.4 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>   3637.7 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>   7258.5 ms  ✓ StaticArrays</span></span>
<span class="line"><span>    598.4 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    620.1 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    630.7 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    595.9 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    664.9 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    894.4 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   4009.2 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    662.3 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    717.4 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5199.2 ms  ✓ NNlib</span></span>
<span class="line"><span>    865.2 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    946.1 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5848.9 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9170.3 ms  ✓ Lux</span></span>
<span class="line"><span>  29 dependencies successfully precompiled in 37 seconds. 91 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>    294.6 ms  ✓ IteratorInterfaceExtensions</span></span>
<span class="line"><span>    351.9 ms  ✓ ExprTools</span></span>
<span class="line"><span>    538.2 ms  ✓ AbstractFFTs</span></span>
<span class="line"><span>    419.3 ms  ✓ SuiteSparse_jll</span></span>
<span class="line"><span>    549.9 ms  ✓ Serialization</span></span>
<span class="line"><span>    474.7 ms  ✓ OrderedCollections</span></span>
<span class="line"><span>    288.2 ms  ✓ DataValueInterfaces</span></span>
<span class="line"><span>    347.0 ms  ✓ DataAPI</span></span>
<span class="line"><span>    326.5 ms  ✓ TableTraits</span></span>
<span class="line"><span>   2299.1 ms  ✓ FixedPointNumbers</span></span>
<span class="line"><span>   2726.5 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>   3750.6 ms  ✓ SparseArrays</span></span>
<span class="line"><span>   6688.7 ms  ✓ LLVM</span></span>
<span class="line"><span>    460.6 ms  ✓ PooledArrays</span></span>
<span class="line"><span>   3990.7 ms  ✓ Test</span></span>
<span class="line"><span>    434.2 ms  ✓ Missings</span></span>
<span class="line"><span>   1744.7 ms  ✓ DataStructures</span></span>
<span class="line"><span>    860.1 ms  ✓ Tables</span></span>
<span class="line"><span>    655.7 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>    958.8 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   2273.5 ms  ✓ ColorTypes</span></span>
<span class="line"><span>   1894.9 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   2159.9 ms  ✓ GPUArrays</span></span>
<span class="line"><span>   1333.5 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>    504.7 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>   1343.5 ms  ✓ LLVM → BFloat16sExt</span></span>
<span class="line"><span>   4418.7 ms  ✓ Colors</span></span>
<span class="line"><span>   1388.4 ms  ✓ NVTX</span></span>
<span class="line"><span>  20641.1 ms  ✓ PrettyTables</span></span>
<span class="line"><span>  27790.8 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>  47430.6 ms  ✓ DataFrames</span></span>
<span class="line"><span>  51963.9 ms  ✓ CUDA</span></span>
<span class="line"><span>   5130.2 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8690.9 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5936.7 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  35 dependencies successfully precompiled in 154 seconds. 65 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesGPUArraysExt...</span></span>
<span class="line"><span>   1469.2 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 42 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersGPUArraysExt...</span></span>
<span class="line"><span>   1558.6 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceSparseArraysExt...</span></span>
<span class="line"><span>    590.3 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 7 already precompiled.</span></span>
<span class="line"><span>Precompiling ChainRulesCoreSparseArraysExt...</span></span>
<span class="line"><span>    643.5 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 11 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    647.4 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 17 already precompiled.</span></span>
<span class="line"><span>Precompiling AbstractFFTsChainRulesCoreExt...</span></span>
<span class="line"><span>    429.9 ms  ✓ AbstractFFTs → AbstractFFTsChainRulesCoreExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 0 seconds. 9 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   5012.5 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   5067.2 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5372.8 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   5194.5 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5269.6 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5322.2 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   5864.1 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 6 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   5090.8 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5660.9 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5046.1 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 107 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   5806.6 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 174 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>  33796.3 ms  ✓ JLD2</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 34 seconds. 31 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    646.5 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>   1194.8 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    607.4 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>    541.1 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    759.5 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>   2044.6 ms  ✓ Distributed</span></span>
<span class="line"><span>    743.7 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>   2400.3 ms  ✓ StatsBase</span></span>
<span class="line"><span>   2887.4 ms  ✓ Transducers</span></span>
<span class="line"><span>    728.0 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   4996.4 ms  ✓ FLoops</span></span>
<span class="line"><span>   6171.6 ms  ✓ MLUtils</span></span>
<span class="line"><span>  12 dependencies successfully precompiled in 16 seconds. 86 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangDataFramesExt...</span></span>
<span class="line"><span>   1668.9 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling TransducersDataFramesExt...</span></span>
<span class="line"><span>   1398.5 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 61 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1609.5 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2423.1 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 177 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    398.2 ms  ✓ FillArrays → FillArraysStatisticsExt</span></span>
<span class="line"><span>    589.8 ms  ✓ SuiteSparse</span></span>
<span class="line"><span>    669.2 ms  ✓ FillArrays → FillArraysSparseArraysExt</span></span>
<span class="line"><span>    746.3 ms  ✓ StructArrays</span></span>
<span class="line"><span>    602.1 ms  ✓ SparseInverseSubset</span></span>
<span class="line"><span>    414.6 ms  ✓ StructArrays → StructArraysAdaptExt</span></span>
<span class="line"><span>    399.2 ms  ✓ StructArrays → StructArraysGPUArraysCoreExt</span></span>
<span class="line"><span>    723.8 ms  ✓ StructArrays → StructArraysSparseArraysExt</span></span>
<span class="line"><span>   5291.4 ms  ✓ ChainRules</span></span>
<span class="line"><span>  34392.4 ms  ✓ Zygote</span></span>
<span class="line"><span>  10 dependencies successfully precompiled in 42 seconds. 76 already precompiled.</span></span>
<span class="line"><span>Precompiling AccessorsStructArraysExt...</span></span>
<span class="line"><span>    469.5 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 16 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    496.1 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 22 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysStaticArraysExt...</span></span>
<span class="line"><span>    655.8 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    823.9 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 39 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    856.4 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1656.4 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   2881.0 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 163 already precompiled.</span></span>
<span class="line"><span>Precompiling ZygoteColorsExt...</span></span>
<span class="line"><span>   1830.7 ms  ✓ Zygote → ZygoteColorsExt</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.4.2/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.4.2/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.4.2/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.4.2/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61762</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60114</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56895</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53797</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51823</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49943</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49180</span></span>
<span class="line"><span>Validation: Loss 0.46457 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46646 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47841</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45443</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43614</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42116</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40509</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40059</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40221</span></span>
<span class="line"><span>Validation: Loss 0.36677 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36869 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36301</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36096</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35169</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33348</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32519</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30763</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29216</span></span>
<span class="line"><span>Validation: Loss 0.28121 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28338 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28277</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27310</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26352</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26458</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24668</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22651</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24597</span></span>
<span class="line"><span>Validation: Loss 0.21097 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21336 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21921</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21288</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20089</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18622</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18467</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17351</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15245</span></span>
<span class="line"><span>Validation: Loss 0.15547 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15796 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14904</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16017</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14675</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13241</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13204</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13922</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15164</span></span>
<span class="line"><span>Validation: Loss 0.11348 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11579 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.13264</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10320</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11710</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09475</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09289</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09447</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08533</span></span>
<span class="line"><span>Validation: Loss 0.08112 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08292 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08555</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07707</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08340</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07285</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06320</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06598</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06602</span></span>
<span class="line"><span>Validation: Loss 0.05668 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05789 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05925</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05730</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05547</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04906</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04984</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04445</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04935</span></span>
<span class="line"><span>Validation: Loss 0.04204 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04284 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04265</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04230</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03959</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04091</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03845</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03706</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03961</span></span>
<span class="line"><span>Validation: Loss 0.03397 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03461 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03564</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03114</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03468</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03287</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03233</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03160</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03439</span></span>
<span class="line"><span>Validation: Loss 0.02878 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02934 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03170</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02814</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02901</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02843</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02558</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02779</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02728</span></span>
<span class="line"><span>Validation: Loss 0.02501 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02550 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02457</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02605</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02644</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02472</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02430</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02809</span></span>
<span class="line"><span>Validation: Loss 0.02211 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02255 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02093</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02390</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02233</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02314</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01939</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02298</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02155</span></span>
<span class="line"><span>Validation: Loss 0.01978 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02018 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02103</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01929</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01901</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02047</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01989</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01744</span></span>
<span class="line"><span>Validation: Loss 0.01785 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01822 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01937</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01578</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01914</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01807</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01926</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01729</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01395</span></span>
<span class="line"><span>Validation: Loss 0.01622 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01657 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01746</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01738</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01625</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01603</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01549</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01443</span></span>
<span class="line"><span>Validation: Loss 0.01483 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01515 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01657</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01632</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01495</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01367</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01487</span></span>
<span class="line"><span>Validation: Loss 0.01363 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01393 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01388</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01490</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01303</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01406</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01311</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01528</span></span>
<span class="line"><span>Validation: Loss 0.01258 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01286 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01283</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01356</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01267</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01240</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01299</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01261</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01173</span></span>
<span class="line"><span>Validation: Loss 0.01163 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01190 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01220</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01300</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01217</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01087</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00953</span></span>
<span class="line"><span>Validation: Loss 0.01072 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01117</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01109</span></span>
<span class="line"><span>Validation: Loss 0.00977 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01000 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00957</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01016</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00937</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00884</span></span>
<span class="line"><span>Validation: Loss 0.00870 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00890 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00774</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00875</span></span>
<span class="line"><span>Validation: Loss 0.00775 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00792 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00770</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00747</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00715</span></span>
<span class="line"><span>Validation: Loss 0.00707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00722 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62686</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60238</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56560</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54008</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51887</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50023</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47535</span></span>
<span class="line"><span>Validation: Loss 0.46081 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46424 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46765</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45766</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44123</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43015</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41003</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39269</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39828</span></span>
<span class="line"><span>Validation: Loss 0.36128 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36546 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36435</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35550</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34530</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33401</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32831</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31457</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29039</span></span>
<span class="line"><span>Validation: Loss 0.27475 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27942 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28417</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27505</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25364</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26571</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24322</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24181</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20724</span></span>
<span class="line"><span>Validation: Loss 0.20460 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20933 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21660</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19363</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20366</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19392</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18396</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17375</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18596</span></span>
<span class="line"><span>Validation: Loss 0.15007 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15442 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16112</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15323</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15114</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14477</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13492</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12403</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10891</span></span>
<span class="line"><span>Validation: Loss 0.10897 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11253 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11893</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11821</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10115</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09920</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09080</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09665</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10272</span></span>
<span class="line"><span>Validation: Loss 0.07781 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08046 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08218</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07534</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07830</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07305</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06916</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06497</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06468</span></span>
<span class="line"><span>Validation: Loss 0.05437 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05613 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06475</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06051</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05020</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04801</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04612</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04395</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05028</span></span>
<span class="line"><span>Validation: Loss 0.04056 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04178 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04429</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04477</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04020</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04198</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03483</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03405</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04023</span></span>
<span class="line"><span>Validation: Loss 0.03280 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03449</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03468</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03225</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03172</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03406</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03108</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03117</span></span>
<span class="line"><span>Validation: Loss 0.02780 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02865 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02827</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03079</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03040</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02774</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02696</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02599</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02700</span></span>
<span class="line"><span>Validation: Loss 0.02417 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02492 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02566</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02585</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02599</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02453</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02391</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02329</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02308</span></span>
<span class="line"><span>Validation: Loss 0.02136 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02203 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02349</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02217</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02085</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02163</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02188</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02212</span></span>
<span class="line"><span>Validation: Loss 0.01911 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01973 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02175</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01884</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02095</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02015</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02003</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01742</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01929</span></span>
<span class="line"><span>Validation: Loss 0.01724 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01781 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01806</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01797</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01884</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01874</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01646</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01731</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01938</span></span>
<span class="line"><span>Validation: Loss 0.01565 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01618 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01657</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01726</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01539</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01578</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01659</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01681</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01549</span></span>
<span class="line"><span>Validation: Loss 0.01429 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01478 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01527</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01676</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01560</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01445</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01412</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01450</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01208</span></span>
<span class="line"><span>Validation: Loss 0.01311 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01357 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01416</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01384</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01432</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01468</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01312</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01259</span></span>
<span class="line"><span>Validation: Loss 0.01209 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01252 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01401</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01288</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01131</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01397</span></span>
<span class="line"><span>Validation: Loss 0.01118 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01158 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01163</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01206</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01201</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01100</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01461</span></span>
<span class="line"><span>Validation: Loss 0.01030 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01067 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01047</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01100</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01245</span></span>
<span class="line"><span>Validation: Loss 0.00935 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00968 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00876</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00960</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00938</span></span>
<span class="line"><span>Validation: Loss 0.00831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00860 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00848</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00884</span></span>
<span class="line"><span>Validation: Loss 0.00744 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00768 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00798</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00739</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00707</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00804</span></span>
<span class="line"><span>Validation: Loss 0.00680 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00701 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.141 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46)]))}const E=a(l,[["render",e]]);export{d as __pageData,E as default};
