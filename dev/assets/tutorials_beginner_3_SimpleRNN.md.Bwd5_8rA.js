import{_ as a,c as n,a2 as p,o as i}from"./chunks/framework.bV3h_rQg.js";const E=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(t,s,c,h,o,k){return i(),n("div",null,s[0]||(s[0]=[p(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    464.1 ms  ✓ ConstructionBase</span></span>
<span class="line"><span>    526.0 ms  ✓ CompilerSupportLibraries_jll</span></span>
<span class="line"><span>    470.4 ms  ✓ Requires</span></span>
<span class="line"><span>    529.6 ms  ✓ Compat</span></span>
<span class="line"><span>    595.7 ms  ✓ DocStringExtensions</span></span>
<span class="line"><span>    370.1 ms  ✓ ConstructionBase → ConstructionBaseLinearAlgebraExt</span></span>
<span class="line"><span>    367.3 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>    408.1 ms  ✓ Adapt</span></span>
<span class="line"><span>    671.0 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>    384.2 ms  ✓ Compat → CompatLinearAlgebraExt</span></span>
<span class="line"><span>    600.6 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    481.7 ms  ✓ GPUArraysCore</span></span>
<span class="line"><span>    450.3 ms  ✓ EnzymeCore → AdaptExt</span></span>
<span class="line"><span>   1499.6 ms  ✓ Setfield</span></span>
<span class="line"><span>   1259.0 ms  ✓ LuxCore</span></span>
<span class="line"><span>   1211.0 ms  ✓ ChainRulesCore</span></span>
<span class="line"><span>    610.7 ms  ✓ Functors</span></span>
<span class="line"><span>   1554.5 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>    384.7 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>    460.0 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    449.6 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>   2611.3 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>   7664.9 ms  ✓ StaticArrays</span></span>
<span class="line"><span>    416.5 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    388.3 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>    630.1 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>    588.3 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>   1262.5 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    814.6 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>    467.1 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    482.1 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>    566.5 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>   1090.7 ms  ✓ Optimisers</span></span>
<span class="line"><span>   1817.4 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    625.9 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    607.3 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>   2964.7 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    599.6 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    615.9 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    647.7 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    631.0 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>   3792.8 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    482.0 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    416.2 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    429.3 ms  ✓ Optimisers → OptimisersAdaptExt</span></span>
<span class="line"><span>   1009.9 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>    969.0 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    897.7 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>    777.8 ms  ✓ Polyester</span></span>
<span class="line"><span>   3968.8 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    660.6 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    722.4 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5463.0 ms  ✓ NNlib</span></span>
<span class="line"><span>    927.8 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    928.7 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5922.3 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9511.8 ms  ✓ Lux</span></span>
<span class="line"><span>  57 dependencies successfully precompiled in 42 seconds. 63 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>   1050.2 ms  ✓ LazyArtifacts</span></span>
<span class="line"><span>   1680.3 ms  ✓ DataStructures</span></span>
<span class="line"><span>    969.4 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   1508.5 ms  ✓ LLVMExtra_jll</span></span>
<span class="line"><span>    525.1 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>   3857.1 ms  ✓ Test</span></span>
<span class="line"><span>   2534.3 ms  ✓ CUDA_Runtime_jll</span></span>
<span class="line"><span>   1453.2 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>   2066.8 ms  ✓ CUDNN_jll</span></span>
<span class="line"><span>   6827.9 ms  ✓ LLVM</span></span>
<span class="line"><span>   1293.9 ms  ✓ LLVM → BFloat16sExt</span></span>
<span class="line"><span>   1927.9 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   2201.6 ms  ✓ GPUArrays</span></span>
<span class="line"><span>  27962.5 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>  48535.7 ms  ✓ DataFrames</span></span>
<span class="line"><span>  52616.5 ms  ✓ CUDA</span></span>
<span class="line"><span>   5172.8 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8346.8 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5333.9 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  19 dependencies successfully precompiled in 124 seconds. 81 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesGPUArraysExt...</span></span>
<span class="line"><span>   1640.9 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 42 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersGPUArraysExt...</span></span>
<span class="line"><span>   1447.6 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling ChainRulesCoreSparseArraysExt...</span></span>
<span class="line"><span>    607.8 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 11 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    651.0 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 17 already precompiled.</span></span>
<span class="line"><span>Precompiling AbstractFFTsChainRulesCoreExt...</span></span>
<span class="line"><span>    410.0 ms  ✓ AbstractFFTs → AbstractFFTsChainRulesCoreExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 0 seconds. 9 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   5143.8 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   5092.4 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5755.8 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   5073.3 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5303.9 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5456.4 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   5897.4 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 6 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   4971.9 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5516.2 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5071.4 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 107 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   6066.7 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 174 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>   4911.2 ms  ✓ FileIO</span></span>
<span class="line"><span>  34147.5 ms  ✓ JLD2</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 39 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    424.2 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>    598.6 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>   1186.2 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    439.2 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    608.8 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>   3029.9 ms  ✓ Accessors</span></span>
<span class="line"><span>   2335.1 ms  ✓ StatsBase</span></span>
<span class="line"><span>    636.5 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>    782.1 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>    797.6 ms  ✓ BangBang</span></span>
<span class="line"><span>    763.4 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>    535.9 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    503.6 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    787.9 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    894.9 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2783.4 ms  ✓ Transducers</span></span>
<span class="line"><span>    650.7 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   5148.1 ms  ✓ FLoops</span></span>
<span class="line"><span>   6621.2 ms  ✓ MLUtils</span></span>
<span class="line"><span>  19 dependencies successfully precompiled in 21 seconds. 79 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangDataFramesExt...</span></span>
<span class="line"><span>   1767.2 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling TransducersDataFramesExt...</span></span>
<span class="line"><span>   1419.9 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 61 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1714.6 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2503.7 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 177 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    759.9 ms  ✓ StructArrays</span></span>
<span class="line"><span>    957.2 ms  ✓ ZygoteRules</span></span>
<span class="line"><span>    404.0 ms  ✓ StructArrays → StructArraysAdaptExt</span></span>
<span class="line"><span>    389.6 ms  ✓ StructArrays → StructArraysGPUArraysCoreExt</span></span>
<span class="line"><span>    671.3 ms  ✓ StructArrays → StructArraysSparseArraysExt</span></span>
<span class="line"><span>   5464.8 ms  ✓ ChainRules</span></span>
<span class="line"><span>  33991.2 ms  ✓ Zygote</span></span>
<span class="line"><span>  7 dependencies successfully precompiled in 41 seconds. 79 already precompiled.</span></span>
<span class="line"><span>Precompiling AccessorsStructArraysExt...</span></span>
<span class="line"><span>    487.7 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 16 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    503.6 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 22 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysStaticArraysExt...</span></span>
<span class="line"><span>    744.3 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    796.7 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 39 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    878.2 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    453.5 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1663.3 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   2853.9 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 163 already precompiled.</span></span>
<span class="line"><span>Precompiling ZygoteColorsExt...</span></span>
<span class="line"><span>   1800.2 ms  ✓ Zygote → ZygoteColorsExt</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62727</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60150</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55754</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54961</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50978</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49356</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49213</span></span>
<span class="line"><span>Validation: Loss 0.46914 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47074 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46237</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45398</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44432</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42042</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40731</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40675</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39426</span></span>
<span class="line"><span>Validation: Loss 0.37195 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37402 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37036</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36639</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34473</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33489</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30719</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31311</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29720</span></span>
<span class="line"><span>Validation: Loss 0.28692 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28919 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29304</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27029</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25164</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26682</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25018</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22840</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21862</span></span>
<span class="line"><span>Validation: Loss 0.21707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21925 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21588</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21504</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18021</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19482</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17010</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18924</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18921</span></span>
<span class="line"><span>Validation: Loss 0.16175 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16362 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16631</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14671</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15048</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14495</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13559</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12377</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12311</span></span>
<span class="line"><span>Validation: Loss 0.11875 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12019 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11212</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12117</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10955</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10558</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09429</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08580</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10236</span></span>
<span class="line"><span>Validation: Loss 0.08497 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08595 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08768</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08320</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07575</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06915</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06558</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06282</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07221</span></span>
<span class="line"><span>Validation: Loss 0.05905 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05968 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05978</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05550</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05476</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05397</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04562</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04548</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04377</span></span>
<span class="line"><span>Validation: Loss 0.04394 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04436 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04655</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04221</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03888</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03932</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03534</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03853</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03993</span></span>
<span class="line"><span>Validation: Loss 0.03563 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03594 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03512</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03306</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03760</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03093</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03212</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03034</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03169</span></span>
<span class="line"><span>Validation: Loss 0.03027 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03052 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03005</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03192</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02981</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02658</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02723</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02576</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02574</span></span>
<span class="line"><span>Validation: Loss 0.02635 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02658 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02525</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02635</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02613</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02285</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02523</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02394</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02355</span></span>
<span class="line"><span>Validation: Loss 0.02334 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02354 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02195</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02442</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02384</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02089</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02230</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01986</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02105</span></span>
<span class="line"><span>Validation: Loss 0.02092 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02110 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02039</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01952</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01943</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02076</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01851</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02144</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01762</span></span>
<span class="line"><span>Validation: Loss 0.01892 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01908 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01703</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01868</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01775</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01819</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01712</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01903</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01988</span></span>
<span class="line"><span>Validation: Loss 0.01722 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01737 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01660</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01702</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01658</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01622</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01601</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01518</span></span>
<span class="line"><span>Validation: Loss 0.01575 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01589 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01547</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01691</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01522</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01476</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01415</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01498</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01114</span></span>
<span class="line"><span>Validation: Loss 0.01448 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01461 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01351</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01434</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01416</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01359</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01437</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01289</span></span>
<span class="line"><span>Validation: Loss 0.01338 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01350 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01354</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01300</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01373</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01231</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01095</span></span>
<span class="line"><span>Validation: Loss 0.01240 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01251 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01181</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01243</span></span>
<span class="line"><span>Validation: Loss 0.01148 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01158 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01113</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01175</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01018</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00891</span></span>
<span class="line"><span>Validation: Loss 0.01051 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01059 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00996</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00986</span></span>
<span class="line"><span>Validation: Loss 0.00943 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00950 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00955</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00892</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00910</span></span>
<span class="line"><span>Validation: Loss 0.00836 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00843 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00730</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00813</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00774</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00678</span></span>
<span class="line"><span>Validation: Loss 0.00756 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00763 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62424</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58557</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56721</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54066</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52147</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49344</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50406</span></span>
<span class="line"><span>Validation: Loss 0.46871 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47191 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47691</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46097</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43476</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42286</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40224</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39710</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39913</span></span>
<span class="line"><span>Validation: Loss 0.37228 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37600 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37990</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36060</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34268</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33173</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31479</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31301</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29823</span></span>
<span class="line"><span>Validation: Loss 0.28788 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29212 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28230</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27692</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26398</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27033</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25083</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22489</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22144</span></span>
<span class="line"><span>Validation: Loss 0.21832 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22278 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20867</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19871</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19334</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19922</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19808</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18154</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17430</span></span>
<span class="line"><span>Validation: Loss 0.16299 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16727 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15789</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15995</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15070</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14181</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13928</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12860</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12104</span></span>
<span class="line"><span>Validation: Loss 0.12003 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12365 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10940</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12387</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11054</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10335</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09887</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08885</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11397</span></span>
<span class="line"><span>Validation: Loss 0.08627 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08900 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08594</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07459</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08062</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07198</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07121</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06840</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06767</span></span>
<span class="line"><span>Validation: Loss 0.05999 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06182 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06609</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05496</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05062</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05094</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04798</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04956</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04441</span></span>
<span class="line"><span>Validation: Loss 0.04418 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04545 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04491</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04616</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03752</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04320</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03841</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03358</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03772</span></span>
<span class="line"><span>Validation: Loss 0.03562 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03665 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03560</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03787</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03271</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03388</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02927</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03097</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03189</span></span>
<span class="line"><span>Validation: Loss 0.03019 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03109 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03171</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03333</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02768</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02713</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02588</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02610</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02673</span></span>
<span class="line"><span>Validation: Loss 0.02625 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02705 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02488</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02375</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02397</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02639</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02486</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02551</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02617</span></span>
<span class="line"><span>Validation: Loss 0.02323 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02396 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02510</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02344</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02145</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02310</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01987</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02340</span></span>
<span class="line"><span>Validation: Loss 0.02077 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02144 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02209</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02326</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01711</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01867</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01872</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01618</span></span>
<span class="line"><span>Validation: Loss 0.01874 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01935 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01960</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01658</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01910</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01797</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01759</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01733</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01791</span></span>
<span class="line"><span>Validation: Loss 0.01705 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01762 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01704</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01618</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01599</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01752</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01604</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01598</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01578</span></span>
<span class="line"><span>Validation: Loss 0.01561 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01614 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01636</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01496</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01552</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01475</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01561</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01339</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01453</span></span>
<span class="line"><span>Validation: Loss 0.01436 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01486 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01323</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01403</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01372</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01425</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01343</span></span>
<span class="line"><span>Validation: Loss 0.01328 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01375 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01341</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01313</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01208</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01094</span></span>
<span class="line"><span>Validation: Loss 0.01232 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01275 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01229</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01291</span></span>
<span class="line"><span>Validation: Loss 0.01142 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01182 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01042</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01102</span></span>
<span class="line"><span>Validation: Loss 0.01048 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01084 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01057</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01015</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00992</span></span>
<span class="line"><span>Validation: Loss 0.00941 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00973 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00948</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00846</span></span>
<span class="line"><span>Validation: Loss 0.00833 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00860 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00708</span></span>
<span class="line"><span>Validation: Loss 0.00751 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00775 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.047 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46)]))}const d=a(l,[["render",e]]);export{E as __pageData,d as default};
