import{_ as a,c as n,a2 as p,o as i}from"./chunks/framework.DPiAi8YZ.js";const d=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(t,s,c,h,o,k){return i(),n("div",null,s[0]||(s[0]=[p(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    494.7 ms  ✓ Requires</span></span>
<span class="line"><span>    592.3 ms  ✓ CpuId</span></span>
<span class="line"><span>    505.2 ms  ✓ Compat</span></span>
<span class="line"><span>    590.8 ms  ✓ DocStringExtensions</span></span>
<span class="line"><span>    450.7 ms  ✓ JLLWrappers</span></span>
<span class="line"><span>    758.0 ms  ✓ Static</span></span>
<span class="line"><span>   2445.0 ms  ✓ MacroTools</span></span>
<span class="line"><span>    361.9 ms  ✓ Compat → CompatLinearAlgebraExt</span></span>
<span class="line"><span>   1023.6 ms  ✓ LazyArtifacts</span></span>
<span class="line"><span>    564.8 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    582.4 ms  ✓ Hwloc_jll</span></span>
<span class="line"><span>    582.9 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>    385.4 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>    627.8 ms  ✓ CommonSubexpressions</span></span>
<span class="line"><span>    984.2 ms  ✓ CPUSummary</span></span>
<span class="line"><span>   1512.2 ms  ✓ DispatchDoctor</span></span>
<span class="line"><span>   1426.8 ms  ✓ Setfield</span></span>
<span class="line"><span>    569.8 ms  ✓ Functors</span></span>
<span class="line"><span>   1175.6 ms  ✓ ChainRulesCore</span></span>
<span class="line"><span>   1452.8 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>   7339.0 ms  ✓ StaticArrays</span></span>
<span class="line"><span>   1365.9 ms  ✓ LLVMExtra_jll</span></span>
<span class="line"><span>    618.4 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>    407.5 ms  ✓ DispatchDoctor → DispatchDoctorEnzymeCoreExt</span></span>
<span class="line"><span>   2118.5 ms  ✓ Hwloc</span></span>
<span class="line"><span>   1171.6 ms  ✓ LuxCore</span></span>
<span class="line"><span>   2559.2 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    804.5 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>    388.6 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    403.0 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>   1129.8 ms  ✓ Optimisers</span></span>
<span class="line"><span>    719.6 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>    466.0 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>   1344.8 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    569.5 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    623.9 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    648.5 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    587.6 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    567.9 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    648.9 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    575.2 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>    497.4 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    490.1 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    460.6 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>   1700.5 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>   2672.5 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    665.7 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>    467.8 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    415.0 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    924.8 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>   6723.2 ms  ✓ LLVM</span></span>
<span class="line"><span>   3819.3 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    901.3 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    829.1 ms  ✓ Polyester</span></span>
<span class="line"><span>    884.8 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   1972.7 ms  ✓ UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   4835.6 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>   1544.1 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>   1605.3 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   6109.4 ms  ✓ NNlib</span></span>
<span class="line"><span>   1740.8 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>   1747.1 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   6504.7 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9941.7 ms  ✓ Lux</span></span>
<span class="line"><span>  64 dependencies successfully precompiled in 53 seconds. 59 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>    359.9 ms  ✓ Scratch</span></span>
<span class="line"><span>   1259.3 ms  ✓ SentinelArrays</span></span>
<span class="line"><span>    899.0 ms  ✓ CUDA_Driver_jll</span></span>
<span class="line"><span>   2662.4 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>    573.5 ms  ✓ NVTX_jll</span></span>
<span class="line"><span>    560.5 ms  ✓ demumble_jll</span></span>
<span class="line"><span>    539.1 ms  ✓ JuliaNVTXCallbacks_jll</span></span>
<span class="line"><span>   3676.6 ms  ✓ Test</span></span>
<span class="line"><span>   1688.9 ms  ✓ DataStructures</span></span>
<span class="line"><span>   1974.1 ms  ✓ StringManipulation</span></span>
<span class="line"><span>   2190.2 ms  ✓ GPUArrays</span></span>
<span class="line"><span>   1857.7 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   2763.2 ms  ✓ CUDA_Runtime_jll</span></span>
<span class="line"><span>   1309.9 ms  ✓ NVTX</span></span>
<span class="line"><span>    499.5 ms  ✓ BFloat16s</span></span>
<span class="line"><span>    483.2 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>   1314.4 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>   2239.2 ms  ✓ CUDNN_jll</span></span>
<span class="line"><span>   1383.0 ms  ✓ LLVM → BFloat16sExt</span></span>
<span class="line"><span>  20121.5 ms  ✓ PrettyTables</span></span>
<span class="line"><span>  27228.2 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>  45372.2 ms  ✓ DataFrames</span></span>
<span class="line"><span>  52700.1 ms  ✓ CUDA</span></span>
<span class="line"><span>   5660.8 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8816.0 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5889.2 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  26 dependencies successfully precompiled in 148 seconds. 74 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesGPUArraysExt...</span></span>
<span class="line"><span>   1532.8 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 42 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersGPUArraysExt...</span></span>
<span class="line"><span>   1509.4 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling ChainRulesCoreSparseArraysExt...</span></span>
<span class="line"><span>    631.9 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 11 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    668.5 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 17 already precompiled.</span></span>
<span class="line"><span>Precompiling AbstractFFTsChainRulesCoreExt...</span></span>
<span class="line"><span>    416.4 ms  ✓ AbstractFFTs → AbstractFFTsChainRulesCoreExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 0 seconds. 9 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   5096.8 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   5305.3 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5643.7 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   5099.6 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5597.0 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5646.9 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   6313.4 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 7 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   5658.6 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5694.9 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5434.0 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 107 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   6345.7 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 174 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>   4466.5 ms  ✓ FileIO</span></span>
<span class="line"><span>  33349.5 ms  ✓ JLD2</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 38 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    390.2 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>    425.5 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>   1161.5 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>    589.7 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>   1125.0 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    428.9 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    588.0 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>   2984.0 ms  ✓ Accessors</span></span>
<span class="line"><span>    705.6 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>    831.2 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>   2316.0 ms  ✓ StatsBase</span></span>
<span class="line"><span>    795.3 ms  ✓ BangBang</span></span>
<span class="line"><span>    706.4 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>    501.7 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    520.5 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    720.2 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    841.3 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2704.9 ms  ✓ Transducers</span></span>
<span class="line"><span>    630.5 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   5113.2 ms  ✓ FLoops</span></span>
<span class="line"><span>   7489.8 ms  ✓ MLUtils</span></span>
<span class="line"><span>  21 dependencies successfully precompiled in 21 seconds. 91 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangDataFramesExt...</span></span>
<span class="line"><span>   1621.7 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling TransducersDataFramesExt...</span></span>
<span class="line"><span>   1378.5 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 61 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   2525.6 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 116 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   3234.6 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 178 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    948.7 ms  ✓ ZygoteRules</span></span>
<span class="line"><span>   1871.1 ms  ✓ IRTools</span></span>
<span class="line"><span>   5590.1 ms  ✓ ChainRules</span></span>
<span class="line"><span>  34780.0 ms  ✓ Zygote</span></span>
<span class="line"><span>  4 dependencies successfully precompiled in 41 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling AccessorsStructArraysExt...</span></span>
<span class="line"><span>    507.2 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 16 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    484.3 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 22 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysStaticArraysExt...</span></span>
<span class="line"><span>    654.9 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    770.0 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 39 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    819.8 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    447.6 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 0 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1650.1 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   3497.9 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 162 already precompiled.</span></span>
<span class="line"><span>Precompiling ZygoteColorsExt...</span></span>
<span class="line"><span>   1776.1 ms  ✓ Zygote → ZygoteColorsExt</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.4.1/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.4.1/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.4.1/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.4.1/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61367</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59712</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56119</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54298</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51899</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50466</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47137</span></span>
<span class="line"><span>Validation: Loss 0.46711 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47555 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46574</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45492</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43776</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42556</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41798</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38898</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39934</span></span>
<span class="line"><span>Validation: Loss 0.36997 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37992 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37373</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36598</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33692</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32881</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30676</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31733</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32936</span></span>
<span class="line"><span>Validation: Loss 0.28536 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29610 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28156</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27879</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25902</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25110</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25401</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23588</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23325</span></span>
<span class="line"><span>Validation: Loss 0.21578 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22649 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22262</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21548</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18405</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18794</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19170</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17481</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15956</span></span>
<span class="line"><span>Validation: Loss 0.16018 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17008 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16037</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14633</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13868</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15006</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13592</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13859</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11643</span></span>
<span class="line"><span>Validation: Loss 0.11760 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12588 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11172</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11035</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10630</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11381</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10564</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07393</span></span>
<span class="line"><span>Validation: Loss 0.08442 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09060 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08092</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08684</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07771</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07386</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06715</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06483</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06035</span></span>
<span class="line"><span>Validation: Loss 0.05895 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06310 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05952</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05917</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05295</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05165</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05267</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04426</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03650</span></span>
<span class="line"><span>Validation: Loss 0.04361 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04654 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04547</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04390</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04146</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04115</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03283</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03817</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03660</span></span>
<span class="line"><span>Validation: Loss 0.03522 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03762 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03587</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03404</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03433</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03436</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03085</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03042</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03078</span></span>
<span class="line"><span>Validation: Loss 0.02986 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03195 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03198</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02974</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02867</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02781</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02750</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02554</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02669</span></span>
<span class="line"><span>Validation: Loss 0.02595 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02781 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02551</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02479</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02459</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02633</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02437</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02461</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02105</span></span>
<span class="line"><span>Validation: Loss 0.02294 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02461 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02234</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02175</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02281</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02226</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02327</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02040</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02104</span></span>
<span class="line"><span>Validation: Loss 0.02052 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02205 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01976</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01968</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02029</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01943</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02006</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01961</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02037</span></span>
<span class="line"><span>Validation: Loss 0.01851 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01993 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01660</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01825</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01938</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01776</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01784</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01812</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01656</span></span>
<span class="line"><span>Validation: Loss 0.01681 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01812 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01682</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01837</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01527</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01523</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01684</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01571</span></span>
<span class="line"><span>Validation: Loss 0.01537 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01659 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01446</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01456</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01542</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01393</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01548</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01444</span></span>
<span class="line"><span>Validation: Loss 0.01414 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01527 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01414</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01495</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01299</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01421</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01391</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01544</span></span>
<span class="line"><span>Validation: Loss 0.01306 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01412 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01372</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01355</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01263</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01175</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01186</span></span>
<span class="line"><span>Validation: Loss 0.01209 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01307 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01219</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01205</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01180</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01099</span></span>
<span class="line"><span>Validation: Loss 0.01119 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01209 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01161</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01181</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01134</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01042</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01010</span></span>
<span class="line"><span>Validation: Loss 0.01026 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01107 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01015</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01016</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01036</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01054</span></span>
<span class="line"><span>Validation: Loss 0.00921 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00992 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00895</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00897</span></span>
<span class="line"><span>Validation: Loss 0.00816 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00877 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00837</span></span>
<span class="line"><span>Validation: Loss 0.00738 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00791 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63487</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60560</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56825</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53385</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51626</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50272</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48309</span></span>
<span class="line"><span>Validation: Loss 0.46197 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45618 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47915</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46448</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45141</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42117</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41135</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38516</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37026</span></span>
<span class="line"><span>Validation: Loss 0.36330 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35627 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37922</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35917</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34686</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32908</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32395</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31569</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28576</span></span>
<span class="line"><span>Validation: Loss 0.27709 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26881 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.30241</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26503</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26196</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26078</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24990</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23581</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21976</span></span>
<span class="line"><span>Validation: Loss 0.20740 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.19881 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22857</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21933</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20009</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18151</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17477</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18024</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18392</span></span>
<span class="line"><span>Validation: Loss 0.15292 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14489 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15658</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15041</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15387</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14809</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13858</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13707</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11415</span></span>
<span class="line"><span>Validation: Loss 0.11155 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10480 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11169</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11783</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09649</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11305</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10724</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09856</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08203</span></span>
<span class="line"><span>Validation: Loss 0.07984 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07474 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08615</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08119</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07777</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07642</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06875</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06414</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07095</span></span>
<span class="line"><span>Validation: Loss 0.05577 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05238 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06346</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05796</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05713</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05191</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04658</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04425</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04977</span></span>
<span class="line"><span>Validation: Loss 0.04153 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03919 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04668</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04291</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04354</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03695</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03875</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03698</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03998</span></span>
<span class="line"><span>Validation: Loss 0.03359 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03171 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03982</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03447</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03164</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03227</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03393</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03081</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03278</span></span>
<span class="line"><span>Validation: Loss 0.02848 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02685 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03180</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03115</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02951</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02772</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02839</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02640</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02476</span></span>
<span class="line"><span>Validation: Loss 0.02476 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02332 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02545</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02697</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02738</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02455</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02388</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02414</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02516</span></span>
<span class="line"><span>Validation: Loss 0.02190 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02060 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02352</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02257</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02298</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02145</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02305</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02212</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02179</span></span>
<span class="line"><span>Validation: Loss 0.01961 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01842 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01987</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02049</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02108</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01990</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02014</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01900</span></span>
<span class="line"><span>Validation: Loss 0.01770 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01660 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01941</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01881</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01938</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01864</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01674</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01716</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01982</span></span>
<span class="line"><span>Validation: Loss 0.01608 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01506 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01845</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01507</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01690</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01715</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01617</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01686</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01696</span></span>
<span class="line"><span>Validation: Loss 0.01468 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01373 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01689</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01580</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01489</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01557</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01535</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01435</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01281</span></span>
<span class="line"><span>Validation: Loss 0.01345 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01256 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01509</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01547</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01367</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01350</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01073</span></span>
<span class="line"><span>Validation: Loss 0.01237 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01153 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01276</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01374</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01321</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01267</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01078</span></span>
<span class="line"><span>Validation: Loss 0.01137 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01059 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01278</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01214</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01210</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01101</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01205</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01113</span></span>
<span class="line"><span>Validation: Loss 0.01033 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00961 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01214</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01091</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00948</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00916</span></span>
<span class="line"><span>Validation: Loss 0.00919 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00856 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01018</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00905</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00878</span></span>
<span class="line"><span>Validation: Loss 0.00817 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00764 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00795</span></span>
<span class="line"><span>Validation: Loss 0.00743 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00697 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00734</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00800</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00736</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00760</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00761</span></span>
<span class="line"><span>Validation: Loss 0.00688 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00647 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>2 devices:</span></span>
<span class="line"><span>  0: Quadro RTX 5000 (sm_75, 14.883 GiB / 16.000 GiB available)</span></span>
<span class="line"><span>  1: Quadro RTX 5000 (sm_75, 15.549 GiB / 16.000 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46)]))}const E=a(l,[["render",e]]);export{d as __pageData,E as default};