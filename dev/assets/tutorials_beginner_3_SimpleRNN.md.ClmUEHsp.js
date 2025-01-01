import{_ as a,c as n,a2 as p,o as i}from"./chunks/framework.bV3h_rQg.js";const d=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(t,s,c,h,o,k){return i(),n("div",null,s[0]||(s[0]=[p(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling ADTypes...</span></span>
<span class="line"><span>    605.9 ms  ✓ ADTypes</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds</span></span>
<span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    318.4 ms  ✓ ConcreteStructs</span></span>
<span class="line"><span>    337.3 ms  ✓ Future</span></span>
<span class="line"><span>    355.9 ms  ✓ CEnum</span></span>
<span class="line"><span>    364.2 ms  ✓ OpenLibm_jll</span></span>
<span class="line"><span>    367.3 ms  ✓ ArgCheck</span></span>
<span class="line"><span>    370.9 ms  ✓ ManualMemory</span></span>
<span class="line"><span>    295.1 ms  ✓ Reexport</span></span>
<span class="line"><span>    298.1 ms  ✓ SIMDTypes</span></span>
<span class="line"><span>    292.7 ms  ✓ IfElse</span></span>
<span class="line"><span>    322.6 ms  ✓ CommonWorldInvalidations</span></span>
<span class="line"><span>    838.8 ms  ✓ IrrationalConstants</span></span>
<span class="line"><span>    319.2 ms  ✓ FastClosures</span></span>
<span class="line"><span>    419.2 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>    475.4 ms  ✓ ADTypes → ADTypesEnzymeCoreExt</span></span>
<span class="line"><span>    353.6 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>    394.9 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    383.7 ms  ✓ NaNMath</span></span>
<span class="line"><span>    752.6 ms  ✓ ThreadingUtilities</span></span>
<span class="line"><span>    774.0 ms  ✓ Static</span></span>
<span class="line"><span>    597.8 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    380.3 ms  ✓ DiffResults</span></span>
<span class="line"><span>    350.3 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>   2101.3 ms  ✓ Hwloc</span></span>
<span class="line"><span>    395.0 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>   1410.9 ms  ✓ Setfield</span></span>
<span class="line"><span>   1015.1 ms  ✓ CPUSummary</span></span>
<span class="line"><span>   1507.9 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>   1314.5 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    456.6 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    637.4 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>    474.0 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>    587.6 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>   2455.3 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    923.6 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>   7200.5 ms  ✓ StaticArrays</span></span>
<span class="line"><span>   1686.2 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    608.9 ms  ✓ DiffRules</span></span>
<span class="line"><span>    756.0 ms  ✓ Polyester</span></span>
<span class="line"><span>    624.4 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    611.2 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>   2605.0 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    607.3 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    624.6 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    689.8 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    921.5 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>   3623.2 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>   3769.0 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    900.2 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>    664.4 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    767.7 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5069.2 ms  ✓ NNlib</span></span>
<span class="line"><span>    824.1 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    965.1 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5571.0 ms  ✓ LuxLib</span></span>
<span class="line"><span>   8967.1 ms  ✓ Lux</span></span>
<span class="line"><span>  55 dependencies successfully precompiled in 39 seconds. 54 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>    289.2 ms  ✓ IteratorInterfaceExtensions</span></span>
<span class="line"><span>    364.3 ms  ✓ ExprTools</span></span>
<span class="line"><span>    540.5 ms  ✓ Serialization</span></span>
<span class="line"><span>    417.8 ms  ✓ SuiteSparse_jll</span></span>
<span class="line"><span>    463.2 ms  ✓ OrderedCollections</span></span>
<span class="line"><span>    303.5 ms  ✓ DataValueInterfaces</span></span>
<span class="line"><span>    341.8 ms  ✓ DataAPI</span></span>
<span class="line"><span>    326.7 ms  ✓ TableTraits</span></span>
<span class="line"><span>   2598.5 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>   4781.7 ms  ✓ Colors</span></span>
<span class="line"><span>   6444.3 ms  ✓ LLVM</span></span>
<span class="line"><span>   3648.5 ms  ✓ Test</span></span>
<span class="line"><span>    454.9 ms  ✓ PooledArrays</span></span>
<span class="line"><span>    425.6 ms  ✓ Missings</span></span>
<span class="line"><span>   1640.3 ms  ✓ DataStructures</span></span>
<span class="line"><span>    793.5 ms  ✓ Tables</span></span>
<span class="line"><span>   3746.2 ms  ✓ SparseArrays</span></span>
<span class="line"><span>   1223.3 ms  ✓ NVTX</span></span>
<span class="line"><span>   1840.2 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   2147.6 ms  ✓ GPUArrays</span></span>
<span class="line"><span>   1347.9 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>   1306.0 ms  ✓ LLVM → BFloat16sExt</span></span>
<span class="line"><span>    510.5 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>    623.3 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>    935.1 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>  19304.3 ms  ✓ PrettyTables</span></span>
<span class="line"><span>  26087.6 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>  45481.1 ms  ✓ DataFrames</span></span>
<span class="line"><span>  51070.7 ms  ✓ CUDA</span></span>
<span class="line"><span>   4991.3 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8075.0 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5269.1 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  32 dependencies successfully precompiled in 147 seconds. 68 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesGPUArraysExt...</span></span>
<span class="line"><span>   1316.6 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 42 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersGPUArraysExt...</span></span>
<span class="line"><span>   1448.5 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceSparseArraysExt...</span></span>
<span class="line"><span>    586.1 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 7 already precompiled.</span></span>
<span class="line"><span>Precompiling ChainRulesCoreSparseArraysExt...</span></span>
<span class="line"><span>    620.2 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 11 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    657.4 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 17 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   4905.3 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   5080.9 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5327.0 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   4968.8 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5084.5 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5207.4 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   5685.4 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 6 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   5061.5 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5336.8 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5068.5 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 107 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   5728.8 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 174 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>  32437.2 ms  ✓ JLD2</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 33 seconds. 31 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    363.7 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    490.5 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    369.9 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    787.1 ms  ✓ InitialValues</span></span>
<span class="line"><span>    412.2 ms  ✓ ShowCases</span></span>
<span class="line"><span>    326.5 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    316.2 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    417.1 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>   1002.9 ms  ✓ Baselet</span></span>
<span class="line"><span>   1947.3 ms  ✓ Distributed</span></span>
<span class="line"><span>    404.5 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>   1110.6 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    587.3 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>    387.4 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>   2235.3 ms  ✓ StatsBase</span></span>
<span class="line"><span>    431.2 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    368.0 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>    380.4 ms  ✓ NameResolution</span></span>
<span class="line"><span>    567.7 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>   2694.5 ms  ✓ Accessors</span></span>
<span class="line"><span>    592.4 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>    782.6 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>    752.6 ms  ✓ BangBang</span></span>
<span class="line"><span>    687.0 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>    484.1 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    719.9 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    490.6 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    891.7 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2722.0 ms  ✓ Transducers</span></span>
<span class="line"><span>    635.6 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>  17069.6 ms  ✓ MLStyle</span></span>
<span class="line"><span>   4113.8 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   4862.7 ms  ✓ FLoops</span></span>
<span class="line"><span>   6132.7 ms  ✓ MLUtils</span></span>
<span class="line"><span>  34 dependencies successfully precompiled in 34 seconds. 64 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangDataFramesExt...</span></span>
<span class="line"><span>   1636.9 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling TransducersDataFramesExt...</span></span>
<span class="line"><span>   1387.5 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 61 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1576.7 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2176.0 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    563.5 ms  ✓ SuiteSparse</span></span>
<span class="line"><span>    722.3 ms  ✓ StructArrays</span></span>
<span class="line"><span>    878.2 ms  ✓ FillArrays</span></span>
<span class="line"><span>    399.3 ms  ✓ StructArrays → StructArraysAdaptExt</span></span>
<span class="line"><span>    595.9 ms  ✓ SparseInverseSubset</span></span>
<span class="line"><span>    383.7 ms  ✓ StructArrays → StructArraysGPUArraysCoreExt</span></span>
<span class="line"><span>    651.9 ms  ✓ StructArrays → StructArraysSparseArraysExt</span></span>
<span class="line"><span>    388.2 ms  ✓ FillArrays → FillArraysStatisticsExt</span></span>
<span class="line"><span>    643.9 ms  ✓ FillArrays → FillArraysSparseArraysExt</span></span>
<span class="line"><span>   5295.8 ms  ✓ ChainRules</span></span>
<span class="line"><span>  32675.3 ms  ✓ Zygote</span></span>
<span class="line"><span>  11 dependencies successfully precompiled in 40 seconds. 75 already precompiled.</span></span>
<span class="line"><span>Precompiling AccessorsStructArraysExt...</span></span>
<span class="line"><span>    455.3 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 0 seconds. 16 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    470.8 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 22 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysStaticArraysExt...</span></span>
<span class="line"><span>    666.8 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    764.6 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 39 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    811.7 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    438.3 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 0 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1601.9 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   2758.4 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 163 already precompiled.</span></span>
<span class="line"><span>Precompiling ZygoteColorsExt...</span></span>
<span class="line"><span>   1779.4 ms  ✓ Zygote → ZygoteColorsExt</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61947</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58840</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57115</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54652</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51605</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49667</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49551</span></span>
<span class="line"><span>Validation: Loss 0.46339 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47587 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46281</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45542</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43421</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43074</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41289</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40478</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38012</span></span>
<span class="line"><span>Validation: Loss 0.36490 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37981 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.38210</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36305</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34584</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32721</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31898</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29816</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30698</span></span>
<span class="line"><span>Validation: Loss 0.27908 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29537 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27455</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27138</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28030</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24884</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24420</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23533</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24672</span></span>
<span class="line"><span>Validation: Loss 0.20911 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22545 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22246</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20885</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19198</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19062</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17884</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18085</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15691</span></span>
<span class="line"><span>Validation: Loss 0.15409 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16879 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16490</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14352</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14819</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14508</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13229</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13080</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12302</span></span>
<span class="line"><span>Validation: Loss 0.11242 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12440 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11832</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11132</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09769</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10017</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10364</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09366</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10423</span></span>
<span class="line"><span>Validation: Loss 0.08048 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08926 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07981</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08701</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07457</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07463</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06838</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06641</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04497</span></span>
<span class="line"><span>Validation: Loss 0.05611 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06193 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06206</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05537</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04962</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05036</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04994</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04753</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04369</span></span>
<span class="line"><span>Validation: Loss 0.04175 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04596 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04566</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04096</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04132</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03847</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03827</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03658</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03733</span></span>
<span class="line"><span>Validation: Loss 0.03378 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03725 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03606</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03238</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03512</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03298</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03245</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02989</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03157</span></span>
<span class="line"><span>Validation: Loss 0.02863 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03165 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02937</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02998</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02891</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02953</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02753</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02561</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02558</span></span>
<span class="line"><span>Validation: Loss 0.02488 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02758 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02423</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02488</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02529</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02463</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02546</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02524</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02178</span></span>
<span class="line"><span>Validation: Loss 0.02200 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02444 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02161</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02202</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02228</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02464</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01986</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02182</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02318</span></span>
<span class="line"><span>Validation: Loss 0.01968 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02191 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01987</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02048</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02017</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01971</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01924</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01966</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01956</span></span>
<span class="line"><span>Validation: Loss 0.01775 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01980 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01962</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01733</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01858</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01521</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01792</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01914</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01777</span></span>
<span class="line"><span>Validation: Loss 0.01611 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01802 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01770</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01636</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01631</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01640</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01498</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01674</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01505</span></span>
<span class="line"><span>Validation: Loss 0.01471 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01649 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01531</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01527</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01578</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01475</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01441</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01380</span></span>
<span class="line"><span>Validation: Loss 0.01352 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01517 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01324</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01431</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01308</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01502</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01341</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01274</span></span>
<span class="line"><span>Validation: Loss 0.01248 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01403 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01351</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01271</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01157</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01229</span></span>
<span class="line"><span>Validation: Loss 0.01156 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01300 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01275</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01154</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01117</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01227</span></span>
<span class="line"><span>Validation: Loss 0.01069 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01201 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01123</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01140</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01001</span></span>
<span class="line"><span>Validation: Loss 0.00978 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01010</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00789</span></span>
<span class="line"><span>Validation: Loss 0.00876 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00979 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00873</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00820</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00679</span></span>
<span class="line"><span>Validation: Loss 0.00779 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00868 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00724</span></span>
<span class="line"><span>Validation: Loss 0.00708 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00786 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63022</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57757</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57041</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53454</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51952</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50700</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48921</span></span>
<span class="line"><span>Validation: Loss 0.47245 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47352 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47644</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45240</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44353</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42997</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41092</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39578</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36037</span></span>
<span class="line"><span>Validation: Loss 0.37625 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37720 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36803</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35978</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34916</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33857</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32299</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30786</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30615</span></span>
<span class="line"><span>Validation: Loss 0.29240 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29343 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29371</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28869</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26517</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24886</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22655</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25066</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22898</span></span>
<span class="line"><span>Validation: Loss 0.22265 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22374 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22540</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20991</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19514</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18889</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18566</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18334</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16150</span></span>
<span class="line"><span>Validation: Loss 0.16646 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16766 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15886</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14810</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14657</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15240</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12820</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13927</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14797</span></span>
<span class="line"><span>Validation: Loss 0.12275 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12388 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12948</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11094</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09922</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10877</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09826</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09539</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09414</span></span>
<span class="line"><span>Validation: Loss 0.08789 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08877 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08317</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08146</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08302</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07327</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06966</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06545</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05881</span></span>
<span class="line"><span>Validation: Loss 0.06094 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06154 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05907</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05360</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05383</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05448</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04719</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05004</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04583</span></span>
<span class="line"><span>Validation: Loss 0.04491 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04537 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04233</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04659</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04293</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04126</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03526</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03550</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03536</span></span>
<span class="line"><span>Validation: Loss 0.03621 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03660 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03356</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03442</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03302</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03108</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03475</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03235</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03293</span></span>
<span class="line"><span>Validation: Loss 0.03071 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03106 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02993</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02931</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03120</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02711</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02659</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02632</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02954</span></span>
<span class="line"><span>Validation: Loss 0.02669 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02701 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02896</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02822</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02306</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02353</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02289</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02438</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01854</span></span>
<span class="line"><span>Validation: Loss 0.02358 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02387 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02193</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02218</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02325</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02248</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02230</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02081</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02029</span></span>
<span class="line"><span>Validation: Loss 0.02112 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02139 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02212</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01791</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02184</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01955</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01838</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01688</span></span>
<span class="line"><span>Validation: Loss 0.01908 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01933 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01803</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01778</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01788</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01796</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01808</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01849</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01596</span></span>
<span class="line"><span>Validation: Loss 0.01737 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01760 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01752</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01536</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01644</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01719</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01619</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01491</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01886</span></span>
<span class="line"><span>Validation: Loss 0.01590 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01612 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01605</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01661</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01464</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01401</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01382</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01497</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01509</span></span>
<span class="line"><span>Validation: Loss 0.01463 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01484 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01265</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01546</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01514</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01305</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01341</span></span>
<span class="line"><span>Validation: Loss 0.01354 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01373 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01302</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01208</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01246</span></span>
<span class="line"><span>Validation: Loss 0.01258 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01276 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01136</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01066</span></span>
<span class="line"><span>Validation: Loss 0.01172 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01189 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01242</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01046</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01060</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01256</span></span>
<span class="line"><span>Validation: Loss 0.01091 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01107 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01163</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01003</span></span>
<span class="line"><span>Validation: Loss 0.01008 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01023 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01036</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00981</span></span>
<span class="line"><span>Validation: Loss 0.00915 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00928 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00892</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00876</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00630</span></span>
<span class="line"><span>Validation: Loss 0.00814 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00825 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
