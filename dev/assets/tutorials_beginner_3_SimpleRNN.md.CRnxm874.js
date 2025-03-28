import{_ as a,c as n,o as e,al as i}from"./chunks/framework.BCN3FD2k.js";const d=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"};function l(t,s,c,r,h,k){return e(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><p>Note: If you wish to use AutoZygote() for automatic differentiation, add Zygote to your project dependencies and include <code>using Zygote</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling ADTypes...</span></span>
<span class="line"><span>    617.6 ms  ✓ ADTypes</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds</span></span>
<span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    323.9 ms  ✓ ConcreteStructs</span></span>
<span class="line"><span>    341.7 ms  ✓ Future</span></span>
<span class="line"><span>    352.0 ms  ✓ OpenLibm_jll</span></span>
<span class="line"><span>    501.4 ms  ✓ Statistics</span></span>
<span class="line"><span>    430.2 ms  ✓ CompilerSupportLibraries_jll</span></span>
<span class="line"><span>    365.2 ms  ✓ ManualMemory</span></span>
<span class="line"><span>   1747.1 ms  ✓ UnsafeAtomics</span></span>
<span class="line"><span>    456.8 ms  ✓ Requires</span></span>
<span class="line"><span>    308.0 ms  ✓ Reexport</span></span>
<span class="line"><span>    306.6 ms  ✓ SIMDTypes</span></span>
<span class="line"><span>    538.6 ms  ✓ EnzymeCore</span></span>
<span class="line"><span>    310.8 ms  ✓ IfElse</span></span>
<span class="line"><span>   1088.5 ms  ✓ IrrationalConstants</span></span>
<span class="line"><span>   2316.6 ms  ✓ MacroTools</span></span>
<span class="line"><span>    327.7 ms  ✓ CommonWorldInvalidations</span></span>
<span class="line"><span>    330.5 ms  ✓ FastClosures</span></span>
<span class="line"><span>    430.7 ms  ✓ ConstructionBase</span></span>
<span class="line"><span>    384.6 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>    579.3 ms  ✓ Compat</span></span>
<span class="line"><span>    626.4 ms  ✓ CpuId</span></span>
<span class="line"><span>    607.4 ms  ✓ DocStringExtensions</span></span>
<span class="line"><span>    445.5 ms  ✓ NaNMath</span></span>
<span class="line"><span>    486.6 ms  ✓ JLLWrappers</span></span>
<span class="line"><span>    403.6 ms  ✓ Adapt</span></span>
<span class="line"><span>    474.3 ms  ✓ Atomix</span></span>
<span class="line"><span>    828.6 ms  ✓ ThreadingUtilities</span></span>
<span class="line"><span>    355.2 ms  ✓ ADTypes → ADTypesEnzymeCoreExt</span></span>
<span class="line"><span>    672.6 ms  ✓ CommonSubexpressions</span></span>
<span class="line"><span>    356.7 ms  ✓ ConstructionBase → ConstructionBaseLinearAlgebraExt</span></span>
<span class="line"><span>    765.5 ms  ✓ Static</span></span>
<span class="line"><span>    364.6 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>    392.2 ms  ✓ DiffResults</span></span>
<span class="line"><span>   1555.3 ms  ✓ DispatchDoctor</span></span>
<span class="line"><span>    377.3 ms  ✓ Compat → CompatLinearAlgebraExt</span></span>
<span class="line"><span>    598.0 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    619.0 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>    434.3 ms  ✓ GPUArraysCore</span></span>
<span class="line"><span>    500.9 ms  ✓ ArrayInterface</span></span>
<span class="line"><span>    362.0 ms  ✓ EnzymeCore → AdaptExt</span></span>
<span class="line"><span>    401.0 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>   1560.0 ms  ✓ Setfield</span></span>
<span class="line"><span>   1021.1 ms  ✓ CPUSummary</span></span>
<span class="line"><span>    404.9 ms  ✓ DispatchDoctor → DispatchDoctorEnzymeCoreExt</span></span>
<span class="line"><span>    577.2 ms  ✓ Functors</span></span>
<span class="line"><span>   1318.0 ms  ✓ ChainRulesCore</span></span>
<span class="line"><span>   1163.5 ms  ✓ LuxCore</span></span>
<span class="line"><span>   1643.1 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>   2503.4 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    371.3 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>   7308.4 ms  ✓ StaticArrays</span></span>
<span class="line"><span>    345.0 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>    601.5 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>    397.6 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    910.3 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>    636.8 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>    397.2 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>   1222.1 ms  ✓ Optimisers</span></span>
<span class="line"><span>    447.9 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    686.9 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>   1364.2 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    436.3 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    450.6 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    499.0 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>    575.6 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    616.1 ms  ✓ DiffRules</span></span>
<span class="line"><span>    626.2 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>   1651.6 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    602.8 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    642.2 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    606.3 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    684.2 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    666.9 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>   2715.7 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    467.2 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    404.0 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    416.1 ms  ✓ Optimisers → OptimisersAdaptExt</span></span>
<span class="line"><span>    911.6 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>    977.4 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    714.3 ms  ✓ Polyester</span></span>
<span class="line"><span>   3531.7 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>   3931.8 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    908.0 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>    668.1 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    726.8 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5838.2 ms  ✓ NNlib</span></span>
<span class="line"><span>    972.3 ms  ✓ NNlib → NNlibSpecialFunctionsExt</span></span>
<span class="line"><span>    988.9 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>   1053.4 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5642.9 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9729.1 ms  ✓ Lux</span></span>
<span class="line"><span>  90 dependencies successfully precompiled in 47 seconds. 20 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>   4111.7 ms  ✓ FileIO</span></span>
<span class="line"><span>  31513.3 ms  ✓ JLD2</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 36 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    410.0 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>    613.9 ms  ✓ Adapt → AdaptSparseArraysExt</span></span>
<span class="line"><span>    663.3 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>    402.7 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>   1126.6 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>    715.4 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>   1636.3 ms  ✓ DataStructures</span></span>
<span class="line"><span>    447.5 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    946.4 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   3680.6 ms  ✓ Test</span></span>
<span class="line"><span>    578.0 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>    500.9 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>   1084.8 ms  ✓ MLCore</span></span>
<span class="line"><span>    596.1 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>   2590.1 ms  ✓ Accessors</span></span>
<span class="line"><span>    906.4 ms  ✓ Accessors → LinearAlgebraExt</span></span>
<span class="line"><span>   1261.0 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    678.5 ms  ✓ Accessors → StaticArraysExt</span></span>
<span class="line"><span>    752.1 ms  ✓ Accessors → TestExt</span></span>
<span class="line"><span>   2279.1 ms  ✓ StatsBase</span></span>
<span class="line"><span>    755.5 ms  ✓ BangBang</span></span>
<span class="line"><span>    705.7 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    731.8 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    775.0 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>   1040.0 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2699.8 ms  ✓ Transducers</span></span>
<span class="line"><span>    697.7 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   5590.1 ms  ✓ FLoops</span></span>
<span class="line"><span>   5853.7 ms  ✓ MLUtils</span></span>
<span class="line"><span>  29 dependencies successfully precompiled in 24 seconds. 73 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceSparseArraysExt...</span></span>
<span class="line"><span>    619.4 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 8 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    708.6 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1723.0 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2070.0 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 169 already precompiled.</span></span>
<span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>    714.2 ms  ✓ ReactantCore</span></span>
<span class="line"><span>   1033.3 ms  ✓ MbedTLS</span></span>
<span class="line"><span>   2009.5 ms  ✓ ObjectFile</span></span>
<span class="line"><span>    410.0 ms  ✓ Scratch</span></span>
<span class="line"><span>    491.1 ms  ✓ LoggingExtras</span></span>
<span class="line"><span>    513.5 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>   2719.7 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>    665.1 ms  ✓ OpenSSL_jll</span></span>
<span class="line"><span>    650.1 ms  ✓ LLVMOpenMP_jll</span></span>
<span class="line"><span>   1180.3 ms  ✓ CUDA_Driver_jll</span></span>
<span class="line"><span>   1014.7 ms  ✓ LazyArtifacts</span></span>
<span class="line"><span>   1916.4 ms  ✓ OpenSSL</span></span>
<span class="line"><span>   1418.1 ms  ✓ LLVMExtra_jll</span></span>
<span class="line"><span>   1423.2 ms  ✓ Enzyme_jll</span></span>
<span class="line"><span>   2558.1 ms  ✓ Reactant_jll</span></span>
<span class="line"><span>   6640.9 ms  ✓ LLVM</span></span>
<span class="line"><span>  19046.1 ms  ✓ HTTP</span></span>
<span class="line"><span>  27423.5 ms  ✓ GPUCompiler</span></span>
<span class="line"><span> 221810.2 ms  ✓ Enzyme</span></span>
<span class="line"><span>   6024.2 ms  ✓ Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>  75428.1 ms  ✓ Reactant</span></span>
<span class="line"><span>  21 dependencies successfully precompiled in 343 seconds. 56 already precompiled.</span></span>
<span class="line"><span>Precompiling UnsafeAtomicsLLVM...</span></span>
<span class="line"><span>   2046.0 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibEnzymeExt...</span></span>
<span class="line"><span>   5949.3 ms  ✓ Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>  10849.1 ms  ✓ Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>  11906.6 ms  ✓ Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   1321.3 ms  ✓ LuxLib → LuxLibEnzymeExt</span></span>
<span class="line"><span>   6567.3 ms  ✓ Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>  5 dependencies successfully precompiled in 13 seconds. 128 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   6869.5 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 148 already precompiled.</span></span>
<span class="line"><span>Precompiling HTTPExt...</span></span>
<span class="line"><span>   1778.8 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 43 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  12929.9 ms  ✓ LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  13635.0 ms  ✓ MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 79 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  13875.2 ms  ✓ Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>  14076.3 ms  ✓ Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>  14225.6 ms  ✓ WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 15 seconds. 91 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantKernelAbstractionsExt...</span></span>
<span class="line"><span>  14430.8 ms  ✓ Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 15 seconds. 89 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantArrayInterfaceExt...</span></span>
<span class="line"><span>  12665.1 ms  ✓ Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 80 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  13970.2 ms  ✓ Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  10873.2 ms  ✓ Lux → LuxReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 11 seconds. 178 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span>2025-03-28 03:44:13.006604: I external/xla/xla/service/service.cc:152] XLA service 0x25dbee0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-28 03:44:13.007183: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1743133453.009980 2738871 se_gpu_pjrt_client.cc:1039] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1743133453.010658 2738871 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743133453.010961 2738871 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743133453.035514 2738871 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>E0000 00:00:1743133500.726998 2738871 buffer_comparator.cc:156] Difference at 32: 0, expected 1.62244</span></span>
<span class="line"><span>E0000 00:00:1743133500.727495 2738871 buffer_comparator.cc:156] Difference at 33: 0, expected 1.87084</span></span>
<span class="line"><span>E0000 00:00:1743133500.727499 2738871 buffer_comparator.cc:156] Difference at 34: 0, expected 1.07351</span></span>
<span class="line"><span>E0000 00:00:1743133500.727502 2738871 buffer_comparator.cc:156] Difference at 35: 0, expected 2.92445</span></span>
<span class="line"><span>E0000 00:00:1743133500.727505 2738871 buffer_comparator.cc:156] Difference at 36: 0, expected 1.98056</span></span>
<span class="line"><span>E0000 00:00:1743133500.727507 2738871 buffer_comparator.cc:156] Difference at 37: 0, expected 2.07715</span></span>
<span class="line"><span>E0000 00:00:1743133500.727510 2738871 buffer_comparator.cc:156] Difference at 38: 0, expected 1.56458</span></span>
<span class="line"><span>E0000 00:00:1743133500.727513 2738871 buffer_comparator.cc:156] Difference at 39: 0, expected 2.27034</span></span>
<span class="line"><span>E0000 00:00:1743133500.727515 2738871 buffer_comparator.cc:156] Difference at 40: 0, expected 2.31795</span></span>
<span class="line"><span>E0000 00:00:1743133500.727518 2738871 buffer_comparator.cc:156] Difference at 41: 0, expected 2.55731</span></span>
<span class="line"><span>2025-03-28 03:45:00.727527: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.731921 2738871 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743133500.731938 2738871 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743133500.731942 2738871 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743133500.731944 2738871 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743133500.731947 2738871 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743133500.731950 2738871 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743133500.731953 2738871 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743133500.731956 2738871 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743133500.731958 2738871 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743133500.731961 2738871 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-03-28 03:45:00.731966: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.734612 2738871 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743133500.734626 2738871 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743133500.734629 2738871 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743133500.734632 2738871 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743133500.734635 2738871 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743133500.734637 2738871 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743133500.734640 2738871 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743133500.734643 2738871 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743133500.734647 2738871 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743133500.734650 2738871 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-03-28 03:45:00.734654: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.737132 2738871 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1743133500.737146 2738871 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1743133500.737149 2738871 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1743133500.737152 2738871 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1743133500.737154 2738871 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1743133500.737157 2738871 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1743133500.737160 2738871 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1743133500.737163 2738871 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1743133500.737165 2738871 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1743133500.737168 2738871 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-03-28 03:45:00.737173: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.739646 2738871 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743133500.739661 2738871 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743133500.739663 2738871 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743133500.739666 2738871 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743133500.739669 2738871 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743133500.739672 2738871 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743133500.739675 2738871 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743133500.739678 2738871 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743133500.739680 2738871 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743133500.739683 2738871 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-03-28 03:45:00.739688: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.742166 2738871 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743133500.742180 2738871 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743133500.742183 2738871 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743133500.742185 2738871 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743133500.742188 2738871 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743133500.742191 2738871 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743133500.742194 2738871 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743133500.742196 2738871 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743133500.742199 2738871 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743133500.742202 2738871 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-03-28 03:45:00.742206: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.744692 2738871 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1743133500.744705 2738871 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1743133500.744708 2738871 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1743133500.744711 2738871 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1743133500.744713 2738871 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1743133500.744716 2738871 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1743133500.744719 2738871 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1743133500.744722 2738871 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1743133500.744724 2738871 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1743133500.744727 2738871 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-03-28 03:45:00.744732: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.747211 2738871 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743133500.747225 2738871 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743133500.747228 2738871 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743133500.747231 2738871 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743133500.747234 2738871 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743133500.747236 2738871 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743133500.747239 2738871 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743133500.747242 2738871 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743133500.747245 2738871 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743133500.747247 2738871 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-03-28 03:45:00.747252: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.749721 2738871 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743133500.749735 2738871 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743133500.749738 2738871 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743133500.749741 2738871 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743133500.749743 2738871 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743133500.749746 2738871 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743133500.749749 2738871 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743133500.749752 2738871 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743133500.749754 2738871 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743133500.749757 2738871 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-03-28 03:45:00.749762: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.752229 2738871 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743133500.752243 2738871 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743133500.752246 2738871 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743133500.752250 2738871 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743133500.752253 2738871 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743133500.752256 2738871 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743133500.752258 2738871 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743133500.752261 2738871 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743133500.752264 2738871 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743133500.752267 2738871 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-03-28 03:45:00.752271: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.754747 2738871 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1743133500.754760 2738871 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1743133500.754763 2738871 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1743133500.754766 2738871 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1743133500.754769 2738871 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1743133500.754772 2738871 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1743133500.754775 2738871 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1743133500.754777 2738871 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1743133500.754780 2738871 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1743133500.754783 2738871 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-03-28 03:45:00.754787: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.757282 2738871 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743133500.757295 2738871 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743133500.757299 2738871 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743133500.757301 2738871 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743133500.757304 2738871 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743133500.757307 2738871 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743133500.757310 2738871 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743133500.757313 2738871 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743133500.757315 2738871 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743133500.757318 2738871 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-03-28 03:45:00.757322: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.759804 2738871 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743133500.759821 2738871 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743133500.759824 2738871 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743133500.759827 2738871 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743133500.759830 2738871 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743133500.759832 2738871 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743133500.759835 2738871 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743133500.759839 2738871 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743133500.759842 2738871 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743133500.759845 2738871 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-03-28 03:45:00.759849: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.781177 2738871 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743133500.781209 2738871 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743133500.781217 2738871 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743133500.781223 2738871 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743133500.781230 2738871 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743133500.781237 2738871 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743133500.781244 2738871 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743133500.781251 2738871 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743133500.781257 2738871 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743133500.781264 2738871 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-03-28 03:45:00.781275: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.784748 2738871 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1743133500.784778 2738871 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1743133500.784785 2738871 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1743133500.784791 2738871 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1743133500.784797 2738871 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1743133500.784804 2738871 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1743133500.784810 2738871 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1743133500.784816 2738871 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1743133500.784822 2738871 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1743133500.784828 2738871 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-03-28 03:45:00.784838: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.787602 2738871 buffer_comparator.cc:156] Difference at 256: 0, expected 0.86224</span></span>
<span class="line"><span>E0000 00:00:1743133500.787625 2738871 buffer_comparator.cc:156] Difference at 257: 0, expected 0.686873</span></span>
<span class="line"><span>E0000 00:00:1743133500.787629 2738871 buffer_comparator.cc:156] Difference at 258: 0, expected 0.252371</span></span>
<span class="line"><span>E0000 00:00:1743133500.787633 2738871 buffer_comparator.cc:156] Difference at 259: 0, expected 0.335927</span></span>
<span class="line"><span>E0000 00:00:1743133500.787637 2738871 buffer_comparator.cc:156] Difference at 260: 0, expected 0.934139</span></span>
<span class="line"><span>E0000 00:00:1743133500.787641 2738871 buffer_comparator.cc:156] Difference at 261: 0, expected 0.274756</span></span>
<span class="line"><span>E0000 00:00:1743133500.787645 2738871 buffer_comparator.cc:156] Difference at 262: 0, expected 0.529946</span></span>
<span class="line"><span>E0000 00:00:1743133500.787649 2738871 buffer_comparator.cc:156] Difference at 263: 0, expected 0.542969</span></span>
<span class="line"><span>E0000 00:00:1743133500.787653 2738871 buffer_comparator.cc:156] Difference at 264: 0, expected 0.895372</span></span>
<span class="line"><span>E0000 00:00:1743133500.787656 2738871 buffer_comparator.cc:156] Difference at 265: 0, expected 0.895664</span></span>
<span class="line"><span>2025-03-28 03:45:00.787663: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133500.790270 2738871 buffer_comparator.cc:156] Difference at 256: 0, expected 0.86224</span></span>
<span class="line"><span>E0000 00:00:1743133500.790292 2738871 buffer_comparator.cc:156] Difference at 257: 0, expected 0.686873</span></span>
<span class="line"><span>E0000 00:00:1743133500.790297 2738871 buffer_comparator.cc:156] Difference at 258: 0, expected 0.252371</span></span>
<span class="line"><span>E0000 00:00:1743133500.790301 2738871 buffer_comparator.cc:156] Difference at 259: 0, expected 0.335927</span></span>
<span class="line"><span>E0000 00:00:1743133500.790305 2738871 buffer_comparator.cc:156] Difference at 260: 0, expected 0.934139</span></span>
<span class="line"><span>E0000 00:00:1743133500.790308 2738871 buffer_comparator.cc:156] Difference at 261: 0, expected 0.274756</span></span>
<span class="line"><span>E0000 00:00:1743133500.790312 2738871 buffer_comparator.cc:156] Difference at 262: 0, expected 0.529946</span></span>
<span class="line"><span>E0000 00:00:1743133500.790316 2738871 buffer_comparator.cc:156] Difference at 263: 0, expected 0.542969</span></span>
<span class="line"><span>E0000 00:00:1743133500.790320 2738871 buffer_comparator.cc:156] Difference at 264: 0, expected 0.895372</span></span>
<span class="line"><span>E0000 00:00:1743133500.790324 2738871 buffer_comparator.cc:156] Difference at 265: 0, expected 0.895664</span></span>
<span class="line"><span>2025-03-28 03:45:00.790330: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.432017 2738871 buffer_comparator.cc:156] Difference at 16: -3.59327, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1743133539.432072 2738871 buffer_comparator.cc:156] Difference at 17: -3.73287, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1743133539.432081 2738871 buffer_comparator.cc:156] Difference at 18: -4.28903, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1743133539.432086 2738871 buffer_comparator.cc:156] Difference at 19: -2.73747, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1743133539.432091 2738871 buffer_comparator.cc:156] Difference at 20: -4.69749, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1743133539.432095 2738871 buffer_comparator.cc:156] Difference at 21: -1.94942, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1743133539.432099 2738871 buffer_comparator.cc:156] Difference at 22: -5.05361, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1743133539.432104 2738871 buffer_comparator.cc:156] Difference at 23: -0.787761, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1743133539.432109 2738871 buffer_comparator.cc:156] Difference at 24: -5.07818, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1743133539.432113 2738871 buffer_comparator.cc:156] Difference at 25: 0.0275401, expected 11.3838</span></span>
<span class="line"><span>2025-03-28 03:45:39.432125: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.434468 2738871 buffer_comparator.cc:156] Difference at 16: -3.59327, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1743133539.434484 2738871 buffer_comparator.cc:156] Difference at 17: -3.73287, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1743133539.434490 2738871 buffer_comparator.cc:156] Difference at 18: -4.28903, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1743133539.434495 2738871 buffer_comparator.cc:156] Difference at 19: -2.73747, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1743133539.434500 2738871 buffer_comparator.cc:156] Difference at 20: -4.69749, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1743133539.434504 2738871 buffer_comparator.cc:156] Difference at 21: -1.94942, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1743133539.434509 2738871 buffer_comparator.cc:156] Difference at 22: -5.05361, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1743133539.434513 2738871 buffer_comparator.cc:156] Difference at 23: -0.787761, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1743133539.434518 2738871 buffer_comparator.cc:156] Difference at 24: -5.07818, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1743133539.434522 2738871 buffer_comparator.cc:156] Difference at 25: 0.0275401, expected 11.3838</span></span>
<span class="line"><span>2025-03-28 03:45:39.434530: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.436744 2738871 buffer_comparator.cc:156] Difference at 32: -3.26609, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1743133539.436760 2738871 buffer_comparator.cc:156] Difference at 33: 3.34559, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1743133539.436768 2738871 buffer_comparator.cc:156] Difference at 34: -2.59757, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1743133539.436773 2738871 buffer_comparator.cc:156] Difference at 35: 3.79163, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1743133539.436778 2738871 buffer_comparator.cc:156] Difference at 36: -1.8023, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1743133539.436783 2738871 buffer_comparator.cc:156] Difference at 37: 4.39967, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1743133539.436787 2738871 buffer_comparator.cc:156] Difference at 38: -0.761131, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1743133539.436792 2738871 buffer_comparator.cc:156] Difference at 39: 4.60016, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1743133539.436796 2738871 buffer_comparator.cc:156] Difference at 40: -0.0202891, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1743133539.436801 2738871 buffer_comparator.cc:156] Difference at 41: 4.50229, expected 8.63119</span></span>
<span class="line"><span>2025-03-28 03:45:39.436808: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.439014 2738871 buffer_comparator.cc:156] Difference at 32: -3.26609, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1743133539.439031 2738871 buffer_comparator.cc:156] Difference at 33: 3.34559, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1743133539.439036 2738871 buffer_comparator.cc:156] Difference at 34: -2.59757, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1743133539.439041 2738871 buffer_comparator.cc:156] Difference at 35: 3.79163, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1743133539.439046 2738871 buffer_comparator.cc:156] Difference at 36: -1.8023, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1743133539.439050 2738871 buffer_comparator.cc:156] Difference at 37: 4.39967, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1743133539.439055 2738871 buffer_comparator.cc:156] Difference at 38: -0.761131, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1743133539.439059 2738871 buffer_comparator.cc:156] Difference at 39: 4.60016, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1743133539.439064 2738871 buffer_comparator.cc:156] Difference at 40: -0.0202891, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1743133539.439069 2738871 buffer_comparator.cc:156] Difference at 41: 4.50229, expected 8.63119</span></span>
<span class="line"><span>2025-03-28 03:45:39.439076: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.441299 2738871 buffer_comparator.cc:156] Difference at 64: 2.80472, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743133539.441315 2738871 buffer_comparator.cc:156] Difference at 65: -2.45324, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743133539.441321 2738871 buffer_comparator.cc:156] Difference at 66: 2.12094, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743133539.441326 2738871 buffer_comparator.cc:156] Difference at 67: -3.08973, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743133539.441330 2738871 buffer_comparator.cc:156] Difference at 68: 1.31399, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743133539.441335 2738871 buffer_comparator.cc:156] Difference at 69: -3.33359, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743133539.441339 2738871 buffer_comparator.cc:156] Difference at 70: 0.549249, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743133539.441344 2738871 buffer_comparator.cc:156] Difference at 71: -3.50826, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743133539.441348 2738871 buffer_comparator.cc:156] Difference at 72: 0.0468082, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743133539.441353 2738871 buffer_comparator.cc:156] Difference at 73: -3.44122, expected 8.82565</span></span>
<span class="line"><span>2025-03-28 03:45:39.441360: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.443588 2738871 buffer_comparator.cc:156] Difference at 64: 2.80472, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743133539.443600 2738871 buffer_comparator.cc:156] Difference at 65: -2.45324, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743133539.443605 2738871 buffer_comparator.cc:156] Difference at 66: 2.12094, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743133539.443608 2738871 buffer_comparator.cc:156] Difference at 67: -3.08973, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743133539.443612 2738871 buffer_comparator.cc:156] Difference at 68: 1.31399, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743133539.443616 2738871 buffer_comparator.cc:156] Difference at 69: -3.33359, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743133539.443618 2738871 buffer_comparator.cc:156] Difference at 70: 0.549249, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743133539.443622 2738871 buffer_comparator.cc:156] Difference at 71: -3.50826, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743133539.443625 2738871 buffer_comparator.cc:156] Difference at 72: 0.0468082, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743133539.443628 2738871 buffer_comparator.cc:156] Difference at 73: -3.44122, expected 8.82565</span></span>
<span class="line"><span>2025-03-28 03:45:39.443632: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.445749 2738871 buffer_comparator.cc:156] Difference at 64: 2.80472, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1743133539.445760 2738871 buffer_comparator.cc:156] Difference at 65: -2.45324, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1743133539.445764 2738871 buffer_comparator.cc:156] Difference at 66: 2.12094, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1743133539.445767 2738871 buffer_comparator.cc:156] Difference at 67: -3.08973, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1743133539.445770 2738871 buffer_comparator.cc:156] Difference at 68: 1.31399, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1743133539.445773 2738871 buffer_comparator.cc:156] Difference at 69: -3.33359, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1743133539.445776 2738871 buffer_comparator.cc:156] Difference at 70: 0.549249, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1743133539.445779 2738871 buffer_comparator.cc:156] Difference at 71: -3.50826, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1743133539.445782 2738871 buffer_comparator.cc:156] Difference at 72: 0.0468082, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1743133539.445785 2738871 buffer_comparator.cc:156] Difference at 73: -3.44122, expected 8.82565</span></span>
<span class="line"><span>2025-03-28 03:45:39.445790: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.456869 2738871 buffer_comparator.cc:156] Difference at 16: 9.1023, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1743133539.456898 2738871 buffer_comparator.cc:156] Difference at 17: 8.26259, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1743133539.456901 2738871 buffer_comparator.cc:156] Difference at 18: 10.5875, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1743133539.456905 2738871 buffer_comparator.cc:156] Difference at 19: 9.23707, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1743133539.456909 2738871 buffer_comparator.cc:156] Difference at 20: 8.65902, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1743133539.456913 2738871 buffer_comparator.cc:156] Difference at 21: 10.9934, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1743133539.456916 2738871 buffer_comparator.cc:156] Difference at 22: 10.0339, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1743133539.456919 2738871 buffer_comparator.cc:156] Difference at 23: 8.29374, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1743133539.456922 2738871 buffer_comparator.cc:156] Difference at 24: 7.65491, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1743133539.456926 2738871 buffer_comparator.cc:156] Difference at 25: 8.47378, expected 36.4575</span></span>
<span class="line"><span>2025-03-28 03:45:39.456934: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.459043 2738871 buffer_comparator.cc:156] Difference at 16: 9.1023, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1743133539.459056 2738871 buffer_comparator.cc:156] Difference at 17: 8.26259, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1743133539.459059 2738871 buffer_comparator.cc:156] Difference at 18: 10.5875, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1743133539.459063 2738871 buffer_comparator.cc:156] Difference at 19: 9.23707, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1743133539.459066 2738871 buffer_comparator.cc:156] Difference at 20: 8.65902, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1743133539.459068 2738871 buffer_comparator.cc:156] Difference at 21: 10.9934, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1743133539.459073 2738871 buffer_comparator.cc:156] Difference at 22: 10.0339, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1743133539.459076 2738871 buffer_comparator.cc:156] Difference at 23: 8.29374, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1743133539.459079 2738871 buffer_comparator.cc:156] Difference at 24: 7.65491, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1743133539.459082 2738871 buffer_comparator.cc:156] Difference at 25: 8.47378, expected 36.4575</span></span>
<span class="line"><span>2025-03-28 03:45:39.459087: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.472078 2738871 buffer_comparator.cc:156] Difference at 2: 37.3354, expected 33.2434</span></span>
<span class="line"><span>E0000 00:00:1743133539.472106 2738871 buffer_comparator.cc:156] Difference at 8: 32.9004, expected 29.0801</span></span>
<span class="line"><span>E0000 00:00:1743133539.472110 2738871 buffer_comparator.cc:156] Difference at 11: 35.2933, expected 30.7625</span></span>
<span class="line"><span>E0000 00:00:1743133539.472113 2738871 buffer_comparator.cc:156] Difference at 12: 39.5031, expected 34.3637</span></span>
<span class="line"><span>E0000 00:00:1743133539.472116 2738871 buffer_comparator.cc:156] Difference at 20: 38.8088, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1743133539.472119 2738871 buffer_comparator.cc:156] Difference at 23: 36.9993, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1743133539.472122 2738871 buffer_comparator.cc:156] Difference at 26: 39.1357, expected 32.4927</span></span>
<span class="line"><span>E0000 00:00:1743133539.472125 2738871 buffer_comparator.cc:156] Difference at 51: 26.8162, expected 33.7879</span></span>
<span class="line"><span>2025-03-28 03:45:39.472134: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.481198 2738871 buffer_comparator.cc:156] Difference at 16: 6.73254, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1743133539.481224 2738871 buffer_comparator.cc:156] Difference at 17: 9.2143, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743133539.481227 2738871 buffer_comparator.cc:156] Difference at 18: 6.85711, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1743133539.481230 2738871 buffer_comparator.cc:156] Difference at 19: 7.1738, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1743133539.481233 2738871 buffer_comparator.cc:156] Difference at 20: 6.9796, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743133539.481236 2738871 buffer_comparator.cc:156] Difference at 21: 7.31867, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1743133539.481239 2738871 buffer_comparator.cc:156] Difference at 22: 8.70739, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1743133539.481242 2738871 buffer_comparator.cc:156] Difference at 23: 7.19731, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1743133539.481245 2738871 buffer_comparator.cc:156] Difference at 24: 8.21217, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1743133539.481248 2738871 buffer_comparator.cc:156] Difference at 25: 9.32022, expected 36.0917</span></span>
<span class="line"><span>2025-03-28 03:45:39.481256: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.483358 2738871 buffer_comparator.cc:156] Difference at 16: 6.73254, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1743133539.483371 2738871 buffer_comparator.cc:156] Difference at 17: 9.2143, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743133539.483374 2738871 buffer_comparator.cc:156] Difference at 18: 6.85711, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1743133539.483377 2738871 buffer_comparator.cc:156] Difference at 19: 7.1738, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1743133539.483380 2738871 buffer_comparator.cc:156] Difference at 20: 6.9796, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743133539.483383 2738871 buffer_comparator.cc:156] Difference at 21: 7.31867, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1743133539.483386 2738871 buffer_comparator.cc:156] Difference at 22: 8.70739, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1743133539.483388 2738871 buffer_comparator.cc:156] Difference at 23: 7.19731, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1743133539.483391 2738871 buffer_comparator.cc:156] Difference at 24: 8.21217, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1743133539.483394 2738871 buffer_comparator.cc:156] Difference at 25: 9.32022, expected 36.0917</span></span>
<span class="line"><span>2025-03-28 03:45:39.483401: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133539.495929 2738871 buffer_comparator.cc:156] Difference at 2: 38.3235, expected 34.2806</span></span>
<span class="line"><span>E0000 00:00:1743133539.495958 2738871 buffer_comparator.cc:156] Difference at 6: 41.1479, expected 36.7103</span></span>
<span class="line"><span>E0000 00:00:1743133539.495962 2738871 buffer_comparator.cc:156] Difference at 13: 31.5782, expected 35.7459</span></span>
<span class="line"><span>E0000 00:00:1743133539.495965 2738871 buffer_comparator.cc:156] Difference at 17: 37.0608, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1743133539.495968 2738871 buffer_comparator.cc:156] Difference at 20: 37.8794, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1743133539.495971 2738871 buffer_comparator.cc:156] Difference at 45: 25.8921, expected 32.5352</span></span>
<span class="line"><span>E0000 00:00:1743133539.495975 2738871 buffer_comparator.cc:156] Difference at 75: 24.6946, expected 28.3085</span></span>
<span class="line"><span>E0000 00:00:1743133539.495978 2738871 buffer_comparator.cc:156] Difference at 77: 19.5083, expected 27.4887</span></span>
<span class="line"><span>E0000 00:00:1743133539.495981 2738871 buffer_comparator.cc:156] Difference at 94: 24.5253, expected 28.5145</span></span>
<span class="line"><span>E0000 00:00:1743133539.495984 2738871 buffer_comparator.cc:156] Difference at 101: 30.5971, expected 26.8436</span></span>
<span class="line"><span>2025-03-28 03:45:39.495993: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.81219</span></span>
<span class="line"><span>Validation:	Loss 0.73835	Accuracy 0.50000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.69831</span></span>
<span class="line"><span>Validation:	Loss 0.64435	Accuracy 0.50000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.60832</span></span>
<span class="line"><span>Validation:	Loss 0.55980	Accuracy 0.50000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.52475</span></span>
<span class="line"><span>Validation:	Loss 0.48101	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.45230</span></span>
<span class="line"><span>Validation:	Loss 0.41275	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.38985</span></span>
<span class="line"><span>Validation:	Loss 0.35301	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.33267</span></span>
<span class="line"><span>Validation:	Loss 0.29934	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.28313</span></span>
<span class="line"><span>Validation:	Loss 0.25476	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.23991</span></span>
<span class="line"><span>Validation:	Loss 0.21975	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.20722</span></span>
<span class="line"><span>Validation:	Loss 0.19172	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.18223</span></span>
<span class="line"><span>Validation:	Loss 0.16904	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.16177</span></span>
<span class="line"><span>Validation:	Loss 0.15035	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.14435</span></span>
<span class="line"><span>Validation:	Loss 0.13498	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.13055</span></span>
<span class="line"><span>Validation:	Loss 0.12188	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.11790</span></span>
<span class="line"><span>Validation:	Loss 0.11085	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.10702</span></span>
<span class="line"><span>Validation:	Loss 0.10143	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.09802</span></span>
<span class="line"><span>Validation:	Loss 0.09327	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.09120</span></span>
<span class="line"><span>Validation:	Loss 0.08614	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.08416</span></span>
<span class="line"><span>Validation:	Loss 0.07979	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.07783</span></span>
<span class="line"><span>Validation:	Loss 0.07398	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.07202</span></span>
<span class="line"><span>Validation:	Loss 0.06822	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.06647</span></span>
<span class="line"><span>Validation:	Loss 0.06162	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.05821</span></span>
<span class="line"><span>Validation:	Loss 0.05323	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.05001</span></span>
<span class="line"><span>Validation:	Loss 0.04451	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.04199</span></span>
<span class="line"><span>Validation:	Loss 0.03857	Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-11/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.75464</span></span>
<span class="line"><span>Validation:	Loss 0.61877	Accuracy 0.51562</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.57199</span></span>
<span class="line"><span>Validation:	Loss 0.51271	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.47717</span></span>
<span class="line"><span>Validation:	Loss 0.42318	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.39032</span></span>
<span class="line"><span>Validation:	Loss 0.33638	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.30362</span></span>
<span class="line"><span>Validation:	Loss 0.25343	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.22383</span></span>
<span class="line"><span>Validation:	Loss 0.18249	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.16075</span></span>
<span class="line"><span>Validation:	Loss 0.12861	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.11289</span></span>
<span class="line"><span>Validation:	Loss 0.09068	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.07950</span></span>
<span class="line"><span>Validation:	Loss 0.06545	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.05865</span></span>
<span class="line"><span>Validation:	Loss 0.04926	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.04484</span></span>
<span class="line"><span>Validation:	Loss 0.03900	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03622</span></span>
<span class="line"><span>Validation:	Loss 0.03217	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.03019</span></span>
<span class="line"><span>Validation:	Loss 0.02737	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02596</span></span>
<span class="line"><span>Validation:	Loss 0.02382	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02273</span></span>
<span class="line"><span>Validation:	Loss 0.02109	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02024</span></span>
<span class="line"><span>Validation:	Loss 0.01892	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01827</span></span>
<span class="line"><span>Validation:	Loss 0.01717	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01669</span></span>
<span class="line"><span>Validation:	Loss 0.01570	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01526</span></span>
<span class="line"><span>Validation:	Loss 0.01444	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01401</span></span>
<span class="line"><span>Validation:	Loss 0.01333	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01301</span></span>
<span class="line"><span>Validation:	Loss 0.01230	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01205</span></span>
<span class="line"><span>Validation:	Loss 0.01131	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01099</span></span>
<span class="line"><span>Validation:	Loss 0.01033	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01006</span></span>
<span class="line"><span>Validation:	Loss 0.00952	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00938</span></span>
<span class="line"><span>Validation:	Loss 0.00892	Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
