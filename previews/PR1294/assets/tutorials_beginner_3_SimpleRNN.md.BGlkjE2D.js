import{_ as a,c as n,o as e,al as i}from"./chunks/framework.Fn4jk33K.js";const d=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"};function l(t,s,c,r,h,k){return e(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><p>Note: If you wish to use AutoZygote() for automatic differentiation, add Zygote to your project dependencies and include <code>using Zygote</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling ADTypes...</span></span>
<span class="line"><span>    642.3 ms  ✓ ADTypes</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds</span></span>
<span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    330.1 ms  ✓ ConcreteStructs</span></span>
<span class="line"><span>    345.4 ms  ✓ Future</span></span>
<span class="line"><span>    373.4 ms  ✓ OpenLibm_jll</span></span>
<span class="line"><span>    527.0 ms  ✓ Statistics</span></span>
<span class="line"><span>    441.5 ms  ✓ CompilerSupportLibraries_jll</span></span>
<span class="line"><span>    367.6 ms  ✓ ManualMemory</span></span>
<span class="line"><span>   1741.2 ms  ✓ UnsafeAtomics</span></span>
<span class="line"><span>    454.1 ms  ✓ Requires</span></span>
<span class="line"><span>    309.2 ms  ✓ Reexport</span></span>
<span class="line"><span>    310.2 ms  ✓ SIMDTypes</span></span>
<span class="line"><span>    365.5 ms  ✓ HashArrayMappedTries</span></span>
<span class="line"><span>   1111.3 ms  ✓ IrrationalConstants</span></span>
<span class="line"><span>    531.4 ms  ✓ EnzymeCore</span></span>
<span class="line"><span>   2374.3 ms  ✓ MacroTools</span></span>
<span class="line"><span>    332.9 ms  ✓ IfElse</span></span>
<span class="line"><span>    330.7 ms  ✓ CommonWorldInvalidations</span></span>
<span class="line"><span>    442.9 ms  ✓ ConstructionBase</span></span>
<span class="line"><span>    341.3 ms  ✓ FastClosures</span></span>
<span class="line"><span>    380.8 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>    551.6 ms  ✓ Compat</span></span>
<span class="line"><span>    460.5 ms  ✓ NaNMath</span></span>
<span class="line"><span>    623.3 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>    527.0 ms  ✓ Atomix</span></span>
<span class="line"><span>    417.5 ms  ✓ Adapt</span></span>
<span class="line"><span>    843.0 ms  ✓ ThreadingUtilities</span></span>
<span class="line"><span>    387.1 ms  ✓ ADTypes → ADTypesEnzymeCoreExt</span></span>
<span class="line"><span>    603.5 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    716.6 ms  ✓ CommonSubexpressions</span></span>
<span class="line"><span>    371.5 ms  ✓ ConstructionBase → ConstructionBaseLinearAlgebraExt</span></span>
<span class="line"><span>    779.6 ms  ✓ Static</span></span>
<span class="line"><span>    390.9 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>    409.2 ms  ✓ DiffResults</span></span>
<span class="line"><span>    374.6 ms  ✓ Compat → CompatLinearAlgebraExt</span></span>
<span class="line"><span>   1717.4 ms  ✓ DispatchDoctor</span></span>
<span class="line"><span>    521.7 ms  ✓ ArrayInterface</span></span>
<span class="line"><span>    457.4 ms  ✓ GPUArraysCore</span></span>
<span class="line"><span>    389.5 ms  ✓ EnzymeCore → AdaptExt</span></span>
<span class="line"><span>   1440.5 ms  ✓ Setfield</span></span>
<span class="line"><span>    403.9 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>   2597.2 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>   1106.5 ms  ✓ CPUSummary</span></span>
<span class="line"><span>   1207.3 ms  ✓ ChainRulesCore</span></span>
<span class="line"><span>    623.0 ms  ✓ Functors</span></span>
<span class="line"><span>    448.5 ms  ✓ DispatchDoctor → DispatchDoctorEnzymeCoreExt</span></span>
<span class="line"><span>   1188.5 ms  ✓ LuxCore</span></span>
<span class="line"><span>    360.9 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>   1549.6 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>    371.5 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>   7421.1 ms  ✓ StaticArrays</span></span>
<span class="line"><span>    626.2 ms  ✓ DiffRules</span></span>
<span class="line"><span>    411.4 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    601.3 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>    644.3 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>    408.8 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>   1266.9 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>   2650.2 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    804.9 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>    601.2 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>   1604.1 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>   1302.5 ms  ✓ Optimisers</span></span>
<span class="line"><span>    447.8 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    431.5 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    443.1 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    484.7 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>    598.4 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    622.2 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    606.5 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    637.0 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    677.5 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    664.9 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    683.1 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>    897.9 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    458.6 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    431.5 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    435.3 ms  ✓ Optimisers → OptimisersAdaptExt</span></span>
<span class="line"><span>    920.7 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>    773.3 ms  ✓ Polyester</span></span>
<span class="line"><span>   3643.2 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    828.0 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   4127.4 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    664.8 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    768.8 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5584.0 ms  ✓ NNlib</span></span>
<span class="line"><span>    872.8 ms  ✓ NNlib → NNlibSpecialFunctionsExt</span></span>
<span class="line"><span>    892.0 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>   1021.3 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5490.1 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9406.3 ms  ✓ Lux</span></span>
<span class="line"><span>  88 dependencies successfully precompiled in 47 seconds. 17 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>    390.8 ms  ✓ Zlib_jll</span></span>
<span class="line"><span>    515.3 ms  ✓ TranscodingStreams</span></span>
<span class="line"><span>    534.5 ms  ✓ OrderedCollections</span></span>
<span class="line"><span>   4211.2 ms  ✓ FileIO</span></span>
<span class="line"><span>  31634.3 ms  ✓ JLD2</span></span>
<span class="line"><span>  5 dependencies successfully precompiled in 36 seconds. 27 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    322.0 ms  ✓ IteratorInterfaceExtensions</span></span>
<span class="line"><span>    440.8 ms  ✓ SuiteSparse_jll</span></span>
<span class="line"><span>    585.9 ms  ✓ Serialization</span></span>
<span class="line"><span>    319.5 ms  ✓ DataValueInterfaces</span></span>
<span class="line"><span>    462.1 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    364.6 ms  ✓ DataAPI</span></span>
<span class="line"><span>    404.3 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    419.5 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>   1118.2 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>    354.2 ms  ✓ TableTraits</span></span>
<span class="line"><span>   1670.9 ms  ✓ DataStructures</span></span>
<span class="line"><span>   2481.6 ms  ✓ Accessors</span></span>
<span class="line"><span>   1941.9 ms  ✓ Distributed</span></span>
<span class="line"><span>    450.8 ms  ✓ Missings</span></span>
<span class="line"><span>    601.6 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>    827.3 ms  ✓ Tables</span></span>
<span class="line"><span>    543.8 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>   3749.1 ms  ✓ SparseArrays</span></span>
<span class="line"><span>   3725.3 ms  ✓ Test</span></span>
<span class="line"><span>    671.2 ms  ✓ Accessors → StaticArraysExt</span></span>
<span class="line"><span>    910.0 ms  ✓ Accessors → LinearAlgebraExt</span></span>
<span class="line"><span>    650.9 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>   1147.9 ms  ✓ MLCore</span></span>
<span class="line"><span>    639.3 ms  ✓ Adapt → AdaptSparseArraysExt</span></span>
<span class="line"><span>    665.1 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>    596.9 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>    952.9 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>    666.5 ms  ✓ Accessors → TestExt</span></span>
<span class="line"><span>    753.7 ms  ✓ BangBang</span></span>
<span class="line"><span>   1301.5 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    505.8 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    712.4 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    499.9 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>   1059.5 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2279.7 ms  ✓ StatsBase</span></span>
<span class="line"><span>   2708.6 ms  ✓ Transducers</span></span>
<span class="line"><span>    681.0 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   5301.7 ms  ✓ FLoops</span></span>
<span class="line"><span>   5820.8 ms  ✓ MLUtils</span></span>
<span class="line"><span>  39 dependencies successfully precompiled in 26 seconds. 58 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceSparseArraysExt...</span></span>
<span class="line"><span>    619.7 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 8 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    664.3 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1478.1 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2037.1 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 164 already precompiled.</span></span>
<span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>    379.5 ms  ✓ CEnum</span></span>
<span class="line"><span>    386.6 ms  ✓ ExprTools</span></span>
<span class="line"><span>    725.1 ms  ✓ ExpressionExplorer</span></span>
<span class="line"><span>    379.7 ms  ✓ StructIO</span></span>
<span class="line"><span>    491.6 ms  ✓ CodecZlib</span></span>
<span class="line"><span>    534.1 ms  ✓ LoggingExtras</span></span>
<span class="line"><span>    524.6 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>    721.8 ms  ✓ ConcurrentUtilities</span></span>
<span class="line"><span>    630.7 ms  ✓ ReactantCore</span></span>
<span class="line"><span>   1065.9 ms  ✓ LazyArtifacts</span></span>
<span class="line"><span>   2058.7 ms  ✓ ObjectFile</span></span>
<span class="line"><span>   2747.4 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>   1434.5 ms  ✓ LLVMExtra_jll</span></span>
<span class="line"><span>   1483.3 ms  ✓ Enzyme_jll</span></span>
<span class="line"><span>   2329.5 ms  ✓ Reactant_jll</span></span>
<span class="line"><span>   6347.2 ms  ✓ LLVM</span></span>
<span class="line"><span>  19099.3 ms  ✓ HTTP</span></span>
<span class="line"><span>  27033.0 ms  ✓ GPUCompiler</span></span>
<span class="line"><span> 219204.6 ms  ✓ Enzyme</span></span>
<span class="line"><span>   5690.7 ms  ✓ Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>  76927.6 ms  ✓ Reactant</span></span>
<span class="line"><span>  21 dependencies successfully precompiled in 341 seconds. 58 already precompiled.</span></span>
<span class="line"><span>Precompiling UnsafeAtomicsLLVM...</span></span>
<span class="line"><span>   1773.5 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibEnzymeExt...</span></span>
<span class="line"><span>   6216.9 ms  ✓ Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>  10910.4 ms  ✓ Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>  11530.7 ms  ✓ Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   6047.0 ms  ✓ Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>   1365.7 ms  ✓ LuxLib → LuxLibEnzymeExt</span></span>
<span class="line"><span>  5 dependencies successfully precompiled in 13 seconds. 128 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   6800.7 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 148 already precompiled.</span></span>
<span class="line"><span>Precompiling HTTPExt...</span></span>
<span class="line"><span>   1774.8 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 43 already precompiled.</span></span>
<span class="line"><span>Precompiling OptimisersReactantExt...</span></span>
<span class="line"><span>  13048.0 ms  ✓ Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>  15572.1 ms  ✓ Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 16 seconds. 87 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  12635.5 ms  ✓ LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 84 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  12898.6 ms  ✓ MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 81 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  12704.4 ms  ✓ Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>  12964.2 ms  ✓ WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 13 seconds. 94 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantKernelAbstractionsExt...</span></span>
<span class="line"><span>  13353.5 ms  ✓ Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 91 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantArrayInterfaceExt...</span></span>
<span class="line"><span>  14414.6 ms  ✓ Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 15 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  13795.8 ms  ✓ Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 14 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  11581.9 ms  ✓ Lux → LuxReactantExt</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR1294/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L,C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR1294/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR1294/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR1294/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>2025-04-08 00:42:32.014823: I external/xla/xla/service/service.cc:152] XLA service 0x9fbcba0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-04-08 00:42:32.014964: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1744072952.016517  920194 se_gpu_pjrt_client.cc:1040] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1744072952.016749  920194 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1744072952.017087  920194 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1744072952.039443  920194 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>E0000 00:00:1744073011.130942  920194 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1744073011.131958  920194 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1744073011.131966  920194 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1744073011.131973  920194 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1744073011.131979  920194 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1744073011.131986  920194 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1744073011.131992  920194 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1744073011.131998  920194 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1744073011.132004  920194 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1744073011.132011  920194 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-08 00:43:31.132028: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.134852  920194 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1744073011.134905  920194 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1744073011.134912  920194 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1744073011.134919  920194 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1744073011.134925  920194 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1744073011.134931  920194 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1744073011.134938  920194 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1744073011.134944  920194 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1744073011.134950  920194 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1744073011.134956  920194 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-08 00:43:31.134971: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.137399  920194 buffer_comparator.cc:156] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1744073011.137437  920194 buffer_comparator.cc:156] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1744073011.137446  920194 buffer_comparator.cc:156] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1744073011.137450  920194 buffer_comparator.cc:156] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1744073011.137455  920194 buffer_comparator.cc:156] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1744073011.137459  920194 buffer_comparator.cc:156] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1744073011.137463  920194 buffer_comparator.cc:156] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1744073011.137467  920194 buffer_comparator.cc:156] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1744073011.137475  920194 buffer_comparator.cc:156] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1744073011.137479  920194 buffer_comparator.cc:156] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-04-08 00:43:31.137489: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.139850  920194 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1744073011.139883  920194 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1744073011.139888  920194 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1744073011.139892  920194 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1744073011.139897  920194 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1744073011.139901  920194 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1744073011.139905  920194 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1744073011.139909  920194 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1744073011.139913  920194 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1744073011.139917  920194 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-08 00:43:31.139926: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.142271  920194 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1744073011.142307  920194 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1744073011.142312  920194 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1744073011.142316  920194 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1744073011.142320  920194 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1744073011.142325  920194 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1744073011.142329  920194 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1744073011.142333  920194 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1744073011.142337  920194 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1744073011.142341  920194 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-08 00:43:31.142351: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.144706  920194 buffer_comparator.cc:156] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1744073011.144740  920194 buffer_comparator.cc:156] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1744073011.144749  920194 buffer_comparator.cc:156] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1744073011.144753  920194 buffer_comparator.cc:156] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1744073011.144758  920194 buffer_comparator.cc:156] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1744073011.144762  920194 buffer_comparator.cc:156] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1744073011.144766  920194 buffer_comparator.cc:156] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1744073011.144770  920194 buffer_comparator.cc:156] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1744073011.144775  920194 buffer_comparator.cc:156] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1744073011.144779  920194 buffer_comparator.cc:156] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-04-08 00:43:31.144788: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.147129  920194 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1744073011.147160  920194 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1744073011.147163  920194 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1744073011.147166  920194 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1744073011.147169  920194 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1744073011.147171  920194 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1744073011.147174  920194 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1744073011.147177  920194 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1744073011.147180  920194 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1744073011.147183  920194 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-08 00:43:31.147189: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.149414  920194 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1744073011.149442  920194 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1744073011.149445  920194 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1744073011.149448  920194 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1744073011.149451  920194 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1744073011.149454  920194 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1744073011.149457  920194 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1744073011.149459  920194 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1744073011.149462  920194 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1744073011.149465  920194 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-08 00:43:31.149472: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.151681  920194 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1744073011.151710  920194 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1744073011.151716  920194 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1744073011.151719  920194 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1744073011.151722  920194 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1744073011.151725  920194 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1744073011.151728  920194 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1744073011.151731  920194 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1744073011.151734  920194 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1744073011.151737  920194 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-08 00:43:31.151743: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.153951  920194 buffer_comparator.cc:156] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1744073011.153978  920194 buffer_comparator.cc:156] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1744073011.153981  920194 buffer_comparator.cc:156] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1744073011.153987  920194 buffer_comparator.cc:156] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1744073011.153990  920194 buffer_comparator.cc:156] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1744073011.153993  920194 buffer_comparator.cc:156] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1744073011.153996  920194 buffer_comparator.cc:156] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1744073011.153999  920194 buffer_comparator.cc:156] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1744073011.154001  920194 buffer_comparator.cc:156] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1744073011.154004  920194 buffer_comparator.cc:156] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-04-08 00:43:31.154011: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.156242  920194 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1744073011.156268  920194 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1744073011.156271  920194 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1744073011.156274  920194 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1744073011.156277  920194 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1744073011.156280  920194 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1744073011.156282  920194 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1744073011.156285  920194 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1744073011.156288  920194 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1744073011.156291  920194 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-08 00:43:31.156297: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.158485  920194 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1744073011.158509  920194 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1744073011.158512  920194 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1744073011.158515  920194 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1744073011.158518  920194 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1744073011.158521  920194 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1744073011.158524  920194 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1744073011.158526  920194 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1744073011.158529  920194 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1744073011.158532  920194 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-08 00:43:31.158539: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.160742  920194 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1744073011.160769  920194 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1744073011.160772  920194 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1744073011.160775  920194 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1744073011.160778  920194 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1744073011.160781  920194 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1744073011.160784  920194 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1744073011.160801  920194 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1744073011.160804  920194 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1744073011.160807  920194 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-08 00:43:31.160814: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.163039  920194 buffer_comparator.cc:156] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1744073011.163068  920194 buffer_comparator.cc:156] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1744073011.163072  920194 buffer_comparator.cc:156] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1744073011.163075  920194 buffer_comparator.cc:156] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1744073011.163078  920194 buffer_comparator.cc:156] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1744073011.163080  920194 buffer_comparator.cc:156] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1744073011.163083  920194 buffer_comparator.cc:156] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1744073011.163086  920194 buffer_comparator.cc:156] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1744073011.163089  920194 buffer_comparator.cc:156] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1744073011.163091  920194 buffer_comparator.cc:156] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-04-08 00:43:31.163099: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.165307  920194 buffer_comparator.cc:156] Difference at 256: 0, expected 0.86224</span></span>
<span class="line"><span>E0000 00:00:1744073011.165336  920194 buffer_comparator.cc:156] Difference at 257: 0, expected 0.686873</span></span>
<span class="line"><span>E0000 00:00:1744073011.165343  920194 buffer_comparator.cc:156] Difference at 258: 0, expected 0.252371</span></span>
<span class="line"><span>E0000 00:00:1744073011.165346  920194 buffer_comparator.cc:156] Difference at 259: 0, expected 0.335927</span></span>
<span class="line"><span>E0000 00:00:1744073011.165349  920194 buffer_comparator.cc:156] Difference at 260: 0, expected 0.934139</span></span>
<span class="line"><span>E0000 00:00:1744073011.165352  920194 buffer_comparator.cc:156] Difference at 261: 0, expected 0.274756</span></span>
<span class="line"><span>E0000 00:00:1744073011.165355  920194 buffer_comparator.cc:156] Difference at 262: 0, expected 0.529946</span></span>
<span class="line"><span>E0000 00:00:1744073011.165358  920194 buffer_comparator.cc:156] Difference at 263: 0, expected 0.542969</span></span>
<span class="line"><span>E0000 00:00:1744073011.165361  920194 buffer_comparator.cc:156] Difference at 264: 0, expected 0.895372</span></span>
<span class="line"><span>E0000 00:00:1744073011.165363  920194 buffer_comparator.cc:156] Difference at 265: 0, expected 0.895664</span></span>
<span class="line"><span>2025-04-08 00:43:31.165371: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.167647  920194 buffer_comparator.cc:156] Difference at 256: 0, expected 0.86224</span></span>
<span class="line"><span>E0000 00:00:1744073011.167675  920194 buffer_comparator.cc:156] Difference at 257: 0, expected 0.686873</span></span>
<span class="line"><span>E0000 00:00:1744073011.167678  920194 buffer_comparator.cc:156] Difference at 258: 0, expected 0.252371</span></span>
<span class="line"><span>E0000 00:00:1744073011.167681  920194 buffer_comparator.cc:156] Difference at 259: 0, expected 0.335927</span></span>
<span class="line"><span>E0000 00:00:1744073011.167684  920194 buffer_comparator.cc:156] Difference at 260: 0, expected 0.934139</span></span>
<span class="line"><span>E0000 00:00:1744073011.167687  920194 buffer_comparator.cc:156] Difference at 261: 0, expected 0.274756</span></span>
<span class="line"><span>E0000 00:00:1744073011.167689  920194 buffer_comparator.cc:156] Difference at 262: 0, expected 0.529946</span></span>
<span class="line"><span>E0000 00:00:1744073011.167692  920194 buffer_comparator.cc:156] Difference at 263: 0, expected 0.542969</span></span>
<span class="line"><span>E0000 00:00:1744073011.167695  920194 buffer_comparator.cc:156] Difference at 264: 0, expected 0.895372</span></span>
<span class="line"><span>E0000 00:00:1744073011.167698  920194 buffer_comparator.cc:156] Difference at 265: 0, expected 0.895664</span></span>
<span class="line"><span>2025-04-08 00:43:31.167705: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073011.171962  920194 buffer_comparator.cc:156] Difference at 32: -nan, expected 1.62244</span></span>
<span class="line"><span>E0000 00:00:1744073011.171998  920194 buffer_comparator.cc:156] Difference at 33: -nan, expected 1.87084</span></span>
<span class="line"><span>E0000 00:00:1744073011.172001  920194 buffer_comparator.cc:156] Difference at 34: -nan, expected 1.07351</span></span>
<span class="line"><span>E0000 00:00:1744073011.172004  920194 buffer_comparator.cc:156] Difference at 35: -nan, expected 2.92445</span></span>
<span class="line"><span>E0000 00:00:1744073011.172006  920194 buffer_comparator.cc:156] Difference at 36: -nan, expected 1.98056</span></span>
<span class="line"><span>E0000 00:00:1744073011.172009  920194 buffer_comparator.cc:156] Difference at 37: -nan, expected 2.07715</span></span>
<span class="line"><span>E0000 00:00:1744073011.172012  920194 buffer_comparator.cc:156] Difference at 38: -nan, expected 1.56458</span></span>
<span class="line"><span>E0000 00:00:1744073011.172014  920194 buffer_comparator.cc:156] Difference at 39: -nan, expected 2.27034</span></span>
<span class="line"><span>E0000 00:00:1744073011.172017  920194 buffer_comparator.cc:156] Difference at 40: -nan, expected 2.31795</span></span>
<span class="line"><span>E0000 00:00:1744073011.172020  920194 buffer_comparator.cc:156] Difference at 41: -nan, expected 2.55731</span></span>
<span class="line"><span>2025-04-08 00:43:31.172028: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.377549  920194 buffer_comparator.cc:156] Difference at 16: 3.81079, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1744073050.377604  920194 buffer_comparator.cc:156] Difference at 17: 3.78006, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1744073050.377610  920194 buffer_comparator.cc:156] Difference at 18: 4.08153, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1744073050.377615  920194 buffer_comparator.cc:156] Difference at 19: 2.92943, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1744073050.377620  920194 buffer_comparator.cc:156] Difference at 20: 4.69908, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1744073050.377624  920194 buffer_comparator.cc:156] Difference at 21: 2.01937, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1744073050.377628  920194 buffer_comparator.cc:156] Difference at 22: 4.96193, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1744073050.377632  920194 buffer_comparator.cc:156] Difference at 23: 1.04859, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1744073050.377637  920194 buffer_comparator.cc:156] Difference at 24: 4.87535, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1744073050.377641  920194 buffer_comparator.cc:156] Difference at 25: -0.0413231, expected 11.3838</span></span>
<span class="line"><span>2025-04-08 00:44:10.377658: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.379825  920194 buffer_comparator.cc:156] Difference at 16: 3.81079, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1744073050.379842  920194 buffer_comparator.cc:156] Difference at 17: 3.78006, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1744073050.379848  920194 buffer_comparator.cc:156] Difference at 18: 4.08153, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1744073050.379852  920194 buffer_comparator.cc:156] Difference at 19: 2.92943, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1744073050.379857  920194 buffer_comparator.cc:156] Difference at 20: 4.69908, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1744073050.379861  920194 buffer_comparator.cc:156] Difference at 21: 2.01937, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1744073050.379865  920194 buffer_comparator.cc:156] Difference at 22: 4.96193, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1744073050.379869  920194 buffer_comparator.cc:156] Difference at 23: 1.04859, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1744073050.379873  920194 buffer_comparator.cc:156] Difference at 24: 4.87535, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1744073050.379878  920194 buffer_comparator.cc:156] Difference at 25: -0.0413231, expected 11.3838</span></span>
<span class="line"><span>2025-04-08 00:44:10.379885: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.382015  920194 buffer_comparator.cc:156] Difference at 32: 3.41104, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1744073050.382031  920194 buffer_comparator.cc:156] Difference at 33: -3.2867, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1744073050.382038  920194 buffer_comparator.cc:156] Difference at 34: 2.54654, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1744073050.382043  920194 buffer_comparator.cc:156] Difference at 35: -3.76211, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1744073050.382047  920194 buffer_comparator.cc:156] Difference at 36: 1.54439, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1744073050.382060  920194 buffer_comparator.cc:156] Difference at 37: -4.24758, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1744073050.382064  920194 buffer_comparator.cc:156] Difference at 38: 0.803378, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1744073050.382069  920194 buffer_comparator.cc:156] Difference at 39: -4.48738, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1744073050.382073  920194 buffer_comparator.cc:156] Difference at 40: 0.173566, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1744073050.382078  920194 buffer_comparator.cc:156] Difference at 41: -4.54777, expected 8.63119</span></span>
<span class="line"><span>2025-04-08 00:44:10.382085: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.384209  920194 buffer_comparator.cc:156] Difference at 32: 3.41104, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1744073050.384225  920194 buffer_comparator.cc:156] Difference at 33: -3.2867, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1744073050.384230  920194 buffer_comparator.cc:156] Difference at 34: 2.54654, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1744073050.384235  920194 buffer_comparator.cc:156] Difference at 35: -3.76211, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1744073050.384239  920194 buffer_comparator.cc:156] Difference at 36: 1.54439, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1744073050.384244  920194 buffer_comparator.cc:156] Difference at 37: -4.24758, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1744073050.384248  920194 buffer_comparator.cc:156] Difference at 38: 0.803378, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1744073050.384253  920194 buffer_comparator.cc:156] Difference at 39: -4.48738, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1744073050.384257  920194 buffer_comparator.cc:156] Difference at 40: 0.173566, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1744073050.384261  920194 buffer_comparator.cc:156] Difference at 41: -4.54777, expected 8.63119</span></span>
<span class="line"><span>2025-04-08 00:44:10.384268: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.386386  920194 buffer_comparator.cc:156] Difference at 64: -2.74383, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1744073050.386398  920194 buffer_comparator.cc:156] Difference at 65: 2.60102, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1744073050.386401  920194 buffer_comparator.cc:156] Difference at 66: -1.99513, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1744073050.386404  920194 buffer_comparator.cc:156] Difference at 67: 3.10962, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1744073050.386407  920194 buffer_comparator.cc:156] Difference at 68: -1.38121, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1744073050.386410  920194 buffer_comparator.cc:156] Difference at 69: 3.38701, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1744073050.386413  920194 buffer_comparator.cc:156] Difference at 70: -0.786454, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1744073050.386416  920194 buffer_comparator.cc:156] Difference at 71: 3.4991, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1744073050.386419  920194 buffer_comparator.cc:156] Difference at 72: -0.0885482, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1744073050.386422  920194 buffer_comparator.cc:156] Difference at 73: 3.67801, expected 8.82565</span></span>
<span class="line"><span>2025-04-08 00:44:10.386427: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.388434  920194 buffer_comparator.cc:156] Difference at 64: -2.74383, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1744073050.388446  920194 buffer_comparator.cc:156] Difference at 65: 2.60102, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1744073050.388450  920194 buffer_comparator.cc:156] Difference at 66: -1.99513, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1744073050.388453  920194 buffer_comparator.cc:156] Difference at 67: 3.10962, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1744073050.388457  920194 buffer_comparator.cc:156] Difference at 68: -1.38121, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1744073050.388460  920194 buffer_comparator.cc:156] Difference at 69: 3.38701, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1744073050.388463  920194 buffer_comparator.cc:156] Difference at 70: -0.786454, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1744073050.388466  920194 buffer_comparator.cc:156] Difference at 71: 3.4991, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1744073050.388469  920194 buffer_comparator.cc:156] Difference at 72: -0.0885482, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1744073050.388472  920194 buffer_comparator.cc:156] Difference at 73: 3.67801, expected 8.82565</span></span>
<span class="line"><span>2025-04-08 00:44:10.388477: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.390476  920194 buffer_comparator.cc:156] Difference at 64: -2.74383, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1744073050.390488  920194 buffer_comparator.cc:156] Difference at 65: 2.60102, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1744073050.390492  920194 buffer_comparator.cc:156] Difference at 66: -1.99513, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1744073050.390495  920194 buffer_comparator.cc:156] Difference at 67: 3.10962, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1744073050.390498  920194 buffer_comparator.cc:156] Difference at 68: -1.38121, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1744073050.390501  920194 buffer_comparator.cc:156] Difference at 69: 3.38701, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1744073050.390504  920194 buffer_comparator.cc:156] Difference at 70: -0.786454, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1744073050.390507  920194 buffer_comparator.cc:156] Difference at 71: 3.4991, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1744073050.390510  920194 buffer_comparator.cc:156] Difference at 72: -0.0885482, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1744073050.390513  920194 buffer_comparator.cc:156] Difference at 73: 3.67801, expected 8.82565</span></span>
<span class="line"><span>2025-04-08 00:44:10.390517: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.401053  920194 buffer_comparator.cc:156] Difference at 16: 9.68745, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1744073050.401092  920194 buffer_comparator.cc:156] Difference at 17: 10.1876, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1744073050.401095  920194 buffer_comparator.cc:156] Difference at 18: 8.84104, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1744073050.401098  920194 buffer_comparator.cc:156] Difference at 19: 10.0381, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1744073050.401101  920194 buffer_comparator.cc:156] Difference at 20: 7.30446, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1744073050.401104  920194 buffer_comparator.cc:156] Difference at 21: 8.26483, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1744073050.401107  920194 buffer_comparator.cc:156] Difference at 22: 10.8549, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1744073050.401110  920194 buffer_comparator.cc:156] Difference at 23: 7.87482, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1744073050.401113  920194 buffer_comparator.cc:156] Difference at 24: 9.78239, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1744073050.401116  920194 buffer_comparator.cc:156] Difference at 25: 11.3838, expected 36.4575</span></span>
<span class="line"><span>2025-04-08 00:44:10.401123: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.403119  920194 buffer_comparator.cc:156] Difference at 16: 9.68745, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1744073050.403135  920194 buffer_comparator.cc:156] Difference at 17: 10.1876, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1744073050.403138  920194 buffer_comparator.cc:156] Difference at 18: 8.84104, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1744073050.403141  920194 buffer_comparator.cc:156] Difference at 19: 10.0381, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1744073050.403144  920194 buffer_comparator.cc:156] Difference at 20: 7.30446, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1744073050.403147  920194 buffer_comparator.cc:156] Difference at 21: 8.26483, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1744073050.403151  920194 buffer_comparator.cc:156] Difference at 22: 10.8549, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1744073050.403154  920194 buffer_comparator.cc:156] Difference at 23: 7.87482, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1744073050.403157  920194 buffer_comparator.cc:156] Difference at 24: 9.78239, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1744073050.403160  920194 buffer_comparator.cc:156] Difference at 25: 11.3838, expected 36.4575</span></span>
<span class="line"><span>2025-04-08 00:44:10.403165: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.414982  920194 buffer_comparator.cc:156] Difference at 2: 37.361, expected 33.2434</span></span>
<span class="line"><span>E0000 00:00:1744073050.414998  920194 buffer_comparator.cc:156] Difference at 8: 32.9182, expected 29.0801</span></span>
<span class="line"><span>E0000 00:00:1744073050.415002  920194 buffer_comparator.cc:156] Difference at 11: 35.3152, expected 30.7625</span></span>
<span class="line"><span>E0000 00:00:1744073050.415005  920194 buffer_comparator.cc:156] Difference at 12: 39.5233, expected 34.3637</span></span>
<span class="line"><span>E0000 00:00:1744073050.415008  920194 buffer_comparator.cc:156] Difference at 20: 38.835, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1744073050.415011  920194 buffer_comparator.cc:156] Difference at 23: 37.0226, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1744073050.415014  920194 buffer_comparator.cc:156] Difference at 26: 39.1599, expected 32.4927</span></span>
<span class="line"><span>E0000 00:00:1744073050.415017  920194 buffer_comparator.cc:156] Difference at 51: 26.8307, expected 33.7879</span></span>
<span class="line"><span>2025-04-08 00:44:10.415022: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.423588  920194 buffer_comparator.cc:156] Difference at 16: 6.73254, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1744073050.423603  920194 buffer_comparator.cc:156] Difference at 17: 9.2143, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1744073050.423607  920194 buffer_comparator.cc:156] Difference at 18: 6.85711, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1744073050.423610  920194 buffer_comparator.cc:156] Difference at 19: 7.1738, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1744073050.423613  920194 buffer_comparator.cc:156] Difference at 20: 6.9796, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1744073050.423616  920194 buffer_comparator.cc:156] Difference at 21: 7.31867, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1744073050.423619  920194 buffer_comparator.cc:156] Difference at 22: 8.70739, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1744073050.423621  920194 buffer_comparator.cc:156] Difference at 23: 7.19731, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1744073050.423624  920194 buffer_comparator.cc:156] Difference at 24: 8.21217, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1744073050.423627  920194 buffer_comparator.cc:156] Difference at 25: 9.32022, expected 36.0917</span></span>
<span class="line"><span>2025-04-08 00:44:10.423632: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.425612  920194 buffer_comparator.cc:156] Difference at 16: 6.73254, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1744073050.425627  920194 buffer_comparator.cc:156] Difference at 17: 9.2143, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1744073050.425630  920194 buffer_comparator.cc:156] Difference at 18: 6.85711, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1744073050.425634  920194 buffer_comparator.cc:156] Difference at 19: 7.1738, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1744073050.425637  920194 buffer_comparator.cc:156] Difference at 20: 6.9796, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1744073050.425639  920194 buffer_comparator.cc:156] Difference at 21: 7.31867, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1744073050.425642  920194 buffer_comparator.cc:156] Difference at 22: 8.70739, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1744073050.425645  920194 buffer_comparator.cc:156] Difference at 23: 7.19731, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1744073050.425648  920194 buffer_comparator.cc:156] Difference at 24: 8.21217, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1744073050.425651  920194 buffer_comparator.cc:156] Difference at 25: 9.32022, expected 36.0917</span></span>
<span class="line"><span>2025-04-08 00:44:10.425666: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744073050.437435  920194 buffer_comparator.cc:156] Difference at 2: 38.3457, expected 34.2806</span></span>
<span class="line"><span>E0000 00:00:1744073050.437453  920194 buffer_comparator.cc:156] Difference at 6: 41.1731, expected 36.7103</span></span>
<span class="line"><span>E0000 00:00:1744073050.437456  920194 buffer_comparator.cc:156] Difference at 13: 31.5951, expected 35.7459</span></span>
<span class="line"><span>E0000 00:00:1744073050.437460  920194 buffer_comparator.cc:156] Difference at 17: 37.0853, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1744073050.437463  920194 buffer_comparator.cc:156] Difference at 20: 37.9014, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1744073050.437466  920194 buffer_comparator.cc:156] Difference at 45: 25.9105, expected 32.5352</span></span>
<span class="line"><span>E0000 00:00:1744073050.437469  920194 buffer_comparator.cc:156] Difference at 75: 24.7101, expected 28.3085</span></span>
<span class="line"><span>E0000 00:00:1744073050.437472  920194 buffer_comparator.cc:156] Difference at 77: 19.521, expected 27.4887</span></span>
<span class="line"><span>E0000 00:00:1744073050.437475  920194 buffer_comparator.cc:156] Difference at 94: 24.541, expected 28.5145</span></span>
<span class="line"><span>E0000 00:00:1744073050.437478  920194 buffer_comparator.cc:156] Difference at 101: 30.6158, expected 26.8436</span></span>
<span class="line"><span>2025-04-08 00:44:10.437483: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48581</span></span>
<span class="line"><span>Validation:	Loss 0.42820	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38694</span></span>
<span class="line"><span>Validation:	Loss 0.31694	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26742</span></span>
<span class="line"><span>Validation:	Loss 0.21165	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18467</span></span>
<span class="line"><span>Validation:	Loss 0.15020	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12923</span></span>
<span class="line"><span>Validation:	Loss 0.10441	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08910</span></span>
<span class="line"><span>Validation:	Loss 0.07212	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06107</span></span>
<span class="line"><span>Validation:	Loss 0.05004	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04251</span></span>
<span class="line"><span>Validation:	Loss 0.03531	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03012</span></span>
<span class="line"><span>Validation:	Loss 0.02542	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02190</span></span>
<span class="line"><span>Validation:	Loss 0.01882	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01656</span></span>
<span class="line"><span>Validation:	Loss 0.01473	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01335</span></span>
<span class="line"><span>Validation:	Loss 0.01223	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01133</span></span>
<span class="line"><span>Validation:	Loss 0.01055	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.00989</span></span>
<span class="line"><span>Validation:	Loss 0.00934	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.00882</span></span>
<span class="line"><span>Validation:	Loss 0.00841	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00797</span></span>
<span class="line"><span>Validation:	Loss 0.00766	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00728</span></span>
<span class="line"><span>Validation:	Loss 0.00703	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00669</span></span>
<span class="line"><span>Validation:	Loss 0.00650	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00621</span></span>
<span class="line"><span>Validation:	Loss 0.00605	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00576</span></span>
<span class="line"><span>Validation:	Loss 0.00564	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00539</span></span>
<span class="line"><span>Validation:	Loss 0.00529	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00505</span></span>
<span class="line"><span>Validation:	Loss 0.00497	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00474</span></span>
<span class="line"><span>Validation:	Loss 0.00468	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00446</span></span>
<span class="line"><span>Validation:	Loss 0.00442	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00422</span></span>
<span class="line"><span>Validation:	Loss 0.00418	Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.69313</span></span>
<span class="line"><span>Validation:	Loss 0.57819	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.50491</span></span>
<span class="line"><span>Validation:	Loss 0.41223	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36635</span></span>
<span class="line"><span>Validation:	Loss 0.30801	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27280</span></span>
<span class="line"><span>Validation:	Loss 0.23197	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20048</span></span>
<span class="line"><span>Validation:	Loss 0.16895	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14858</span></span>
<span class="line"><span>Validation:	Loss 0.13081	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11571</span></span>
<span class="line"><span>Validation:	Loss 0.10153	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09067</span></span>
<span class="line"><span>Validation:	Loss 0.07905	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.07140</span></span>
<span class="line"><span>Validation:	Loss 0.06272	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.05750</span></span>
<span class="line"><span>Validation:	Loss 0.05125	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.04765</span></span>
<span class="line"><span>Validation:	Loss 0.04327	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.04070</span></span>
<span class="line"><span>Validation:	Loss 0.03756	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.03560</span></span>
<span class="line"><span>Validation:	Loss 0.03328	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.03172</span></span>
<span class="line"><span>Validation:	Loss 0.02995	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02864</span></span>
<span class="line"><span>Validation:	Loss 0.02726	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02612</span></span>
<span class="line"><span>Validation:	Loss 0.02501	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.02402</span></span>
<span class="line"><span>Validation:	Loss 0.02310	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.02221</span></span>
<span class="line"><span>Validation:	Loss 0.02145	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.02065</span></span>
<span class="line"><span>Validation:	Loss 0.02000	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01926</span></span>
<span class="line"><span>Validation:	Loss 0.01871	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01806</span></span>
<span class="line"><span>Validation:	Loss 0.01757	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01697</span></span>
<span class="line"><span>Validation:	Loss 0.01654	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01597</span></span>
<span class="line"><span>Validation:	Loss 0.01561	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01508</span></span>
<span class="line"><span>Validation:	Loss 0.01476	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.01428</span></span>
<span class="line"><span>Validation:	Loss 0.01399	Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
