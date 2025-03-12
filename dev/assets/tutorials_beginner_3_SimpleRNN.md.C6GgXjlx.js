import{_ as a,c as n,o as i,al as p}from"./chunks/framework.BCN3FD2k.js";const E=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(t,s,h,k,r,c){return i(),n("div",null,s[0]||(s[0]=[p(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><p>Note: If you wish to use AutoZygote() for automatic differentiation, add Zygote to your project dependencies and include <code>using Zygote</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling ADTypes...</span></span>
<span class="line"><span>    666.2 ms  ✓ ADTypes</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds</span></span>
<span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    347.2 ms  ✓ ConcreteStructs</span></span>
<span class="line"><span>    357.7 ms  ✓ Future</span></span>
<span class="line"><span>    404.5 ms  ✓ OpenLibm_jll</span></span>
<span class="line"><span>    424.8 ms  ✓ CEnum</span></span>
<span class="line"><span>    422.2 ms  ✓ ArgCheck</span></span>
<span class="line"><span>    544.7 ms  ✓ Statistics</span></span>
<span class="line"><span>    403.6 ms  ✓ ManualMemory</span></span>
<span class="line"><span>   1906.1 ms  ✓ UnsafeAtomics</span></span>
<span class="line"><span>    358.3 ms  ✓ Reexport</span></span>
<span class="line"><span>    357.0 ms  ✓ SIMDTypes</span></span>
<span class="line"><span>    577.2 ms  ✓ EnzymeCore</span></span>
<span class="line"><span>    335.2 ms  ✓ IfElse</span></span>
<span class="line"><span>   1142.6 ms  ✓ IrrationalConstants</span></span>
<span class="line"><span>    361.9 ms  ✓ CommonWorldInvalidations</span></span>
<span class="line"><span>    347.1 ms  ✓ FastClosures</span></span>
<span class="line"><span>   2532.5 ms  ✓ MacroTools</span></span>
<span class="line"><span>    395.7 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>    398.0 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>    552.9 ms  ✓ ArrayInterface</span></span>
<span class="line"><span>    505.7 ms  ✓ GPUArraysCore</span></span>
<span class="line"><span>    665.8 ms  ✓ CpuId</span></span>
<span class="line"><span>    663.7 ms  ✓ DocStringExtensions</span></span>
<span class="line"><span>    529.5 ms  ✓ JLLWrappers</span></span>
<span class="line"><span>    496.4 ms  ✓ NaNMath</span></span>
<span class="line"><span>    859.7 ms  ✓ ThreadingUtilities</span></span>
<span class="line"><span>   1301.1 ms  ✓ ChainRulesCore</span></span>
<span class="line"><span>    521.1 ms  ✓ Atomix</span></span>
<span class="line"><span>    388.8 ms  ✓ EnzymeCore → AdaptExt</span></span>
<span class="line"><span>    400.1 ms  ✓ ADTypes → ADTypesEnzymeCoreExt</span></span>
<span class="line"><span>    792.2 ms  ✓ Static</span></span>
<span class="line"><span>    835.8 ms  ✓ CommonSubexpressions</span></span>
<span class="line"><span>    453.3 ms  ✓ DiffResults</span></span>
<span class="line"><span>   1778.5 ms  ✓ DispatchDoctor</span></span>
<span class="line"><span>    383.5 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>   1626.5 ms  ✓ Setfield</span></span>
<span class="line"><span>    392.7 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>    633.1 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    629.7 ms  ✓ Hwloc_jll</span></span>
<span class="line"><span>    438.9 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    656.5 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>    726.8 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>   1283.5 ms  ✓ Optimisers</span></span>
<span class="line"><span>    420.0 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>    415.9 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>   1576.9 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>   1227.4 ms  ✓ CPUSummary</span></span>
<span class="line"><span>    441.5 ms  ✓ DispatchDoctor → DispatchDoctorEnzymeCoreExt</span></span>
<span class="line"><span>    672.3 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>   1244.5 ms  ✓ LuxCore</span></span>
<span class="line"><span>   7792.6 ms  ✓ StaticArrays</span></span>
<span class="line"><span>   1384.2 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    555.9 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    560.5 ms  ✓ Optimisers → OptimisersAdaptExt</span></span>
<span class="line"><span>    507.4 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>   2199.4 ms  ✓ Hwloc</span></span>
<span class="line"><span>    615.0 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    624.3 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>   2746.6 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    646.4 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>    479.0 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    459.6 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    480.9 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    493.6 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    666.3 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    635.7 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    654.8 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    631.6 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    707.7 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    950.8 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>    644.3 ms  ✓ DiffRules</span></span>
<span class="line"><span>   1806.7 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    731.3 ms  ✓ Polyester</span></span>
<span class="line"><span>   2801.2 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    960.5 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>   4259.9 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    675.4 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    741.1 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   3863.5 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    863.2 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   5779.5 ms  ✓ NNlib</span></span>
<span class="line"><span>    856.5 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    912.4 ms  ✓ NNlib → NNlibSpecialFunctionsExt</span></span>
<span class="line"><span>   1037.1 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   6341.1 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9838.2 ms  ✓ Lux</span></span>
<span class="line"><span>  85 dependencies successfully precompiled in 49 seconds. 25 already precompiled.</span></span>
<span class="line"><span>Precompiling JLD2...</span></span>
<span class="line"><span>    413.1 ms  ✓ Zlib_jll</span></span>
<span class="line"><span>    554.9 ms  ✓ TranscodingStreams</span></span>
<span class="line"><span>    571.1 ms  ✓ OrderedCollections</span></span>
<span class="line"><span>   4226.6 ms  ✓ FileIO</span></span>
<span class="line"><span>  32656.6 ms  ✓ JLD2</span></span>
<span class="line"><span>  5 dependencies successfully precompiled in 37 seconds. 27 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    331.6 ms  ✓ IteratorInterfaceExtensions</span></span>
<span class="line"><span>    409.7 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    447.6 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    872.8 ms  ✓ InitialValues</span></span>
<span class="line"><span>    603.0 ms  ✓ Serialization</span></span>
<span class="line"><span>    398.4 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    454.7 ms  ✓ ShowCases</span></span>
<span class="line"><span>    460.5 ms  ✓ SuiteSparse_jll</span></span>
<span class="line"><span>    344.1 ms  ✓ DataValueInterfaces</span></span>
<span class="line"><span>    374.3 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    367.9 ms  ✓ PtrArrays</span></span>
<span class="line"><span>    363.8 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    468.8 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    383.9 ms  ✓ DataAPI</span></span>
<span class="line"><span>   1062.1 ms  ✓ Baselet</span></span>
<span class="line"><span>   1191.7 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>    440.1 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>    375.3 ms  ✓ TableTraits</span></span>
<span class="line"><span>    444.4 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>    495.4 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>   1759.5 ms  ✓ DataStructures</span></span>
<span class="line"><span>   1965.3 ms  ✓ Distributed</span></span>
<span class="line"><span>    418.3 ms  ✓ NameResolution</span></span>
<span class="line"><span>   3840.3 ms  ✓ Test</span></span>
<span class="line"><span>    422.1 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>    476.8 ms  ✓ AliasTables</span></span>
<span class="line"><span>    452.2 ms  ✓ Missings</span></span>
<span class="line"><span>    640.1 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>   3808.4 ms  ✓ SparseArrays</span></span>
<span class="line"><span>    529.2 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>    875.6 ms  ✓ Tables</span></span>
<span class="line"><span>    618.3 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>   1323.3 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    704.0 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>    643.1 ms  ✓ Adapt → AdaptSparseArraysExt</span></span>
<span class="line"><span>   2458.5 ms  ✓ Accessors</span></span>
<span class="line"><span>    723.3 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>    991.1 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   1132.5 ms  ✓ MLCore</span></span>
<span class="line"><span>    961.5 ms  ✓ Accessors → LinearAlgebraExt</span></span>
<span class="line"><span>    749.2 ms  ✓ Accessors → TestExt</span></span>
<span class="line"><span>   2546.5 ms  ✓ StatsBase</span></span>
<span class="line"><span>    700.0 ms  ✓ Accessors → StaticArraysExt</span></span>
<span class="line"><span>    949.5 ms  ✓ BangBang</span></span>
<span class="line"><span>  18179.7 ms  ✓ MLStyle</span></span>
<span class="line"><span>    748.3 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    752.9 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    599.1 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>   1195.6 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   3000.5 ms  ✓ Transducers</span></span>
<span class="line"><span>   4865.9 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>    709.6 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   5415.7 ms  ✓ FLoops</span></span>
<span class="line"><span>   5900.1 ms  ✓ MLUtils</span></span>
<span class="line"><span>  54 dependencies successfully precompiled in 37 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceSparseArraysExt...</span></span>
<span class="line"><span>    640.7 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 8 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    683.3 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1635.3 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2202.4 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 170 already precompiled.</span></span>
<span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>    369.5 ms  ✓ EnumX</span></span>
<span class="line"><span>    408.1 ms  ✓ ExprTools</span></span>
<span class="line"><span>    692.8 ms  ✓ ExpressionExplorer</span></span>
<span class="line"><span>    400.2 ms  ✓ StructIO</span></span>
<span class="line"><span>    431.3 ms  ✓ Scratch</span></span>
<span class="line"><span>    635.7 ms  ✓ LLVMOpenMP_jll</span></span>
<span class="line"><span>   1192.3 ms  ✓ CUDA_Driver_jll</span></span>
<span class="line"><span>   1087.0 ms  ✓ LazyArtifacts</span></span>
<span class="line"><span>    640.7 ms  ✓ ReactantCore</span></span>
<span class="line"><span>   1451.7 ms  ✓ Enzyme_jll</span></span>
<span class="line"><span>   2061.4 ms  ✓ ObjectFile</span></span>
<span class="line"><span>   2763.1 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>   1499.6 ms  ✓ LLVMExtra_jll</span></span>
<span class="line"><span>   2312.3 ms  ✓ Reactant_jll</span></span>
<span class="line"><span>   6340.1 ms  ✓ LLVM</span></span>
<span class="line"><span>  26510.6 ms  ✓ GPUCompiler</span></span>
<span class="line"><span> 218633.7 ms  ✓ Enzyme</span></span>
<span class="line"><span>   5760.2 ms  ✓ Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>  71179.4 ms  ✓ Reactant</span></span>
<span class="line"><span>  19 dependencies successfully precompiled in 334 seconds. 45 already precompiled.</span></span>
<span class="line"><span>Precompiling UnsafeAtomicsLLVM...</span></span>
<span class="line"><span>   1874.9 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibEnzymeExt...</span></span>
<span class="line"><span>   6106.9 ms  ✓ Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   8491.3 ms  ✓ Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   1424.9 ms  ✓ LuxLib → LuxLibEnzymeExt</span></span>
<span class="line"><span>  11519.1 ms  ✓ Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   6064.8 ms  ✓ Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>  5 dependencies successfully precompiled in 13 seconds. 128 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   6827.0 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 148 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  12020.6 ms  ✓ LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 12 seconds. 69 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  12407.5 ms  ✓ MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 66 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibReactantExt...</span></span>
<span class="line"><span>  12417.0 ms  ✓ Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>  13301.7 ms  ✓ LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>  13591.9 ms  ✓ Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>  13081.1 ms  ✓ Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>  11952.3 ms  ✓ Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>  12539.3 ms  ✓ Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>  6 dependencies successfully precompiled in 26 seconds. 143 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  12070.1 ms  ✓ WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 12 seconds. 80 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  10345.6 ms  ✓ Lux → LuxReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 11 seconds. 166 already precompiled.</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##230&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> st_classifier, lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> st_lstm))</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), model) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples) (total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-8/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>2025-03-11 22:46:33.618744: I external/xla/xla/service/service.cc:152] XLA service 0x8941cc0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-11 22:46:33.619371: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1741733193.621299  999918 se_gpu_pjrt_client.cc:951] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1741733193.621452  999918 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741733193.621823  999918 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741733193.640362  999918 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.74438</span></span>
<span class="line"><span>Validation:	Loss 0.68827	Accuracy 0.50000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.64720</span></span>
<span class="line"><span>Validation:	Loss 0.58855	Accuracy 0.74219</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.54445</span></span>
<span class="line"><span>Validation:	Loss 0.48560	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.44412</span></span>
<span class="line"><span>Validation:	Loss 0.39013	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.36092</span></span>
<span class="line"><span>Validation:	Loss 0.32314	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.30250</span></span>
<span class="line"><span>Validation:	Loss 0.27289	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.25654</span></span>
<span class="line"><span>Validation:	Loss 0.23241	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.21793</span></span>
<span class="line"><span>Validation:	Loss 0.19780	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.18569</span></span>
<span class="line"><span>Validation:	Loss 0.16624	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.15337</span></span>
<span class="line"><span>Validation:	Loss 0.13069	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.11425</span></span>
<span class="line"><span>Validation:	Loss 0.09165	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.07697</span></span>
<span class="line"><span>Validation:	Loss 0.05846	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.04897</span></span>
<span class="line"><span>Validation:	Loss 0.03708	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.03152</span></span>
<span class="line"><span>Validation:	Loss 0.02527	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02241</span></span>
<span class="line"><span>Validation:	Loss 0.01909	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01754</span></span>
<span class="line"><span>Validation:	Loss 0.01565	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01460</span></span>
<span class="line"><span>Validation:	Loss 0.01329	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01250</span></span>
<span class="line"><span>Validation:	Loss 0.01154	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01090</span></span>
<span class="line"><span>Validation:	Loss 0.01009	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00951</span></span>
<span class="line"><span>Validation:	Loss 0.00878	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00828</span></span>
<span class="line"><span>Validation:	Loss 0.00768	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00728</span></span>
<span class="line"><span>Validation:	Loss 0.00691	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00666</span></span>
<span class="line"><span>Validation:	Loss 0.00637	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00617</span></span>
<span class="line"><span>Validation:	Loss 0.00595	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00580</span></span>
<span class="line"><span>Validation:	Loss 0.00558	Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-8/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.69127</span></span>
<span class="line"><span>Validation:	Loss 0.62883	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.59773</span></span>
<span class="line"><span>Validation:	Loss 0.55691	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.53535</span></span>
<span class="line"><span>Validation:	Loss 0.50087	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.47979</span></span>
<span class="line"><span>Validation:	Loss 0.45229	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.43386</span></span>
<span class="line"><span>Validation:	Loss 0.40632	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.38648</span></span>
<span class="line"><span>Validation:	Loss 0.35964	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.33984</span></span>
<span class="line"><span>Validation:	Loss 0.30956	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.28701</span></span>
<span class="line"><span>Validation:	Loss 0.25509	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.23092</span></span>
<span class="line"><span>Validation:	Loss 0.19590	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.17012</span></span>
<span class="line"><span>Validation:	Loss 0.13502	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.11352</span></span>
<span class="line"><span>Validation:	Loss 0.08692	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.07430</span></span>
<span class="line"><span>Validation:	Loss 0.06068	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.05426</span></span>
<span class="line"><span>Validation:	Loss 0.04693	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.04305</span></span>
<span class="line"><span>Validation:	Loss 0.03859	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.03607</span></span>
<span class="line"><span>Validation:	Loss 0.03279	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.03084</span></span>
<span class="line"><span>Validation:	Loss 0.02864	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.02733</span></span>
<span class="line"><span>Validation:	Loss 0.02545	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.02431</span></span>
<span class="line"><span>Validation:	Loss 0.02293	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.02205</span></span>
<span class="line"><span>Validation:	Loss 0.02087	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.02008</span></span>
<span class="line"><span>Validation:	Loss 0.01913	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01852</span></span>
<span class="line"><span>Validation:	Loss 0.01766	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01712</span></span>
<span class="line"><span>Validation:	Loss 0.01638	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01588</span></span>
<span class="line"><span>Validation:	Loss 0.01527	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01479</span></span>
<span class="line"><span>Validation:	Loss 0.01428	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.01393</span></span>
<span class="line"><span>Validation:	Loss 0.01341	Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,47)]))}const o=a(l,[["render",e]]);export{E as __pageData,o as default};
