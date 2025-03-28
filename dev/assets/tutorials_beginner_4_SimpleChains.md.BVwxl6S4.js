import{_ as a,c as n,o as i,al as p}from"./chunks/framework.BCN3FD2k.js";const E=JSON.parse('{"title":"MNIST Classification with SimpleChains","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/4_SimpleChains.md","filePath":"tutorials/beginner/4_SimpleChains.md","lastUpdated":null}'),l={name:"tutorials/beginner/4_SimpleChains.md"};function e(t,s,h,k,r,c){return i(),n("div",null,s[0]||(s[0]=[p(`<h1 id="MNIST-Classification-with-SimpleChains" tabindex="-1">MNIST Classification with SimpleChains <a class="header-anchor" href="#MNIST-Classification-with-SimpleChains" aria-label="Permalink to &quot;MNIST Classification with SimpleChains {#MNIST-Classification-with-SimpleChains}&quot;">​</a></h1><p>SimpleChains.jl is an excellent framework for training small neural networks. In this tutorial we will demonstrate how to use the same API as Lux.jl to train a model using SimpleChains.jl. We will use the tutorial from <a href="https://pumasai.github.io/SimpleChains.jl/dev/examples/mnist/" target="_blank" rel="noreferrer">SimpleChains.jl</a> as a reference.</p><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Statistics, Printf, Reactant</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDatasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MNIST</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SimpleChains</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SimpleChains</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_default_backend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;cpu&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    416.2 ms  ✓ ConcreteStructs</span></span>
<span class="line"><span>    370.1 ms  ✓ Future</span></span>
<span class="line"><span>    366.0 ms  ✓ OpenLibm_jll</span></span>
<span class="line"><span>    589.9 ms  ✓ ADTypes</span></span>
<span class="line"><span>    509.6 ms  ✓ Statistics</span></span>
<span class="line"><span>    381.4 ms  ✓ ArgCheck</span></span>
<span class="line"><span>    368.1 ms  ✓ ManualMemory</span></span>
<span class="line"><span>   1781.7 ms  ✓ UnsafeAtomics</span></span>
<span class="line"><span>    326.4 ms  ✓ Reexport</span></span>
<span class="line"><span>    310.3 ms  ✓ SIMDTypes</span></span>
<span class="line"><span>    533.2 ms  ✓ EnzymeCore</span></span>
<span class="line"><span>    312.6 ms  ✓ IfElse</span></span>
<span class="line"><span>   1101.8 ms  ✓ IrrationalConstants</span></span>
<span class="line"><span>    338.7 ms  ✓ CommonWorldInvalidations</span></span>
<span class="line"><span>    342.6 ms  ✓ FastClosures</span></span>
<span class="line"><span>   2333.0 ms  ✓ MacroTools</span></span>
<span class="line"><span>    377.6 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>    559.7 ms  ✓ ArrayInterface</span></span>
<span class="line"><span>    471.3 ms  ✓ NaNMath</span></span>
<span class="line"><span>    432.3 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    419.3 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>    504.8 ms  ✓ Atomix</span></span>
<span class="line"><span>    370.8 ms  ✓ EnzymeCore → AdaptExt</span></span>
<span class="line"><span>    791.4 ms  ✓ ThreadingUtilities</span></span>
<span class="line"><span>    385.5 ms  ✓ ADTypes → ADTypesEnzymeCoreExt</span></span>
<span class="line"><span>    439.8 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    610.3 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    782.1 ms  ✓ Static</span></span>
<span class="line"><span>    702.4 ms  ✓ CommonSubexpressions</span></span>
<span class="line"><span>    404.4 ms  ✓ DiffResults</span></span>
<span class="line"><span>   1560.7 ms  ✓ DispatchDoctor</span></span>
<span class="line"><span>    365.2 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>   1444.1 ms  ✓ Setfield</span></span>
<span class="line"><span>    408.6 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>    370.4 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>   1316.3 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    422.5 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>   2520.5 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>   1097.4 ms  ✓ CPUSummary</span></span>
<span class="line"><span>    417.6 ms  ✓ DispatchDoctor → DispatchDoctorEnzymeCoreExt</span></span>
<span class="line"><span>    652.7 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>   1551.0 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>   1188.9 ms  ✓ LuxCore</span></span>
<span class="line"><span>   7248.3 ms  ✓ StaticArrays</span></span>
<span class="line"><span>   1756.2 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    645.3 ms  ✓ DiffRules</span></span>
<span class="line"><span>    507.5 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>    635.9 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>    603.1 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    635.6 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>    478.1 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    424.1 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>   2826.8 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    461.4 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    460.9 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    621.3 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    603.0 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    624.7 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    594.5 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    682.5 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    936.2 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>   1003.7 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    726.1 ms  ✓ Polyester</span></span>
<span class="line"><span>   3786.7 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    892.0 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   3997.0 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    740.2 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    755.8 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5623.2 ms  ✓ NNlib</span></span>
<span class="line"><span>    861.4 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    917.3 ms  ✓ NNlib → NNlibSpecialFunctionsExt</span></span>
<span class="line"><span>    988.2 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5652.1 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9735.7 ms  ✓ Lux</span></span>
<span class="line"><span>  74 dependencies successfully precompiled in 45 seconds. 33 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    321.1 ms  ✓ IteratorInterfaceExtensions</span></span>
<span class="line"><span>    387.0 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    430.3 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    850.2 ms  ✓ InitialValues</span></span>
<span class="line"><span>    595.9 ms  ✓ Serialization</span></span>
<span class="line"><span>    381.2 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    427.9 ms  ✓ ShowCases</span></span>
<span class="line"><span>    461.2 ms  ✓ SuiteSparse_jll</span></span>
<span class="line"><span>    331.3 ms  ✓ DataValueInterfaces</span></span>
<span class="line"><span>    365.7 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    570.5 ms  ✓ OrderedCollections</span></span>
<span class="line"><span>    334.5 ms  ✓ PtrArrays</span></span>
<span class="line"><span>    356.4 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    504.2 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    379.6 ms  ✓ DataAPI</span></span>
<span class="line"><span>   1084.4 ms  ✓ Baselet</span></span>
<span class="line"><span>   1201.9 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>    356.1 ms  ✓ TableTraits</span></span>
<span class="line"><span>    425.3 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>    459.8 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>   2028.9 ms  ✓ Distributed</span></span>
<span class="line"><span>    419.0 ms  ✓ NameResolution</span></span>
<span class="line"><span>   3674.1 ms  ✓ Test</span></span>
<span class="line"><span>    402.9 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>   1693.7 ms  ✓ DataStructures</span></span>
<span class="line"><span>    509.9 ms  ✓ AliasTables</span></span>
<span class="line"><span>   3892.7 ms  ✓ SparseArrays</span></span>
<span class="line"><span>    456.9 ms  ✓ Missings</span></span>
<span class="line"><span>    805.2 ms  ✓ Tables</span></span>
<span class="line"><span>    649.5 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>   1344.3 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    534.4 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>    649.9 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>    663.8 ms  ✓ Adapt → AdaptSparseArraysExt</span></span>
<span class="line"><span>   2622.8 ms  ✓ Accessors</span></span>
<span class="line"><span>    702.3 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>    966.6 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   1099.5 ms  ✓ MLCore</span></span>
<span class="line"><span>    900.1 ms  ✓ Accessors → LinearAlgebraExt</span></span>
<span class="line"><span>    659.8 ms  ✓ Accessors → TestExt</span></span>
<span class="line"><span>   2313.1 ms  ✓ StatsBase</span></span>
<span class="line"><span>    685.6 ms  ✓ Accessors → StaticArraysExt</span></span>
<span class="line"><span>    801.2 ms  ✓ BangBang</span></span>
<span class="line"><span>    505.3 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    730.9 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    530.2 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>  17806.8 ms  ✓ MLStyle</span></span>
<span class="line"><span>   1129.6 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2738.9 ms  ✓ Transducers</span></span>
<span class="line"><span>    676.6 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   4424.7 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   5333.0 ms  ✓ FLoops</span></span>
<span class="line"><span>   5912.4 ms  ✓ MLUtils</span></span>
<span class="line"><span>  53 dependencies successfully precompiled in 36 seconds. 47 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceSparseArraysExt...</span></span>
<span class="line"><span>    648.4 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 8 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    670.2 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1555.9 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2160.9 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    370.1 ms  ✓ CEnum</span></span>
<span class="line"><span>    594.1 ms  ✓ AbstractFFTs</span></span>
<span class="line"><span>    368.2 ms  ✓ RealDot</span></span>
<span class="line"><span>    920.0 ms  ✓ FillArrays</span></span>
<span class="line"><span>    423.5 ms  ✓ Zlib_jll</span></span>
<span class="line"><span>    407.3 ms  ✓ HashArrayMappedTries</span></span>
<span class="line"><span>    626.1 ms  ✓ SuiteSparse</span></span>
<span class="line"><span>    766.5 ms  ✓ StructArrays</span></span>
<span class="line"><span>    455.0 ms  ✓ AbstractFFTs → AbstractFFTsChainRulesCoreExt</span></span>
<span class="line"><span>   1049.3 ms  ✓ ZygoteRules</span></span>
<span class="line"><span>    424.8 ms  ✓ FillArrays → FillArraysStatisticsExt</span></span>
<span class="line"><span>   1920.2 ms  ✓ IRTools</span></span>
<span class="line"><span>    712.4 ms  ✓ FillArrays → FillArraysSparseArraysExt</span></span>
<span class="line"><span>    650.8 ms  ✓ SparseInverseSubset</span></span>
<span class="line"><span>    411.6 ms  ✓ StructArrays → StructArraysAdaptExt</span></span>
<span class="line"><span>    666.9 ms  ✓ StructArrays → StructArraysSparseArraysExt</span></span>
<span class="line"><span>    666.4 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>    432.3 ms  ✓ StructArrays → StructArraysLinearAlgebraExt</span></span>
<span class="line"><span>    707.9 ms  ✓ StructArrays → StructArraysGPUArraysCoreExt</span></span>
<span class="line"><span>   6809.6 ms  ✓ LLVM</span></span>
<span class="line"><span>   5460.8 ms  ✓ ChainRules</span></span>
<span class="line"><span>   1765.9 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   4616.2 ms  ✓ GPUArrays</span></span>
<span class="line"><span>  33860.2 ms  ✓ Zygote</span></span>
<span class="line"><span>  24 dependencies successfully precompiled in 50 seconds. 79 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysExt...</span></span>
<span class="line"><span>    492.4 ms  ✓ Accessors → StructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 20 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    504.1 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 24 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    793.1 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    841.7 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 41 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    456.0 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling AbstractFFTsTestExt...</span></span>
<span class="line"><span>   1330.7 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 7 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1670.2 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>   1759.6 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 2 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   1738.2 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>   2774.0 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 3 seconds. 165 already precompiled.</span></span>
<span class="line"><span>Precompiling OneHotArrays...</span></span>
<span class="line"><span>    972.0 ms  ✓ OneHotArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 28 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesOneHotArraysExt...</span></span>
<span class="line"><span>    721.1 ms  ✓ MLDataDevices → MLDataDevicesOneHotArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 35 already precompiled.</span></span>
<span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>    498.3 ms  ✓ CodecZlib</span></span>
<span class="line"><span>    673.0 ms  ✓ ReactantCore</span></span>
<span class="line"><span>    734.0 ms  ✓ ConcurrentUtilities</span></span>
<span class="line"><span>    540.8 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>   2070.5 ms  ✓ ObjectFile</span></span>
<span class="line"><span>  18910.8 ms  ✓ HTTP</span></span>
<span class="line"><span>  27429.0 ms  ✓ GPUCompiler</span></span>
<span class="line"><span> 220787.0 ms  ✓ Enzyme</span></span>
<span class="line"><span>   5653.7 ms  ✓ Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>  71741.7 ms  ✓ Reactant</span></span>
<span class="line"><span>  10 dependencies successfully precompiled in 326 seconds. 67 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibEnzymeExt...</span></span>
<span class="line"><span>   6283.7 ms  ✓ Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   7873.1 ms  ✓ Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   1322.9 ms  ✓ LuxLib → LuxLibEnzymeExt</span></span>
<span class="line"><span>  11018.8 ms  ✓ Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   5865.4 ms  ✓ Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>  5 dependencies successfully precompiled in 13 seconds. 126 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   6768.8 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 146 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  12738.0 ms  ✓ LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  12847.7 ms  ✓ MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 79 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibReactantExt...</span></span>
<span class="line"><span>  12649.4 ms  ✓ Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>  13473.0 ms  ✓ LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>  13503.0 ms  ✓ Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>  13262.6 ms  ✓ Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>  12548.9 ms  ✓ Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>  12683.5 ms  ✓ Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>  6 dependencies successfully precompiled in 27 seconds. 154 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  12720.6 ms  ✓ WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantAbstractFFTsExt...</span></span>
<span class="line"><span>  12681.5 ms  ✓ Reactant → ReactantAbstractFFTsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 79 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  10973.5 ms  ✓ Lux → LuxReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 11 seconds. 177 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDatasets...</span></span>
<span class="line"><span>    622.1 ms  ✓ ZipFile</span></span>
<span class="line"><span>    669.5 ms  ✓ GZip</span></span>
<span class="line"><span>    494.7 ms  ✓ PooledArrays</span></span>
<span class="line"><span>    602.1 ms  ✓ Unitful → InverseFunctionsUnitfulExt</span></span>
<span class="line"><span>    622.1 ms  ✓ Accessors → UnitfulExt</span></span>
<span class="line"><span>   2240.4 ms  ✓ FixedPointNumbers</span></span>
<span class="line"><span>    770.6 ms  ✓ WeakRefStrings</span></span>
<span class="line"><span>   2288.8 ms  ✓ AtomsBase</span></span>
<span class="line"><span>    528.4 ms  ✓ FilePathsBase → FilePathsBaseMmapExt</span></span>
<span class="line"><span>   1298.8 ms  ✓ FilePathsBase → FilePathsBaseTestExt</span></span>
<span class="line"><span>   1781.1 ms  ✓ HDF5_jll</span></span>
<span class="line"><span>   9845.7 ms  ✓ JSON3</span></span>
<span class="line"><span>   3280.4 ms  ✓ DataDeps</span></span>
<span class="line"><span>   1798.2 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>   1516.9 ms  ✓ NPZ</span></span>
<span class="line"><span>  19848.4 ms  ✓ PrettyTables</span></span>
<span class="line"><span>   2273.8 ms  ✓ Pickle</span></span>
<span class="line"><span>   1402.6 ms  ✓ ColorTypes</span></span>
<span class="line"><span>   2481.9 ms  ✓ Chemfiles</span></span>
<span class="line"><span>   7504.1 ms  ✓ HDF5</span></span>
<span class="line"><span>  16533.9 ms  ✓ CSV</span></span>
<span class="line"><span>  32180.7 ms  ✓ JLD2</span></span>
<span class="line"><span>   2141.8 ms  ✓ ColorVectorSpace</span></span>
<span class="line"><span>   2519.9 ms  ✓ MAT</span></span>
<span class="line"><span>   4715.1 ms  ✓ Colors</span></span>
<span class="line"><span>   3529.3 ms  ✓ ColorSchemes</span></span>
<span class="line"><span>  18360.7 ms  ✓ ImageCore</span></span>
<span class="line"><span>   2035.2 ms  ✓ ImageBase</span></span>
<span class="line"><span>   1819.6 ms  ✓ ImageShow</span></span>
<span class="line"><span>  44311.0 ms  ✓ DataFrames</span></span>
<span class="line"><span>   1345.0 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>   1581.0 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>   9190.2 ms  ✓ MLDatasets</span></span>
<span class="line"><span>  33 dependencies successfully precompiled in 87 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling SimpleChains...</span></span>
<span class="line"><span>    508.0 ms  ✓ StaticArrayInterface → StaticArrayInterfaceOffsetArraysExt</span></span>
<span class="line"><span>    787.9 ms  ✓ HostCPUFeatures</span></span>
<span class="line"><span>   7518.6 ms  ✓ VectorizationBase</span></span>
<span class="line"><span>   1007.3 ms  ✓ SLEEFPirates</span></span>
<span class="line"><span>   1253.8 ms  ✓ VectorizedRNG</span></span>
<span class="line"><span>    728.9 ms  ✓ VectorizedRNG → VectorizedRNGStaticArraysExt</span></span>
<span class="line"><span>  26874.1 ms  ✓ LoopVectorization</span></span>
<span class="line"><span>   1118.2 ms  ✓ LoopVectorization → SpecialFunctionsExt</span></span>
<span class="line"><span>   1273.1 ms  ✓ LoopVectorization → ForwardDiffExt</span></span>
<span class="line"><span>   6119.3 ms  ✓ SimpleChains</span></span>
<span class="line"><span>  10 dependencies successfully precompiled in 44 seconds. 64 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibSLEEFPiratesExt...</span></span>
<span class="line"><span>   2396.3 ms  ✓ LuxLib → LuxLibSLEEFPiratesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 95 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantOffsetArraysExt...</span></span>
<span class="line"><span>  12546.0 ms  ✓ Reactant → ReactantOffsetArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 79 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibLoopVectorizationExt...</span></span>
<span class="line"><span>   3951.9 ms  ✓ LuxLib → LuxLibLoopVectorizationExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 103 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxSimpleChainsExt...</span></span>
<span class="line"><span>   1940.6 ms  ✓ Lux → LuxSimpleChainsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 124 already precompiled.</span></span>
<span class="line"><span>2025-03-17 02:17:43.738141: I external/xla/xla/service/service.cc:152] XLA service 0x7b4f960 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-17 02:17:43.738484: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1742177863.739271  195554 se_gpu_pjrt_client.cc:951] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1742177863.739349  195554 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1742177863.739400  195554 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1742177863.754362  195554 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span></code></pre></div><h2 id="Loading-MNIST" tabindex="-1">Loading MNIST <a class="header-anchor" href="#Loading-MNIST" aria-label="Permalink to &quot;Loading MNIST {#Loading-MNIST}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadmnist</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, train_split)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Load MNIST</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> parse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Bool, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ENV</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;CI&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;false&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1500</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> :</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dataset </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> MNIST</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; split</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">!==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        imgs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataset</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">features[:, :, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        labels_raw </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataset</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">targets[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        imgs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataset</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">features</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        labels_raw </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataset</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">targets</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Process images into (H, W, C, BS) batches</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(labels_raw, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_test, y_test) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, y_data); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">train_split)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the test data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_test, y_test)); batchsize, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>loadmnist (generic function with 1 method)</span></span></code></pre></div><h2 id="Define-the-Model" tabindex="-1">Define the Model <a class="header-anchor" href="#Define-the-Model" aria-label="Permalink to &quot;Define the Model {#Define-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lux_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    MaxPool</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">6</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    MaxPool</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    FlattenLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 84</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">84</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Chain(</span></span>
<span class="line"><span>    layer_1 = Conv((5, 5), 1 =&gt; 6, relu),  # 156 parameters</span></span>
<span class="line"><span>    layer_2 = MaxPool((2, 2)),</span></span>
<span class="line"><span>    layer_3 = Conv((5, 5), 6 =&gt; 16, relu),  # 2_416 parameters</span></span>
<span class="line"><span>    layer_4 = MaxPool((2, 2)),</span></span>
<span class="line"><span>    layer_5 = Lux.FlattenLayer{Static.StaticInt{3}}(static(3)),</span></span>
<span class="line"><span>    layer_6 = Chain(</span></span>
<span class="line"><span>        layer_1 = Dense(256 =&gt; 128, relu),  # 32_896 parameters</span></span>
<span class="line"><span>        layer_2 = Dense(128 =&gt; 84, relu),  # 10_836 parameters</span></span>
<span class="line"><span>        layer_3 = Dense(84 =&gt; 10),      # 850 parameters</span></span>
<span class="line"><span>    ),</span></span>
<span class="line"><span>)         # Total: 47_154 parameters,</span></span>
<span class="line"><span>          #        plus 0 states.</span></span></code></pre></div><p>We now need to convert the lux_model to SimpleChains.jl. We need to do this by defining the <a href="/dev/api/Lux/interop#Lux.ToSimpleChainsAdaptor"><code>ToSimpleChainsAdaptor</code></a> and providing the input dimensions.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">adaptor </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ToSimpleChainsAdaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">simple_chains_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> adaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lux_model)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SimpleChainsLayer(</span></span>
<span class="line"><span>    Chain(</span></span>
<span class="line"><span>        layer_1 = Conv((5, 5), 1 =&gt; 6, relu),  # 156 parameters</span></span>
<span class="line"><span>        layer_2 = MaxPool((2, 2)),</span></span>
<span class="line"><span>        layer_3 = Conv((5, 5), 6 =&gt; 16, relu),  # 2_416 parameters</span></span>
<span class="line"><span>        layer_4 = MaxPool((2, 2)),</span></span>
<span class="line"><span>        layer_5 = Lux.FlattenLayer{Static.StaticInt{3}}(static(3)),</span></span>
<span class="line"><span>        layer_6 = Chain(</span></span>
<span class="line"><span>            layer_1 = Dense(256 =&gt; 128, relu),  # 32_896 parameters</span></span>
<span class="line"><span>            layer_2 = Dense(128 =&gt; 84, relu),  # 10_836 parameters</span></span>
<span class="line"><span>            layer_3 = Dense(84 =&gt; 10),  # 850 parameters</span></span>
<span class="line"><span>        ),</span></span>
<span class="line"><span>    ),</span></span>
<span class="line"><span>)         # Total: 47_154 parameters,</span></span>
<span class="line"><span>          #        plus 0 states.</span></span></code></pre></div><h2 id="Helper-Functions" tabindex="-1">Helper Functions <a class="header-anchor" href="#Helper-Functions" aria-label="Permalink to &quot;Helper Functions {#Helper-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; logits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, dataloader)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    total_correct, total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        predicted_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st))))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> predicted_class)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(target_class)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Define-the-Training-Loop" tabindex="-1">Define the Training Loop <a class="header-anchor" href="#Define-the-Training-Loop" aria-label="Permalink to &quot;Define the Training Loop {#Define-the-Training-Loop}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, dev</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(); rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_dataloader, test_dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loadmnist</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    vjp </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3.0f-4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_ra </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(test_dataloader)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_ra, ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    ### Lets train the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    nepochs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 10</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    tr_acc, te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nepochs</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        stime </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> time</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            _, _, _, train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                vjp, lossfn, (x, y), train_state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ttime </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> time</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> stime</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        tr_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                model_compiled, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, train_dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                model_compiled, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, test_dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;[%2d/%2d] </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Time %.2fs </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Training Accuracy: %.2f%% </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Test Accuracy: \\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 %.2f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch nepochs ttime tr_acc te_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> tr_acc, te_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>train (generic function with 2 methods)</span></span></code></pre></div><h2 id="Finally-Training-the-Model" tabindex="-1">Finally Training the Model <a class="header-anchor" href="#Finally-Training-the-Model" aria-label="Permalink to &quot;Finally Training the Model {#Finally-Training-the-Model}&quot;">​</a></h2><p>First we will train the Lux model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tr_acc, te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lux_model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ 1/10] 	 Time 309.13s 	 Training Accuracy: 13.91% 	 Test Accuracy: 16.41%</span></span>
<span class="line"><span>[ 2/10] 	 Time 0.40s 	 Training Accuracy: 17.81% 	 Test Accuracy: 14.84%</span></span>
<span class="line"><span>[ 3/10] 	 Time 0.38s 	 Training Accuracy: 19.22% 	 Test Accuracy: 16.41%</span></span>
<span class="line"><span>[ 4/10] 	 Time 0.37s 	 Training Accuracy: 31.09% 	 Test Accuracy: 25.00%</span></span>
<span class="line"><span>[ 5/10] 	 Time 0.38s 	 Training Accuracy: 37.66% 	 Test Accuracy: 34.38%</span></span>
<span class="line"><span>[ 6/10] 	 Time 0.38s 	 Training Accuracy: 45.16% 	 Test Accuracy: 35.16%</span></span>
<span class="line"><span>[ 7/10] 	 Time 0.36s 	 Training Accuracy: 53.67% 	 Test Accuracy: 43.75%</span></span>
<span class="line"><span>[ 8/10] 	 Time 0.37s 	 Training Accuracy: 59.53% 	 Test Accuracy: 50.00%</span></span>
<span class="line"><span>[ 9/10] 	 Time 0.38s 	 Training Accuracy: 65.31% 	 Test Accuracy: 55.47%</span></span>
<span class="line"><span>[10/10] 	 Time 0.41s 	 Training Accuracy: 70.08% 	 Test Accuracy: 62.50%</span></span></code></pre></div><p>Now we will train the SimpleChains model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tr_acc, te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(simple_chains_model)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ 1/10] 	 Time 876.25s 	 Training Accuracy: 32.27% 	 Test Accuracy: 36.72%</span></span>
<span class="line"><span>[ 2/10] 	 Time 12.94s 	 Training Accuracy: 52.19% 	 Test Accuracy: 52.34%</span></span>
<span class="line"><span>[ 3/10] 	 Time 12.94s 	 Training Accuracy: 65.70% 	 Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 4/10] 	 Time 12.94s 	 Training Accuracy: 69.77% 	 Test Accuracy: 67.19%</span></span>
<span class="line"><span>[ 5/10] 	 Time 12.94s 	 Training Accuracy: 77.11% 	 Test Accuracy: 75.78%</span></span>
<span class="line"><span>[ 6/10] 	 Time 12.96s 	 Training Accuracy: 78.75% 	 Test Accuracy: 77.34%</span></span>
<span class="line"><span>[ 7/10] 	 Time 12.95s 	 Training Accuracy: 81.25% 	 Test Accuracy: 78.91%</span></span>
<span class="line"><span>[ 8/10] 	 Time 12.95s 	 Training Accuracy: 83.67% 	 Test Accuracy: 82.81%</span></span>
<span class="line"><span>[ 9/10] 	 Time 12.95s 	 Training Accuracy: 86.02% 	 Test Accuracy: 86.72%</span></span>
<span class="line"><span>[10/10] 	 Time 12.95s 	 Training Accuracy: 87.89% 	 Test Accuracy: 87.50%</span></span></code></pre></div><p>On my local machine we see a 3-4x speedup when using SimpleChains.jl. The conditions of the server this documentation is being built on is not ideal for CPU benchmarking hence, the speedup may not be as significant and even there might be regressions.</p><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  JULIA_DEBUG = Literate</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,33)]))}const y=a(l,[["render",e]]);export{E as __pageData,y as default};
