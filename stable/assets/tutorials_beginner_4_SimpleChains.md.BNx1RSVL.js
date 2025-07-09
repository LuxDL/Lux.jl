import{_ as a,c as n,o as i,al as p}from"./chunks/framework.Bjd2Ow0y.js";const E=JSON.parse('{"title":"MNIST Classification with SimpleChains","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/4_SimpleChains.md","filePath":"tutorials/beginner/4_SimpleChains.md","lastUpdated":null}'),l={name:"tutorials/beginner/4_SimpleChains.md"};function e(t,s,h,k,c,r){return i(),n("div",null,s[0]||(s[0]=[p(`<h1 id="MNIST-Classification-with-SimpleChains" tabindex="-1">MNIST Classification with SimpleChains <a class="header-anchor" href="#MNIST-Classification-with-SimpleChains" aria-label="Permalink to &quot;MNIST Classification with SimpleChains {#MNIST-Classification-with-SimpleChains}&quot;">​</a></h1><p>SimpleChains.jl is an excellent framework for training small neural networks. In this tutorial we will demonstrate how to use the same API as Lux.jl to train a model using SimpleChains.jl. We will use the tutorial from <a href="https://pumasai.github.io/SimpleChains.jl/dev/examples/mnist/" target="_blank" rel="noreferrer">SimpleChains.jl</a> as a reference.</p><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Statistics, Printf, Reactant</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDatasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MNIST</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SimpleChains</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SimpleChains</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_default_backend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;cpu&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    497.8 ms  ✓ Requires</span></span>
<span class="line"><span>    514.0 ms  ✓ Compat</span></span>
<span class="line"><span>    582.8 ms  ✓ CpuId</span></span>
<span class="line"><span>    441.5 ms  ✓ JLLWrappers</span></span>
<span class="line"><span>    812.7 ms  ✓ Static</span></span>
<span class="line"><span>   1571.1 ms  ✓ DispatchDoctor</span></span>
<span class="line"><span>    392.0 ms  ✓ Compat → CompatLinearAlgebraExt</span></span>
<span class="line"><span>    406.5 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>    586.7 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>    409.5 ms  ✓ DispatchDoctor → DispatchDoctorEnzymeCoreExt</span></span>
<span class="line"><span>   1017.3 ms  ✓ CPUSummary</span></span>
<span class="line"><span>    614.7 ms  ✓ Functors</span></span>
<span class="line"><span>   1187.7 ms  ✓ ChainRulesCore</span></span>
<span class="line"><span>   1246.9 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>   1164.1 ms  ✓ LuxCore</span></span>
<span class="line"><span>    640.6 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>    819.3 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>    413.0 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>   7138.3 ms  ✓ StaticArrays</span></span>
<span class="line"><span>   2484.8 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    390.4 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>   1346.4 ms  ✓ Optimisers</span></span>
<span class="line"><span>    597.0 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>   1267.0 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    475.5 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>    561.1 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    576.8 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>    467.5 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    421.6 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    463.0 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    642.8 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>    443.6 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    649.3 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    600.0 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    607.9 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    613.9 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    641.6 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>   1609.5 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    434.5 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    479.5 ms  ✓ Optimisers → OptimisersAdaptExt</span></span>
<span class="line"><span>   2732.4 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    911.7 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>   3543.3 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    860.4 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    785.0 ms  ✓ Polyester</span></span>
<span class="line"><span>    822.1 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   4306.0 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    683.8 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    763.9 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5596.1 ms  ✓ NNlib</span></span>
<span class="line"><span>    881.3 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    888.8 ms  ✓ NNlib → NNlibSpecialFunctionsExt</span></span>
<span class="line"><span>    911.9 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5370.3 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9583.5 ms  ✓ Lux</span></span>
<span class="line"><span>  55 dependencies successfully precompiled in 41 seconds. 50 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    395.5 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>    405.2 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>   1142.1 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>    640.8 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>   1618.2 ms  ✓ DataStructures</span></span>
<span class="line"><span>    943.4 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>    559.4 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>   3688.6 ms  ✓ Test</span></span>
<span class="line"><span>    485.6 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>    932.2 ms  ✓ MLCore</span></span>
<span class="line"><span>    568.6 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>   2470.9 ms  ✓ Accessors</span></span>
<span class="line"><span>   1062.1 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    804.2 ms  ✓ Accessors → LinearAlgebraExt</span></span>
<span class="line"><span>    614.8 ms  ✓ Accessors → TestExt</span></span>
<span class="line"><span>    689.2 ms  ✓ Accessors → StaticArraysExt</span></span>
<span class="line"><span>    731.1 ms  ✓ BangBang</span></span>
<span class="line"><span>   2123.7 ms  ✓ StatsBase</span></span>
<span class="line"><span>    479.1 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    681.2 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    479.2 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    891.4 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2653.7 ms  ✓ Transducers</span></span>
<span class="line"><span>    641.1 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   4939.2 ms  ✓ FLoops</span></span>
<span class="line"><span>   5779.7 ms  ✓ MLUtils</span></span>
<span class="line"><span>  26 dependencies successfully precompiled in 22 seconds. 71 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    646.2 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1490.4 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2017.7 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 164 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    426.1 ms  ✓ AbstractFFTs → AbstractFFTsChainRulesCoreExt</span></span>
<span class="line"><span>    894.1 ms  ✓ ZygoteRules</span></span>
<span class="line"><span>   5315.7 ms  ✓ ChainRules</span></span>
<span class="line"><span>  33411.2 ms  ✓ Zygote</span></span>
<span class="line"><span>  4 dependencies successfully precompiled in 39 seconds. 61 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysExt...</span></span>
<span class="line"><span>    447.3 ms  ✓ Accessors → StructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 20 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    472.1 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 24 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysStaticArraysExt...</span></span>
<span class="line"><span>    650.3 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 19 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysGPUArraysCoreExt...</span></span>
<span class="line"><span>    691.8 ms  ✓ StructArrays → StructArraysGPUArraysCoreExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 34 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    749.5 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    796.4 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 41 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    419.8 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling AbstractFFTsTestExt...</span></span>
<span class="line"><span>   1285.1 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 7 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1493.3 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 71 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   2516.3 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 143 already precompiled.</span></span>
<span class="line"><span>Precompiling OneHotArrays...</span></span>
<span class="line"><span>    943.5 ms  ✓ OneHotArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 31 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesOneHotArraysExt...</span></span>
<span class="line"><span>    725.4 ms  ✓ MLDataDevices → MLDataDevicesOneHotArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 38 already precompiled.</span></span>
<span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>    425.1 ms  ✓ Scratch</span></span>
<span class="line"><span>    502.1 ms  ✓ LoggingExtras</span></span>
<span class="line"><span>    488.3 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>   1018.6 ms  ✓ MbedTLS</span></span>
<span class="line"><span>    602.0 ms  ✓ OpenSSL_jll</span></span>
<span class="line"><span>    645.9 ms  ✓ LLVMOpenMP_jll</span></span>
<span class="line"><span>    559.5 ms  ✓ LibTracyClient_jll</span></span>
<span class="line"><span>   1515.8 ms  ✓ CUDA_Driver_jll</span></span>
<span class="line"><span>    992.7 ms  ✓ LazyArtifacts</span></span>
<span class="line"><span>    868.8 ms  ✓ Tracy</span></span>
<span class="line"><span>   1824.6 ms  ✓ OpenSSL</span></span>
<span class="line"><span>   1548.9 ms  ✓ Enzyme_jll</span></span>
<span class="line"><span>   1416.7 ms  ✓ LLVMExtra_jll</span></span>
<span class="line"><span>   3053.7 ms  ✓ Reactant_jll</span></span>
<span class="line"><span>   6900.5 ms  ✓ LLVM</span></span>
<span class="line"><span>  19274.2 ms  ✓ HTTP</span></span>
<span class="line"><span>  29545.7 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>  93127.3 ms  ✓ Enzyme</span></span>
<span class="line"><span>   7055.1 ms  ✓ Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>  90463.4 ms  ✓ Reactant</span></span>
<span class="line"><span>  20 dependencies successfully precompiled in 232 seconds. 60 already precompiled.</span></span>
<span class="line"><span>Precompiling UnsafeAtomicsLLVM...</span></span>
<span class="line"><span>   1768.1 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 30 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibEnzymeExt...</span></span>
<span class="line"><span>   6913.6 ms  ✓ Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   6535.5 ms  ✓ Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>  14586.0 ms  ✓ Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   1244.6 ms  ✓ LuxLib → LuxLibEnzymeExt</span></span>
<span class="line"><span>  14928.2 ms  ✓ Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>  5 dependencies successfully precompiled in 15 seconds. 129 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   7666.6 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 8 seconds. 149 already precompiled.</span></span>
<span class="line"><span>Precompiling OptimisersReactantExt...</span></span>
<span class="line"><span>  17453.4 ms  ✓ Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>  19713.6 ms  ✓ Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 20 seconds. 88 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  17215.6 ms  ✓ LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 17 seconds. 85 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  17435.2 ms  ✓ MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 18 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibReactantExt...</span></span>
<span class="line"><span>  17282.5 ms  ✓ Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>  18149.0 ms  ✓ LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>  18159.9 ms  ✓ Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>  17211.4 ms  ✓ Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>  4 dependencies successfully precompiled in 35 seconds. 158 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  17359.7 ms  ✓ WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 18 seconds. 96 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantAbstractFFTsExt...</span></span>
<span class="line"><span>  17234.1 ms  ✓ Reactant → ReactantAbstractFFTsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 18 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantOneHotArraysExt...</span></span>
<span class="line"><span>  17530.8 ms  ✓ Reactant → ReactantOneHotArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 18 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  19930.8 ms  ✓ Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 20 seconds. 103 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  12512.3 ms  ✓ Lux → LuxReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 180 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDatasets...</span></span>
<span class="line"><span>    467.7 ms  ✓ StridedViews</span></span>
<span class="line"><span>    555.7 ms  ✓ BFloat16s</span></span>
<span class="line"><span>    648.3 ms  ✓ ZipFile</span></span>
<span class="line"><span>    780.4 ms  ✓ StructTypes</span></span>
<span class="line"><span>   1292.3 ms  ✓ SentinelArrays</span></span>
<span class="line"><span>    572.7 ms  ✓ MPIPreferences</span></span>
<span class="line"><span>    565.0 ms  ✓ Chemfiles_jll</span></span>
<span class="line"><span>    611.8 ms  ✓ libaec_jll</span></span>
<span class="line"><span>    591.7 ms  ✓ Hwloc_jll</span></span>
<span class="line"><span>    464.1 ms  ✓ MicrosoftMPI_jll</span></span>
<span class="line"><span>    693.8 ms  ✓ Libiconv_jll</span></span>
<span class="line"><span>   2093.9 ms  ✓ StringManipulation</span></span>
<span class="line"><span>   3977.9 ms  ✓ ColorSchemes</span></span>
<span class="line"><span>   1048.8 ms  ✓ FilePathsBase</span></span>
<span class="line"><span>   4383.1 ms  ✓ FileIO</span></span>
<span class="line"><span>    574.9 ms  ✓ InlineStrings → ParsersExt</span></span>
<span class="line"><span>   3137.7 ms  ✓ DataDeps</span></span>
<span class="line"><span>    375.8 ms  ✓ StridedViews → StridedViewsPtrArraysExt</span></span>
<span class="line"><span>  20313.9 ms  ✓ Unitful</span></span>
<span class="line"><span>   1218.0 ms  ✓ MPItrampoline_jll</span></span>
<span class="line"><span>  18778.9 ms  ✓ ImageCore</span></span>
<span class="line"><span>   1178.9 ms  ✓ OpenMPI_jll</span></span>
<span class="line"><span>   1398.1 ms  ✓ MPICH_jll</span></span>
<span class="line"><span>    597.7 ms  ✓ StringEncodings</span></span>
<span class="line"><span>    523.4 ms  ✓ FilePathsBase → FilePathsBaseMmapExt</span></span>
<span class="line"><span>   1244.2 ms  ✓ FilePathsBase → FilePathsBaseTestExt</span></span>
<span class="line"><span>   1830.6 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>   8994.7 ms  ✓ JSON3</span></span>
<span class="line"><span>   1468.3 ms  ✓ NPZ</span></span>
<span class="line"><span>    753.5 ms  ✓ WeakRefStrings</span></span>
<span class="line"><span>    595.8 ms  ✓ Unitful → ConstructionBaseUnitfulExt</span></span>
<span class="line"><span>    592.4 ms  ✓ Unitful → InverseFunctionsUnitfulExt</span></span>
<span class="line"><span>    692.2 ms  ✓ Unitful → PrintfExt</span></span>
<span class="line"><span>    610.4 ms  ✓ Accessors → UnitfulExt</span></span>
<span class="line"><span>   2079.6 ms  ✓ ImageBase</span></span>
<span class="line"><span>   1981.1 ms  ✓ HDF5_jll</span></span>
<span class="line"><span>   2307.5 ms  ✓ Pickle</span></span>
<span class="line"><span>  19980.6 ms  ✓ PrettyTables</span></span>
<span class="line"><span>   2917.4 ms  ✓ UnitfulAtomic</span></span>
<span class="line"><span>   2535.2 ms  ✓ PeriodicTable</span></span>
<span class="line"><span>   2048.7 ms  ✓ ImageShow</span></span>
<span class="line"><span>  18544.2 ms  ✓ CSV</span></span>
<span class="line"><span>   7699.0 ms  ✓ HDF5</span></span>
<span class="line"><span>   2355.5 ms  ✓ AtomsBase</span></span>
<span class="line"><span>  34052.0 ms  ✓ JLD2</span></span>
<span class="line"><span>   2420.6 ms  ✓ MAT</span></span>
<span class="line"><span>   2429.5 ms  ✓ Chemfiles</span></span>
<span class="line"><span>  46720.0 ms  ✓ DataFrames</span></span>
<span class="line"><span>   1366.9 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>   1625.7 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>   9385.3 ms  ✓ MLDatasets</span></span>
<span class="line"><span>  51 dependencies successfully precompiled in 115 seconds. 153 already precompiled.</span></span>
<span class="line"><span>Precompiling SimpleChains...</span></span>
<span class="line"><span>    471.6 ms  ✓ StaticArrayInterface → StaticArrayInterfaceOffsetArraysExt</span></span>
<span class="line"><span>    768.7 ms  ✓ HostCPUFeatures</span></span>
<span class="line"><span>   7592.2 ms  ✓ VectorizationBase</span></span>
<span class="line"><span>    992.0 ms  ✓ SLEEFPirates</span></span>
<span class="line"><span>   1206.7 ms  ✓ VectorizedRNG</span></span>
<span class="line"><span>    699.2 ms  ✓ VectorizedRNG → VectorizedRNGStaticArraysExt</span></span>
<span class="line"><span>  27403.3 ms  ✓ LoopVectorization</span></span>
<span class="line"><span>    982.7 ms  ✓ LoopVectorization → SpecialFunctionsExt</span></span>
<span class="line"><span>   1182.1 ms  ✓ LoopVectorization → ForwardDiffExt</span></span>
<span class="line"><span>   5986.5 ms  ✓ SimpleChains</span></span>
<span class="line"><span>  10 dependencies successfully precompiled in 44 seconds. 59 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibSLEEFPiratesExt...</span></span>
<span class="line"><span>   2352.4 ms  ✓ LuxLib → LuxLibSLEEFPiratesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantOffsetArraysExt...</span></span>
<span class="line"><span>  17140.8 ms  ✓ Reactant → ReactantOffsetArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 17 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibLoopVectorizationExt...</span></span>
<span class="line"><span>   4504.3 ms  ✓ LuxLib → LuxLibLoopVectorizationExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxSimpleChainsExt...</span></span>
<span class="line"><span>   1852.2 ms  ✓ Lux → LuxSimpleChainsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 122 already precompiled.</span></span>
<span class="line"><span>2025-07-09 04:38:18.707494: I external/xla/xla/service/service.cc:153] XLA service 0x27ed2e80 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-07-09 04:38:18.707584: I external/xla/xla/service/service.cc:161]   StreamExecutor device (0): Quadro RTX 5000, Compute Capability 7.5</span></span>
<span class="line"><span>2025-07-09 04:38:18.707587: I external/xla/xla/service/service.cc:161]   StreamExecutor device (1): Quadro RTX 5000, Compute Capability 7.5</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1752035898.712398 1228555 se_gpu_pjrt_client.cc:1370] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1752035898.712562 1228555 gpu_helpers.cc:136] XLA backend allocating 12528893952 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1752035898.712656 1228555 gpu_helpers.cc:136] XLA backend allocating 12528893952 bytes on device 1 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1752035898.712680 1228555 gpu_helpers.cc:177] XLA backend will use up to 4176297984 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1752035898.712698 1228555 gpu_helpers.cc:177] XLA backend will use up to 4176297984 bytes on device 1 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1752035898.728345 1228555 cuda_dnn.cc:471] Loaded cuDNN version 90800</span></span></code></pre></div><h2 id="Loading-MNIST" tabindex="-1">Loading MNIST <a class="header-anchor" href="#Loading-MNIST" aria-label="Permalink to &quot;Loading MNIST {#Loading-MNIST}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadmnist</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, train_split)</span></span>
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
<span class="line"><span>          #        plus 0 states.</span></span></code></pre></div><p>We now need to convert the lux_model to SimpleChains.jl. We need to do this by defining the <a href="/stable/api/Lux/interop#Lux.ToSimpleChainsAdaptor"><code>ToSimpleChainsAdaptor</code></a> and providing the input dimensions.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">adaptor </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ToSimpleChainsAdaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_ra, ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>train (generic function with 2 methods)</span></span></code></pre></div><h2 id="Finally-Training-the-Model" tabindex="-1">Finally Training the Model <a class="header-anchor" href="#Finally-Training-the-Model" aria-label="Permalink to &quot;Finally Training the Model {#Finally-Training-the-Model}&quot;">​</a></h2><p>First we will train the Lux model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tr_acc, te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lux_model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ 1/10] 	 Time 397.11s 	 Training Accuracy: 18.36% 	 Test Accuracy: 21.88%</span></span>
<span class="line"><span>[ 2/10] 	 Time 0.11s 	 Training Accuracy: 31.95% 	 Test Accuracy: 33.59%</span></span>
<span class="line"><span>[ 3/10] 	 Time 0.10s 	 Training Accuracy: 43.59% 	 Test Accuracy: 40.62%</span></span>
<span class="line"><span>[ 4/10] 	 Time 0.10s 	 Training Accuracy: 55.86% 	 Test Accuracy: 47.66%</span></span>
<span class="line"><span>[ 5/10] 	 Time 0.10s 	 Training Accuracy: 64.77% 	 Test Accuracy: 58.59%</span></span>
<span class="line"><span>[ 6/10] 	 Time 0.09s 	 Training Accuracy: 72.97% 	 Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 7/10] 	 Time 0.10s 	 Training Accuracy: 77.03% 	 Test Accuracy: 66.41%</span></span>
<span class="line"><span>[ 8/10] 	 Time 0.13s 	 Training Accuracy: 80.00% 	 Test Accuracy: 74.22%</span></span>
<span class="line"><span>[ 9/10] 	 Time 0.10s 	 Training Accuracy: 81.02% 	 Test Accuracy: 75.00%</span></span>
<span class="line"><span>[10/10] 	 Time 0.09s 	 Training Accuracy: 82.97% 	 Test Accuracy: 75.00%</span></span></code></pre></div><p>Now we will train the SimpleChains model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tr_acc, te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(simple_chains_model)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ 1/10] 	 Time 905.02s 	 Training Accuracy: 22.11% 	 Test Accuracy: 16.41%</span></span>
<span class="line"><span>[ 2/10] 	 Time 13.47s 	 Training Accuracy: 41.17% 	 Test Accuracy: 39.06%</span></span>
<span class="line"><span>[ 3/10] 	 Time 13.47s 	 Training Accuracy: 54.53% 	 Test Accuracy: 51.56%</span></span>
<span class="line"><span>[ 4/10] 	 Time 13.45s 	 Training Accuracy: 64.84% 	 Test Accuracy: 58.59%</span></span>
<span class="line"><span>[ 5/10] 	 Time 13.44s 	 Training Accuracy: 72.97% 	 Test Accuracy: 70.31%</span></span>
<span class="line"><span>[ 6/10] 	 Time 13.44s 	 Training Accuracy: 76.72% 	 Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 7/10] 	 Time 13.42s 	 Training Accuracy: 78.98% 	 Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 8/10] 	 Time 13.43s 	 Training Accuracy: 79.38% 	 Test Accuracy: 76.56%</span></span>
<span class="line"><span>[ 9/10] 	 Time 13.55s 	 Training Accuracy: 82.42% 	 Test Accuracy: 78.91%</span></span>
<span class="line"><span>[10/10] 	 Time 13.45s 	 Training Accuracy: 83.83% 	 Test Accuracy: 83.59%</span></span></code></pre></div><p>On my local machine we see a 3-4x speedup when using SimpleChains.jl. The conditions of the server this documentation is being built on is not ideal for CPU benchmarking hence, the speedup may not be as significant and even there might be regressions.</p><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,33)]))}const y=a(l,[["render",e]]);export{E as __pageData,y as default};
