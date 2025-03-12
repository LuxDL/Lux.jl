import{_ as a,c as n,o as i,al as p}from"./chunks/framework.BCN3FD2k.js";const E=JSON.parse('{"title":"MNIST Classification with SimpleChains","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/4_SimpleChains.md","filePath":"tutorials/beginner/4_SimpleChains.md","lastUpdated":null}'),l={name:"tutorials/beginner/4_SimpleChains.md"};function e(t,s,h,r,c,k){return i(),n("div",null,s[0]||(s[0]=[p(`<h1 id="MNIST-Classification-with-SimpleChains" tabindex="-1">MNIST Classification with SimpleChains <a class="header-anchor" href="#MNIST-Classification-with-SimpleChains" aria-label="Permalink to &quot;MNIST Classification with SimpleChains {#MNIST-Classification-with-SimpleChains}&quot;">​</a></h1><p>SimpleChains.jl is an excellent framework for training small neural networks. In this tutorial we will demonstrate how to use the same API as Lux.jl to train a model using SimpleChains.jl. We will use the tutorial from <a href="https://pumasai.github.io/SimpleChains.jl/dev/examples/mnist/" target="_blank" rel="noreferrer">SimpleChains.jl</a> as a reference.</p><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Statistics, Printf, Reactant</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDatasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MNIST</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SimpleChains</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SimpleChains</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_default_backend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;cpu&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    369.1 ms  ✓ Future</span></span>
<span class="line"><span>    423.7 ms  ✓ ConcreteStructs</span></span>
<span class="line"><span>    388.5 ms  ✓ CEnum</span></span>
<span class="line"><span>    592.0 ms  ✓ ADTypes</span></span>
<span class="line"><span>    415.5 ms  ✓ OpenLibm_jll</span></span>
<span class="line"><span>    569.8 ms  ✓ Statistics</span></span>
<span class="line"><span>    395.7 ms  ✓ ArgCheck</span></span>
<span class="line"><span>   1765.8 ms  ✓ UnsafeAtomics</span></span>
<span class="line"><span>    387.9 ms  ✓ ManualMemory</span></span>
<span class="line"><span>    334.3 ms  ✓ Reexport</span></span>
<span class="line"><span>    334.6 ms  ✓ SIMDTypes</span></span>
<span class="line"><span>    564.3 ms  ✓ EnzymeCore</span></span>
<span class="line"><span>    329.5 ms  ✓ IfElse</span></span>
<span class="line"><span>   1165.9 ms  ✓ IrrationalConstants</span></span>
<span class="line"><span>    341.9 ms  ✓ CommonWorldInvalidations</span></span>
<span class="line"><span>    354.8 ms  ✓ FastClosures</span></span>
<span class="line"><span>    392.6 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>   2516.1 ms  ✓ MacroTools</span></span>
<span class="line"><span>    512.2 ms  ✓ ArrayInterface</span></span>
<span class="line"><span>    458.9 ms  ✓ GPUArraysCore</span></span>
<span class="line"><span>    665.5 ms  ✓ CpuId</span></span>
<span class="line"><span>    660.7 ms  ✓ DocStringExtensions</span></span>
<span class="line"><span>    479.9 ms  ✓ JLLWrappers</span></span>
<span class="line"><span>    398.6 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>    462.0 ms  ✓ NaNMath</span></span>
<span class="line"><span>    507.6 ms  ✓ Atomix</span></span>
<span class="line"><span>   1252.8 ms  ✓ ChainRulesCore</span></span>
<span class="line"><span>    379.9 ms  ✓ EnzymeCore → AdaptExt</span></span>
<span class="line"><span>    853.4 ms  ✓ ThreadingUtilities</span></span>
<span class="line"><span>    395.0 ms  ✓ ADTypes → ADTypesEnzymeCoreExt</span></span>
<span class="line"><span>    384.7 ms  ✓ DiffResults</span></span>
<span class="line"><span>    806.0 ms  ✓ Static</span></span>
<span class="line"><span>    681.9 ms  ✓ CommonSubexpressions</span></span>
<span class="line"><span>   1523.0 ms  ✓ DispatchDoctor</span></span>
<span class="line"><span>   1454.7 ms  ✓ Setfield</span></span>
<span class="line"><span>    396.6 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>    394.2 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>    623.7 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    600.5 ms  ✓ Hwloc_jll</span></span>
<span class="line"><span>    642.8 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>    686.2 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>   1287.2 ms  ✓ Optimisers</span></span>
<span class="line"><span>    415.5 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    425.5 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>    457.2 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>   1036.9 ms  ✓ CPUSummary</span></span>
<span class="line"><span>   1589.9 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>    441.0 ms  ✓ DispatchDoctor → DispatchDoctorEnzymeCoreExt</span></span>
<span class="line"><span>    660.6 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>   7469.5 ms  ✓ StaticArrays</span></span>
<span class="line"><span>   1225.5 ms  ✓ LuxCore</span></span>
<span class="line"><span>   1364.2 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    438.4 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    451.4 ms  ✓ Optimisers → OptimisersAdaptExt</span></span>
<span class="line"><span>   2197.9 ms  ✓ Hwloc</span></span>
<span class="line"><span>    488.3 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>    625.4 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>   2561.1 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    618.0 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    654.9 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    626.0 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    639.0 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    622.4 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    695.1 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    476.4 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    667.6 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>    459.0 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    483.1 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    471.2 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    621.5 ms  ✓ DiffRules</span></span>
<span class="line"><span>   1746.5 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    939.5 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>   2756.9 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    781.4 ms  ✓ Polyester</span></span>
<span class="line"><span>    952.0 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>   3710.0 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>   4180.1 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    912.9 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>    680.5 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    785.8 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5657.1 ms  ✓ NNlib</span></span>
<span class="line"><span>    868.8 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    928.7 ms  ✓ NNlib → NNlibSpecialFunctionsExt</span></span>
<span class="line"><span>   1006.5 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   6468.7 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9957.4 ms  ✓ Lux</span></span>
<span class="line"><span>  86 dependencies successfully precompiled in 49 seconds. 24 already precompiled.</span></span>
<span class="line"><span>Precompiling MLUtils...</span></span>
<span class="line"><span>    326.5 ms  ✓ IteratorInterfaceExtensions</span></span>
<span class="line"><span>    405.4 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    440.1 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    894.1 ms  ✓ InitialValues</span></span>
<span class="line"><span>    613.6 ms  ✓ Serialization</span></span>
<span class="line"><span>    393.4 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    445.4 ms  ✓ ShowCases</span></span>
<span class="line"><span>    455.5 ms  ✓ SuiteSparse_jll</span></span>
<span class="line"><span>    342.8 ms  ✓ DataValueInterfaces</span></span>
<span class="line"><span>    335.6 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    576.7 ms  ✓ OrderedCollections</span></span>
<span class="line"><span>    352.2 ms  ✓ PtrArrays</span></span>
<span class="line"><span>    384.5 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    474.1 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    393.1 ms  ✓ DataAPI</span></span>
<span class="line"><span>   1098.9 ms  ✓ Baselet</span></span>
<span class="line"><span>   1203.1 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>    439.5 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>    378.5 ms  ✓ TableTraits</span></span>
<span class="line"><span>    437.7 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>    499.6 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>   2022.9 ms  ✓ Distributed</span></span>
<span class="line"><span>    419.1 ms  ✓ NameResolution</span></span>
<span class="line"><span>   3840.5 ms  ✓ Test</span></span>
<span class="line"><span>    421.1 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>   3790.4 ms  ✓ SparseArrays</span></span>
<span class="line"><span>   1721.0 ms  ✓ DataStructures</span></span>
<span class="line"><span>    496.1 ms  ✓ AliasTables</span></span>
<span class="line"><span>    467.6 ms  ✓ Missings</span></span>
<span class="line"><span>    599.5 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>    824.7 ms  ✓ Tables</span></span>
<span class="line"><span>    627.6 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>   1347.1 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    664.7 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>    689.3 ms  ✓ Adapt → AdaptSparseArraysExt</span></span>
<span class="line"><span>   2486.0 ms  ✓ Accessors</span></span>
<span class="line"><span>    698.4 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>    524.0 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>   1054.7 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   1075.5 ms  ✓ MLCore</span></span>
<span class="line"><span>    893.9 ms  ✓ Accessors → LinearAlgebraExt</span></span>
<span class="line"><span>    690.6 ms  ✓ Accessors → TestExt</span></span>
<span class="line"><span>    705.0 ms  ✓ Accessors → StaticArraysExt</span></span>
<span class="line"><span>    806.5 ms  ✓ BangBang</span></span>
<span class="line"><span>    692.3 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>   2328.1 ms  ✓ StatsBase</span></span>
<span class="line"><span>    530.9 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>  17890.5 ms  ✓ MLStyle</span></span>
<span class="line"><span>    533.5 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>   1137.9 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2932.0 ms  ✓ Transducers</span></span>
<span class="line"><span>    716.3 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   4569.4 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   5380.1 ms  ✓ FLoops</span></span>
<span class="line"><span>   6184.9 ms  ✓ MLUtils</span></span>
<span class="line"><span>  55 dependencies successfully precompiled in 36 seconds. 45 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceSparseArraysExt...</span></span>
<span class="line"><span>    623.0 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 8 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    706.0 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1669.8 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2134.3 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 170 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    674.7 ms  ✓ AbstractFFTs</span></span>
<span class="line"><span>    657.7 ms  ✓ RealDot</span></span>
<span class="line"><span>    938.8 ms  ✓ FillArrays</span></span>
<span class="line"><span>    474.7 ms  ✓ HashArrayMappedTries</span></span>
<span class="line"><span>    485.2 ms  ✓ Zlib_jll</span></span>
<span class="line"><span>    675.9 ms  ✓ SuiteSparse</span></span>
<span class="line"><span>    836.5 ms  ✓ StructArrays</span></span>
<span class="line"><span>    471.9 ms  ✓ AbstractFFTs → AbstractFFTsChainRulesCoreExt</span></span>
<span class="line"><span>   1057.1 ms  ✓ ZygoteRules</span></span>
<span class="line"><span>    483.5 ms  ✓ FillArrays → FillArraysStatisticsExt</span></span>
<span class="line"><span>   1994.3 ms  ✓ IRTools</span></span>
<span class="line"><span>    434.1 ms  ✓ ScopedValues</span></span>
<span class="line"><span>    742.8 ms  ✓ FillArrays → FillArraysSparseArraysExt</span></span>
<span class="line"><span>    601.3 ms  ✓ StructArrays → StructArraysAdaptExt</span></span>
<span class="line"><span>    672.8 ms  ✓ SparseInverseSubset</span></span>
<span class="line"><span>   1275.6 ms  ✓ LazyArtifacts</span></span>
<span class="line"><span>    681.7 ms  ✓ StructArrays → StructArraysSparseArraysExt</span></span>
<span class="line"><span>    714.5 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>    450.9 ms  ✓ StructArrays → StructArraysLinearAlgebraExt</span></span>
<span class="line"><span>    772.7 ms  ✓ StructArrays → StructArraysGPUArraysCoreExt</span></span>
<span class="line"><span>   1637.2 ms  ✓ LLVMExtra_jll</span></span>
<span class="line"><span>   5658.4 ms  ✓ ChainRules</span></span>
<span class="line"><span>   6311.0 ms  ✓ LLVM</span></span>
<span class="line"><span>   1781.6 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   4673.4 ms  ✓ GPUArrays</span></span>
<span class="line"><span>  33592.1 ms  ✓ Zygote</span></span>
<span class="line"><span>  26 dependencies successfully precompiled in 53 seconds. 77 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysExt...</span></span>
<span class="line"><span>    642.3 ms  ✓ Accessors → StructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 20 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    494.1 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 24 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    808.4 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    853.0 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 41 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    477.8 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling AbstractFFTsTestExt...</span></span>
<span class="line"><span>   1385.9 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 7 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1960.1 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>   2047.3 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 2 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   1714.0 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>   3025.2 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 3 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling OneHotArrays...</span></span>
<span class="line"><span>    999.8 ms  ✓ OneHotArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 28 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesOneHotArraysExt...</span></span>
<span class="line"><span>    775.3 ms  ✓ MLDataDevices → MLDataDevicesOneHotArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 35 already precompiled.</span></span>
<span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>    387.0 ms  ✓ EnumX</span></span>
<span class="line"><span>    438.4 ms  ✓ ExprTools</span></span>
<span class="line"><span>    702.4 ms  ✓ ExpressionExplorer</span></span>
<span class="line"><span>    387.7 ms  ✓ StructIO</span></span>
<span class="line"><span>    442.9 ms  ✓ Scratch</span></span>
<span class="line"><span>    585.0 ms  ✓ LLVMOpenMP_jll</span></span>
<span class="line"><span>   1179.8 ms  ✓ CUDA_Driver_jll</span></span>
<span class="line"><span>   1413.3 ms  ✓ Enzyme_jll</span></span>
<span class="line"><span>    654.1 ms  ✓ ReactantCore</span></span>
<span class="line"><span>   2160.7 ms  ✓ ObjectFile</span></span>
<span class="line"><span>   3111.5 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>   2530.2 ms  ✓ Reactant_jll</span></span>
<span class="line"><span>  27278.6 ms  ✓ GPUCompiler</span></span>
<span class="line"><span> 220191.9 ms  ✓ Enzyme</span></span>
<span class="line"><span>   5652.2 ms  ✓ Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>  73368.0 ms  ✓ Reactant</span></span>
<span class="line"><span>  16 dependencies successfully precompiled in 331 seconds. 48 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibEnzymeExt...</span></span>
<span class="line"><span>   6765.8 ms  ✓ Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   8497.7 ms  ✓ Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   1362.3 ms  ✓ LuxLib → LuxLibEnzymeExt</span></span>
<span class="line"><span>  11686.6 ms  ✓ Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   6065.3 ms  ✓ Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>  5 dependencies successfully precompiled in 13 seconds. 128 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   6925.8 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 148 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  12186.0 ms  ✓ LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 12 seconds. 69 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  12080.6 ms  ✓ MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 12 seconds. 66 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibReactantExt...</span></span>
<span class="line"><span>  12963.5 ms  ✓ Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>  13332.2 ms  ✓ Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>  13945.3 ms  ✓ LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>  12561.8 ms  ✓ Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>  12331.4 ms  ✓ Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>  13328.0 ms  ✓ Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>  6 dependencies successfully precompiled in 27 seconds. 143 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  12096.1 ms  ✓ WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 12 seconds. 80 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantAbstractFFTsExt...</span></span>
<span class="line"><span>  12149.4 ms  ✓ Reactant → ReactantAbstractFFTsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 12 seconds. 65 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  10406.2 ms  ✓ Lux → LuxReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 11 seconds. 166 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDatasets...</span></span>
<span class="line"><span>    411.9 ms  ✓ LaTeXStrings</span></span>
<span class="line"><span>    420.6 ms  ✓ Glob</span></span>
<span class="line"><span>    439.8 ms  ✓ TensorCore</span></span>
<span class="line"><span>    435.1 ms  ✓ WorkerUtilities</span></span>
<span class="line"><span>    476.5 ms  ✓ BufferedStreams</span></span>
<span class="line"><span>    637.5 ms  ✓ InlineStrings</span></span>
<span class="line"><span>   1091.0 ms  ✓ OffsetArrays</span></span>
<span class="line"><span>    635.4 ms  ✓ URIs</span></span>
<span class="line"><span>    376.6 ms  ✓ SimpleBufferStream</span></span>
<span class="line"><span>    403.3 ms  ✓ LazyModules</span></span>
<span class="line"><span>    568.2 ms  ✓ TranscodingStreams</span></span>
<span class="line"><span>    401.0 ms  ✓ InvertedIndices</span></span>
<span class="line"><span>    346.4 ms  ✓ PackageExtensionCompat</span></span>
<span class="line"><span>    391.7 ms  ✓ BitFlags</span></span>
<span class="line"><span>    437.9 ms  ✓ MappedArrays</span></span>
<span class="line"><span>   1206.5 ms  ✓ Crayons</span></span>
<span class="line"><span>    713.7 ms  ✓ GZip</span></span>
<span class="line"><span>    792.2 ms  ✓ ConcurrentUtilities</span></span>
<span class="line"><span>    638.7 ms  ✓ ZipFile</span></span>
<span class="line"><span>    498.6 ms  ✓ PooledArrays</span></span>
<span class="line"><span>    912.6 ms  ✓ StructTypes</span></span>
<span class="line"><span>   1351.5 ms  ✓ SentinelArrays</span></span>
<span class="line"><span>   2480.8 ms  ✓ FixedPointNumbers</span></span>
<span class="line"><span>   1145.5 ms  ✓ MbedTLS</span></span>
<span class="line"><span>    553.7 ms  ✓ LoggingExtras</span></span>
<span class="line"><span>    581.4 ms  ✓ MPIPreferences</span></span>
<span class="line"><span>    393.1 ms  ✓ InternedStrings</span></span>
<span class="line"><span>    542.0 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>    584.7 ms  ✓ BFloat16s</span></span>
<span class="line"><span>    650.6 ms  ✓ OpenSSL_jll</span></span>
<span class="line"><span>    632.7 ms  ✓ Chemfiles_jll</span></span>
<span class="line"><span>    518.5 ms  ✓ MicrosoftMPI_jll</span></span>
<span class="line"><span>    700.1 ms  ✓ libaec_jll</span></span>
<span class="line"><span>    652.9 ms  ✓ Libiconv_jll</span></span>
<span class="line"><span>   1061.7 ms  ✓ FilePathsBase</span></span>
<span class="line"><span>   2209.1 ms  ✓ StringManipulation</span></span>
<span class="line"><span>    563.3 ms  ✓ InlineStrings → ParsersExt</span></span>
<span class="line"><span>    436.2 ms  ✓ OffsetArrays → OffsetArraysAdaptExt</span></span>
<span class="line"><span>    439.3 ms  ✓ StackViews</span></span>
<span class="line"><span>    459.1 ms  ✓ PaddedViews</span></span>
<span class="line"><span>    520.8 ms  ✓ CodecZlib</span></span>
<span class="line"><span>    476.8 ms  ✓ StridedViews</span></span>
<span class="line"><span>   4684.1 ms  ✓ FileIO</span></span>
<span class="line"><span>   1557.2 ms  ✓ ColorTypes</span></span>
<span class="line"><span>   1192.1 ms  ✓ OpenMPI_jll</span></span>
<span class="line"><span>   1705.7 ms  ✓ MPICH_jll</span></span>
<span class="line"><span>   1392.7 ms  ✓ MPItrampoline_jll</span></span>
<span class="line"><span>   2015.4 ms  ✓ OpenSSL</span></span>
<span class="line"><span>    640.3 ms  ✓ StringEncodings</span></span>
<span class="line"><span>  10143.6 ms  ✓ JSON3</span></span>
<span class="line"><span>    559.6 ms  ✓ FilePathsBase → FilePathsBaseMmapExt</span></span>
<span class="line"><span>   1291.7 ms  ✓ FilePathsBase → FilePathsBaseTestExt</span></span>
<span class="line"><span>    846.7 ms  ✓ WeakRefStrings</span></span>
<span class="line"><span>    507.1 ms  ✓ MosaicViews</span></span>
<span class="line"><span>  21886.9 ms  ✓ Unitful</span></span>
<span class="line"><span>   1562.5 ms  ✓ NPZ</span></span>
<span class="line"><span>   2133.5 ms  ✓ ColorVectorSpace</span></span>
<span class="line"><span>   5022.1 ms  ✓ Colors</span></span>
<span class="line"><span>   1887.6 ms  ✓ HDF5_jll</span></span>
<span class="line"><span>  21073.7 ms  ✓ PrettyTables</span></span>
<span class="line"><span>   2425.7 ms  ✓ Pickle</span></span>
<span class="line"><span>  20024.0 ms  ✓ HTTP</span></span>
<span class="line"><span>    639.3 ms  ✓ Unitful → ConstructionBaseUnitfulExt</span></span>
<span class="line"><span>    626.9 ms  ✓ Unitful → InverseFunctionsUnitfulExt</span></span>
<span class="line"><span>   3136.9 ms  ✓ UnitfulAtomic</span></span>
<span class="line"><span>  35038.7 ms  ✓ JLD2</span></span>
<span class="line"><span>    660.8 ms  ✓ Accessors → UnitfulExt</span></span>
<span class="line"><span>   2491.9 ms  ✓ PeriodicTable</span></span>
<span class="line"><span>  17534.6 ms  ✓ CSV</span></span>
<span class="line"><span>   3718.6 ms  ✓ ColorSchemes</span></span>
<span class="line"><span>   7809.7 ms  ✓ HDF5</span></span>
<span class="line"><span>   3221.0 ms  ✓ DataDeps</span></span>
<span class="line"><span>   1863.1 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>   2328.8 ms  ✓ AtomsBase</span></span>
<span class="line"><span>  19783.1 ms  ✓ ImageCore</span></span>
<span class="line"><span>   2540.0 ms  ✓ MAT</span></span>
<span class="line"><span>   2177.6 ms  ✓ ImageBase</span></span>
<span class="line"><span>   2445.9 ms  ✓ Chemfiles</span></span>
<span class="line"><span>   1918.1 ms  ✓ ImageShow</span></span>
<span class="line"><span>  45995.5 ms  ✓ DataFrames</span></span>
<span class="line"><span>   1445.0 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>   1731.0 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>  10009.6 ms  ✓ MLDatasets</span></span>
<span class="line"><span>  83 dependencies successfully precompiled in 126 seconds. 117 already precompiled.</span></span>
<span class="line"><span>Precompiling SimpleChains...</span></span>
<span class="line"><span>    366.0 ms  ✓ UnPack</span></span>
<span class="line"><span>    517.1 ms  ✓ StaticArrayInterface → StaticArrayInterfaceOffsetArraysExt</span></span>
<span class="line"><span>    886.9 ms  ✓ HostCPUFeatures</span></span>
<span class="line"><span>   7603.0 ms  ✓ VectorizationBase</span></span>
<span class="line"><span>    991.8 ms  ✓ SLEEFPirates</span></span>
<span class="line"><span>   1340.0 ms  ✓ VectorizedRNG</span></span>
<span class="line"><span>    748.8 ms  ✓ VectorizedRNG → VectorizedRNGStaticArraysExt</span></span>
<span class="line"><span>  26938.7 ms  ✓ LoopVectorization</span></span>
<span class="line"><span>   1153.8 ms  ✓ LoopVectorization → SpecialFunctionsExt</span></span>
<span class="line"><span>   1256.2 ms  ✓ LoopVectorization → ForwardDiffExt</span></span>
<span class="line"><span>   6144.0 ms  ✓ SimpleChains</span></span>
<span class="line"><span>  11 dependencies successfully precompiled in 44 seconds. 63 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibSLEEFPiratesExt...</span></span>
<span class="line"><span>   2438.5 ms  ✓ LuxLib → LuxLibSLEEFPiratesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 98 already precompiled.</span></span>
<span class="line"><span>Precompiling ReactantOffsetArraysExt...</span></span>
<span class="line"><span>  11866.4 ms  ✓ Reactant → ReactantOffsetArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 12 seconds. 66 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibLoopVectorizationExt...</span></span>
<span class="line"><span>   4117.7 ms  ✓ LuxLib → LuxLibLoopVectorizationExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxSimpleChainsExt...</span></span>
<span class="line"><span>   2051.5 ms  ✓ Lux → LuxSimpleChainsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 127 already precompiled.</span></span>
<span class="line"><span>2025-03-11 22:47:16.870590: I external/xla/xla/service/service.cc:152] XLA service 0x5907b40 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-11 22:47:16.871013: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1741733236.873427  997531 se_gpu_pjrt_client.cc:951] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1741733236.873626  997531 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741733236.873819  997531 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741733236.890114  997531 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span></code></pre></div><h2 id="Loading-MNIST" tabindex="-1">Loading MNIST <a class="header-anchor" href="#Loading-MNIST" aria-label="Permalink to &quot;Loading MNIST {#Loading-MNIST}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadmnist</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, train_split)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Load MNIST</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> parse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Bool, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ENV</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;CI&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;false&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1500</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> :</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dataset </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> MNIST</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; split </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> :train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_test, y_test) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, y_data); at </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_split)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize, shuffle </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the test data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_test, y_test)); batchsize, shuffle </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>loadmnist (generic function with 1 method)</span></span></code></pre></div><h2 id="Define-the-Model" tabindex="-1">Define the Model <a class="header-anchor" href="#Define-the-Model" aria-label="Permalink to &quot;Define the Model {#Define-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lux_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    MaxPool</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">6</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    MaxPool</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    FlattenLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 84</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">84</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
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
<span class="line"><span>          #        plus 0 states.</span></span></code></pre></div><h2 id="Helper-Functions" tabindex="-1">Helper Functions <a class="header-anchor" href="#Helper-Functions" aria-label="Permalink to &quot;Helper Functions {#Helper-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; logits </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Define-the-Training-Loop" tabindex="-1">Define the Training Loop <a class="header-anchor" href="#Define-the-Training-Loop" aria-label="Permalink to &quot;Define the Training Loop {#Define-the-Training-Loop}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(); rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_dataloader, test_dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> loadmnist</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        tr_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            model_compiled, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, train_dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            100</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            model_compiled, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, test_dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            100</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;[%2d/%2d] </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Time %.2fs </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Training Accuracy: %.2f%% </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Test Accuracy: \\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 %.2f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch nepochs ttime tr_acc te_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> tr_acc, te_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>train (generic function with 2 methods)</span></span></code></pre></div><h2 id="Finally-Training-the-Model" tabindex="-1">Finally Training the Model <a class="header-anchor" href="#Finally-Training-the-Model" aria-label="Permalink to &quot;Finally Training the Model {#Finally-Training-the-Model}&quot;">​</a></h2><p>First we will train the Lux model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tr_acc, te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lux_model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ 1/10] 	 Time 321.37s 	 Training Accuracy: 14.06% 	 Test Accuracy: 17.19%</span></span>
<span class="line"><span>[ 2/10] 	 Time 0.39s 	 Training Accuracy: 29.53% 	 Test Accuracy: 34.38%</span></span>
<span class="line"><span>[ 3/10] 	 Time 0.39s 	 Training Accuracy: 43.75% 	 Test Accuracy: 41.41%</span></span>
<span class="line"><span>[ 4/10] 	 Time 0.39s 	 Training Accuracy: 54.30% 	 Test Accuracy: 50.00%</span></span>
<span class="line"><span>[ 5/10] 	 Time 0.39s 	 Training Accuracy: 63.75% 	 Test Accuracy: 57.81%</span></span>
<span class="line"><span>[ 6/10] 	 Time 0.38s 	 Training Accuracy: 71.25% 	 Test Accuracy: 63.28%</span></span>
<span class="line"><span>[ 7/10] 	 Time 0.39s 	 Training Accuracy: 75.23% 	 Test Accuracy: 66.41%</span></span>
<span class="line"><span>[ 8/10] 	 Time 0.39s 	 Training Accuracy: 79.69% 	 Test Accuracy: 67.19%</span></span>
<span class="line"><span>[ 9/10] 	 Time 0.40s 	 Training Accuracy: 82.97% 	 Test Accuracy: 71.88%</span></span>
<span class="line"><span>[10/10] 	 Time 0.38s 	 Training Accuracy: 84.69% 	 Test Accuracy: 73.44%</span></span></code></pre></div><p>Now we will train the SimpleChains model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tr_acc, te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(simple_chains_model)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ 1/10] 	 Time 903.97s 	 Training Accuracy: 42.03% 	 Test Accuracy: 41.41%</span></span>
<span class="line"><span>[ 2/10] 	 Time 12.28s 	 Training Accuracy: 52.73% 	 Test Accuracy: 52.34%</span></span>
<span class="line"><span>[ 3/10] 	 Time 12.23s 	 Training Accuracy: 60.47% 	 Test Accuracy: 58.59%</span></span>
<span class="line"><span>[ 4/10] 	 Time 12.23s 	 Training Accuracy: 69.38% 	 Test Accuracy: 66.41%</span></span>
<span class="line"><span>[ 5/10] 	 Time 12.22s 	 Training Accuracy: 74.53% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 6/10] 	 Time 12.22s 	 Training Accuracy: 78.44% 	 Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 7/10] 	 Time 12.22s 	 Training Accuracy: 80.47% 	 Test Accuracy: 78.91%</span></span>
<span class="line"><span>[ 8/10] 	 Time 12.24s 	 Training Accuracy: 82.50% 	 Test Accuracy: 79.69%</span></span>
<span class="line"><span>[ 9/10] 	 Time 12.23s 	 Training Accuracy: 84.61% 	 Test Accuracy: 82.03%</span></span>
<span class="line"><span>[10/10] 	 Time 12.23s 	 Training Accuracy: 87.03% 	 Test Accuracy: 82.03%</span></span></code></pre></div><p>On my local machine we see a 3-4x speedup when using SimpleChains.jl. The conditions of the server this documentation is being built on is not ideal for CPU benchmarking hence, the speedup may not be as significant and even there might be regressions.</p><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  JULIA_DEBUG = Literate</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,33)]))}const o=a(l,[["render",e]]);export{E as __pageData,o as default};
