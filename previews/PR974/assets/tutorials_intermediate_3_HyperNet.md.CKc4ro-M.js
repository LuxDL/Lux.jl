import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.BTmvXJuB.js";const d=JSON.parse('{"title":"Training a HyperNetwork on MNIST and FashionMNIST","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/3_HyperNet.md","filePath":"tutorials/intermediate/3_HyperNet.md","lastUpdated":null}'),l={name:"tutorials/intermediate/3_HyperNet.md"};function t(e,s,h,k,r,c){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" tabindex="-1">Training a HyperNetwork on MNIST and FashionMNIST <a class="header-anchor" href="#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" aria-label="Permalink to &quot;Training a HyperNetwork on MNIST and FashionMNIST {#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST}&quot;">​</a></h1><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, ADTypes, ComponentArrays, LuxCUDA, MLDatasets, MLUtils, OneHotArrays, Optimisers,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Printf, Random, Setfield, Statistics, Zygote</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">allowscalar</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    752.1 ms  ✓ LLVMLoopInfo</span></span>
<span class="line"><span>    626.3 ms  ✓ ConcreteStructs</span></span>
<span class="line"><span>    824.0 ms  ✓ AbstractFFTs</span></span>
<span class="line"><span>    648.2 ms  ✓ LaTeXStrings</span></span>
<span class="line"><span>    680.4 ms  ✓ ExprTools</span></span>
<span class="line"><span>    600.9 ms  ✓ IteratorInterfaceExtensions</span></span>
<span class="line"><span>    662.8 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    640.2 ms  ✓ Future</span></span>
<span class="line"><span>    881.6 ms  ✓ ADTypes</span></span>
<span class="line"><span>   1614.7 ms  ✓ UnsafeAtomics</span></span>
<span class="line"><span>   1137.7 ms  ✓ InitialValues</span></span>
<span class="line"><span>    632.6 ms  ✓ CEnum</span></span>
<span class="line"><span>    589.1 ms  ✓ UnPack</span></span>
<span class="line"><span>   1422.1 ms  ✓ OffsetArrays</span></span>
<span class="line"><span>    650.0 ms  ✓ OpenLibm_jll</span></span>
<span class="line"><span>   1285.8 ms  ✓ FillArrays</span></span>
<span class="line"><span>    732.6 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    834.3 ms  ✓ Statistics</span></span>
<span class="line"><span>    685.6 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    715.8 ms  ✓ ArgCheck</span></span>
<span class="line"><span>    797.1 ms  ✓ ShowCases</span></span>
<span class="line"><span>    775.9 ms  ✓ CompilerSupportLibraries_jll</span></span>
<span class="line"><span>    722.2 ms  ✓ SuiteSparse_jll</span></span>
<span class="line"><span>    691.8 ms  ✓ ManualMemory</span></span>
<span class="line"><span>   1296.5 ms  ✓ RandomNumbers</span></span>
<span class="line"><span>   2428.7 ms  ✓ Distributed</span></span>
<span class="line"><span>    778.9 ms  ✓ OrderedCollections</span></span>
<span class="line"><span>    779.7 ms  ✓ Requires</span></span>
<span class="line"><span>    631.8 ms  ✓ DataValueInterfaces</span></span>
<span class="line"><span>    632.7 ms  ✓ RealDot</span></span>
<span class="line"><span>┌ Warning: attempting to remove probably stale pidfile</span></span>
<span class="line"><span>│   path = &quot;/root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/compiled/v1.11/Zlib_jll/xjq3Q_EHVjQ.ji.pidfile&quot;</span></span>
<span class="line"><span>└ @ FileWatching.Pidfile ~/.cache/julia-buildkite-plugin/julia_installs/bin/linux/x64/1.11/julia-1.11-latest-linux-x86_64/share/julia/stdlib/v1.11/FileWatching/src/pidfile.jl:249</span></span>
<span class="line"><span>    673.6 ms  ✓ Reexport</span></span>
<span class="line"><span>    599.0 ms  ✓ SIMDTypes</span></span>
<span class="line"><span>    744.1 ms  ✓ Zlib_jll</span></span>
<span class="line"><span>    950.9 ms  ✓ InlineStrings</span></span>
<span class="line"><span>    613.4 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>   1198.2 ms  ✓ IrrationalConstants</span></span>
<span class="line"><span>    756.9 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    896.3 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    753.3 ms  ✓ InvertedIndices</span></span>
<span class="line"><span>   1954.9 ms  ✓ CUDA_Runtime_Discovery</span></span>
<span class="line"><span>    642.6 ms  ✓ IfElse</span></span>
<span class="line"><span>    988.0 ms  ✓ EnzymeCore</span></span>
<span class="line"><span>    839.7 ms  ✓ ConstructionBase</span></span>
<span class="line"><span>    637.8 ms  ✓ CommonWorldInvalidations</span></span>
<span class="line"><span>    655.7 ms  ✓ DataAPI</span></span>
<span class="line"><span>    618.1 ms  ✓ FastClosures</span></span>
<span class="line"><span>   1518.8 ms  ✓ Crayons</span></span>
<span class="line"><span>    815.3 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>   1148.9 ms  ✓ Functors</span></span>
<span class="line"><span>   1459.7 ms  ✓ Baselet</span></span>
<span class="line"><span>   1852.8 ms  ✓ SentinelArrays</span></span>
<span class="line"><span>    847.5 ms  ✓ Scratch</span></span>
<span class="line"><span>   2817.1 ms  ✓ MacroTools</span></span>
<span class="line"><span>    981.5 ms  ✓ CpuId</span></span>
<span class="line"><span>    893.6 ms  ✓ Compat</span></span>
<span class="line"><span>   4321.8 ms  ✓ Test</span></span>
<span class="line"><span>    932.1 ms  ✓ DocStringExtensions</span></span>
<span class="line"><span>    780.5 ms  ✓ JLLWrappers</span></span>
<span class="line"><span>  18885.7 ms  ✓ MLStyle</span></span>
<span class="line"><span>   2364.3 ms  ✓ StringManipulation</span></span>
<span class="line"><span>    708.2 ms  ✓ TableTraits</span></span>
<span class="line"><span>    843.0 ms  ✓ Atomix</span></span>
<span class="line"><span>   3177.3 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>    725.2 ms  ✓ NaNMath</span></span>
<span class="line"><span>    680.2 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>    695.0 ms  ✓ FillArrays → FillArraysStatisticsExt</span></span>
<span class="line"><span>    692.1 ms  ✓ NameResolution</span></span>
<span class="line"><span>   1162.5 ms  ✓ ThreadingUtilities</span></span>
<span class="line"><span>   1434.1 ms  ✓ Random123</span></span>
<span class="line"><span>    925.4 ms  ✓ Adapt</span></span>
<span class="line"><span>   4676.9 ms  ✓ SparseArrays</span></span>
<span class="line"><span>   1519.3 ms  ✓ LazyArtifacts</span></span>
<span class="line"><span>    776.5 ms  ✓ InlineStrings → ParsersExt</span></span>
<span class="line"><span>  23156.9 ms  ✓ Unitful</span></span>
<span class="line"><span>    637.4 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>    632.5 ms  ✓ ADTypes → ADTypesEnzymeCoreExt</span></span>
<span class="line"><span>    636.3 ms  ✓ ConstructionBase → ConstructionBaseLinearAlgebraExt</span></span>
<span class="line"><span>    784.9 ms  ✓ PooledArrays</span></span>
<span class="line"><span>    770.5 ms  ✓ Missings</span></span>
<span class="line"><span>   1215.0 ms  ✓ Static</span></span>
<span class="line"><span>    705.4 ms  ✓ DiffResults</span></span>
<span class="line"><span>   1005.7 ms  ✓ CommonSubexpressions</span></span>
<span class="line"><span>   2012.8 ms  ✓ DispatchDoctor</span></span>
<span class="line"><span>   2379.3 ms  ✓ IRTools</span></span>
<span class="line"><span>   1460.9 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>    732.6 ms  ✓ Compat → CompatLinearAlgebraExt</span></span>
<span class="line"><span>    903.1 ms  ✓ BFloat16s</span></span>
<span class="line"><span>    925.8 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>   1746.3 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>    920.7 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>   1287.1 ms  ✓ CUDA_Driver_jll</span></span>
<span class="line"><span>    956.7 ms  ✓ NVTX_jll</span></span>
<span class="line"><span>    931.5 ms  ✓ Hwloc_jll</span></span>
<span class="line"><span>   8178.1 ms  ✓ StaticArrays</span></span>
<span class="line"><span>    874.0 ms  ✓ demumble_jll</span></span>
<span class="line"><span>    889.3 ms  ✓ JuliaNVTXCallbacks_jll</span></span>
<span class="line"><span>    959.8 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>   1200.5 ms  ✓ Tables</span></span>
<span class="line"><span>    760.0 ms  ✓ OffsetArrays → OffsetArraysAdaptExt</span></span>
<span class="line"><span>    647.4 ms  ✓ EnzymeCore → AdaptExt</span></span>
<span class="line"><span>    937.8 ms  ✓ SuiteSparse</span></span>
<span class="line"><span>   1015.9 ms  ✓ FillArrays → FillArraysSparseArraysExt</span></span>
<span class="line"><span>    962.4 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>    878.5 ms  ✓ Unitful → InverseFunctionsUnitfulExt</span></span>
<span class="line"><span>   1701.2 ms  ✓ LLVMExtra_jll</span></span>
<span class="line"><span>    878.4 ms  ✓ Unitful → ConstructionBaseUnitfulExt</span></span>
<span class="line"><span>    709.1 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>   5033.2 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>    716.6 ms  ✓ DispatchDoctor → DispatchDoctorEnzymeCoreExt</span></span>
<span class="line"><span>   1436.7 ms  ✓ CPUSummary</span></span>
<span class="line"><span>    715.5 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>   1777.9 ms  ✓ ChainRulesCore</span></span>
<span class="line"><span>    763.4 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>   2270.1 ms  ✓ DataStructures</span></span>
<span class="line"><span>    901.2 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>   2585.7 ms  ✓ Hwloc</span></span>
<span class="line"><span>   2670.3 ms  ✓ CUDA_Runtime_jll</span></span>
<span class="line"><span>    912.5 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    928.8 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    986.4 ms  ✓ SparseInverseSubset</span></span>
<span class="line"><span>   1848.3 ms  ✓ LossFunctions</span></span>
<span class="line"><span>   2830.8 ms  ✓ FixedPointNumbers</span></span>
<span class="line"><span>   1075.7 ms  ✓ HostCPUFeatures</span></span>
<span class="line"><span>   1040.7 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>    964.9 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>    998.6 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>    729.0 ms  ✓ AbstractFFTs → AbstractFFTsChainRulesCoreExt</span></span>
<span class="line"><span>    906.9 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    920.4 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>   7811.2 ms  ✓ LLVM</span></span>
<span class="line"><span>   1626.3 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    928.3 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    810.7 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>   2137.1 ms  ✓ CUDNN_jll</span></span>
<span class="line"><span>   3338.5 ms  ✓ Accessors</span></span>
<span class="line"><span>   1877.7 ms  ✓ Setfield</span></span>
<span class="line"><span>   1164.8 ms  ✓ ArrayInterface</span></span>
<span class="line"><span>   1199.1 ms  ✓ StructArrays</span></span>
<span class="line"><span>    958.1 ms  ✓ GPUArraysCore</span></span>
<span class="line"><span>   1162.5 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>   1645.4 ms  ✓ ZygoteRules</span></span>
<span class="line"><span>   2805.2 ms  ✓ ColorTypes</span></span>
<span class="line"><span>   1495.6 ms  ✓ Optimisers</span></span>
<span class="line"><span>  21855.2 ms  ✓ PrettyTables</span></span>
<span class="line"><span>   1677.1 ms  ✓ LuxCore</span></span>
<span class="line"><span>   1633.9 ms  ✓ LLVM → BFloat16sExt</span></span>
<span class="line"><span>    986.3 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>   3037.2 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>   1379.3 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>   2893.4 ms  ✓ StatsBase</span></span>
<span class="line"><span>    667.1 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>   1582.5 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    974.4 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>    694.7 ms  ✓ StructArrays → StructArraysAdaptExt</span></span>
<span class="line"><span>   1139.8 ms  ✓ StructArrays → StructArraysSparseArraysExt</span></span>
<span class="line"><span>   1004.0 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>    781.5 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>    765.2 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>    875.3 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>    777.7 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>   1020.0 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>    896.0 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>    765.1 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    824.0 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    944.3 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>   4774.4 ms  ✓ Colors</span></span>
<span class="line"><span>   2549.7 ms  ✓ UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   2712.4 ms  ✓ GPUArrays</span></span>
<span class="line"><span>   2103.4 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    970.7 ms  ✓ Accessors → AccessorsUnitfulExt</span></span>
<span class="line"><span>    962.7 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>    666.9 ms  ✓ StructArrays → StructArraysGPUArraysCoreExt</span></span>
<span class="line"><span>   1002.7 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>   1844.5 ms  ✓ NVTX</span></span>
<span class="line"><span>   5376.0 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>   1653.1 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>    978.0 ms  ✓ DiffRules</span></span>
<span class="line"><span>   3336.3 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>   1207.5 ms  ✓ BangBang</span></span>
<span class="line"><span>  29194.7 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>   2065.0 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>   6611.1 ms  ✓ ChainRules</span></span>
<span class="line"><span>   1808.6 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>   1389.4 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>   1169.1 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>   4727.9 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    761.6 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    770.3 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>   1992.4 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   2218.4 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   1118.4 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>   1452.9 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>    764.2 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  50598.5 ms  ✓ DataFrames</span></span>
<span class="line"><span>   7379.3 ms  ✓ NNlib</span></span>
<span class="line"><span>   2035.0 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>   2142.8 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>   2089.1 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   1242.5 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   3231.7 ms  ✓ Transducers</span></span>
<span class="line"><span>   1864.9 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>   6123.5 ms  ✓ FLoops</span></span>
<span class="line"><span>  37430.6 ms  ✓ Zygote</span></span>
<span class="line"><span>   1882.7 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>   2217.2 ms  ✓ Zygote → ZygoteColorsExt</span></span>
<span class="line"><span>  60473.5 ms  ✓ CUDA</span></span>
<span class="line"><span>   5482.0 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5651.3 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   5675.8 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5409.3 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>   5377.4 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>   5821.0 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>   2030.1 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>    957.0 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    779.8 ms  ✓ StaticArrayInterface → StaticArrayInterfaceOffsetArraysExt</span></span>
<span class="line"><span>    772.7 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>   5391.6 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>    914.9 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>   1394.6 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>   1263.0 ms  ✓ Polyester</span></span>
<span class="line"><span>   9142.7 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5803.0 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>   6348.2 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>   9474.6 ms  ✓ VectorizationBase</span></span>
<span class="line"><span>   1658.1 ms  ✓ SLEEFPirates</span></span>
<span class="line"><span>   8308.5 ms  ✓ MLUtils</span></span>
<span class="line"><span>   2937.3 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  30580.6 ms  ✓ LoopVectorization</span></span>
<span class="line"><span>   1533.7 ms  ✓ LoopVectorization → SpecialFunctionsExt</span></span>
<span class="line"><span>   1671.5 ms  ✓ LoopVectorization → ForwardDiffExt</span></span>
<span class="line"><span>   5416.1 ms  ✓ Octavian</span></span>
<span class="line"><span>   5038.9 ms  ✓ Octavian → ForwardDiffExt</span></span>
<span class="line"><span>  11739.1 ms  ✓ LuxLib</span></span>
<span class="line"><span>   6320.8 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>   6609.8 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  11486.5 ms  ✓ Lux</span></span>
<span class="line"><span>   4009.4 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>   4805.2 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  238 dependencies successfully precompiled in 314 seconds. 28 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArrays...</span></span>
<span class="line"><span>    596.9 ms  ✓ PackageExtensionCompat</span></span>
<span class="line"><span>   1238.9 ms  ✓ ComponentArrays</span></span>
<span class="line"><span>    839.0 ms  ✓ ComponentArrays → ComponentArraysAdaptExt</span></span>
<span class="line"><span>   2063.3 ms  ✓ ComponentArrays → ComponentArraysGPUArraysExt</span></span>
<span class="line"><span>  4 dependencies successfully precompiled in 5 seconds. 155 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxComponentArraysExt...</span></span>
<span class="line"><span>    779.7 ms  ✓ ComponentArrays → ComponentArraysOptimisersExt</span></span>
<span class="line"><span>    850.8 ms  ✓ ComponentArrays → ComponentArraysConstructionBaseExt</span></span>
<span class="line"><span>   1983.2 ms  ✓ ComponentArrays → ComponentArraysZygoteExt</span></span>
<span class="line"><span>   2914.5 ms  ✓ Lux → LuxComponentArraysExt</span></span>
<span class="line"><span>  4 dependencies successfully precompiled in 5 seconds. 270 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>   5550.1 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 123 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDatasets...</span></span>
<span class="line"><span>    686.6 ms  ✓ Glob</span></span>
<span class="line"><span>    723.3 ms  ✓ TensorCore</span></span>
<span class="line"><span>┌ Warning: attempting to remove probably stale pidfile</span></span>
<span class="line"><span>│   path = &quot;/root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/compiled/v1.11/TranscodingStreams/eJQ7D_EHVjQ.ji.pidfile&quot;</span></span>
<span class="line"><span>└ @ FileWatching.Pidfile ~/.cache/julia-buildkite-plugin/julia_installs/bin/linux/x64/1.11/julia-1.11-latest-linux-x86_64/share/julia/stdlib/v1.11/FileWatching/src/pidfile.jl:249</span></span>
<span class="line"><span>   1012.9 ms  ✓ WorkerUtilities</span></span>
<span class="line"><span>    764.3 ms  ✓ BufferedStreams</span></span>
<span class="line"><span>    671.5 ms  ✓ LazyModules</span></span>
<span class="line"><span>   1197.2 ms  ✓ TranscodingStreams</span></span>
<span class="line"><span>    711.1 ms  ✓ MappedArrays</span></span>
<span class="line"><span>   1002.3 ms  ✓ GZip</span></span>
<span class="line"><span>   1030.8 ms  ✓ ConcurrentUtilities</span></span>
<span class="line"><span>    936.8 ms  ✓ ZipFile</span></span>
<span class="line"><span>    746.8 ms  ✓ StridedViews</span></span>
<span class="line"><span>    701.2 ms  ✓ StackViews</span></span>
<span class="line"><span>    715.3 ms  ✓ PaddedViews</span></span>
<span class="line"><span>    817.7 ms  ✓ LoggingExtras</span></span>
<span class="line"><span>   1209.4 ms  ✓ StructTypes</span></span>
<span class="line"><span>   1383.1 ms  ✓ MbedTLS</span></span>
<span class="line"><span>    890.2 ms  ✓ MPIPreferences</span></span>
<span class="line"><span>    786.4 ms  ✓ InternedStrings</span></span>
<span class="line"><span>    818.2 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>   3196.1 ms  ✓ UnitfulAtomic</span></span>
<span class="line"><span>    986.3 ms  ✓ OpenSSL_jll</span></span>
<span class="line"><span>   3069.1 ms  ✓ PeriodicTable</span></span>
<span class="line"><span>    905.5 ms  ✓ Chemfiles_jll</span></span>
<span class="line"><span>    779.1 ms  ✓ MicrosoftMPI_jll</span></span>
<span class="line"><span>    922.8 ms  ✓ libaec_jll</span></span>
<span class="line"><span>    912.9 ms  ✓ Libiconv_jll</span></span>
<span class="line"><span>   1419.5 ms  ✓ FilePathsBase</span></span>
<span class="line"><span>   1119.6 ms  ✓ WeakRefStrings</span></span>
<span class="line"><span>    758.6 ms  ✓ CodecZlib</span></span>
<span class="line"><span>   2845.9 ms  ✓ ColorVectorSpace</span></span>
<span class="line"><span>    751.2 ms  ✓ MosaicViews</span></span>
<span class="line"><span>   5002.9 ms  ✓ FileIO</span></span>
<span class="line"><span>   1966.8 ms  ✓ MPICH_jll</span></span>
<span class="line"><span>   5190.9 ms  ✓ StridedViews → StridedViewsCUDAExt</span></span>
<span class="line"><span>   1598.4 ms  ✓ MPItrampoline_jll</span></span>
<span class="line"><span>   1641.3 ms  ✓ OpenMPI_jll</span></span>
<span class="line"><span>   2478.2 ms  ✓ OpenSSL</span></span>
<span class="line"><span>   1885.0 ms  ✓ AtomsBase</span></span>
<span class="line"><span>    855.9 ms  ✓ StringEncodings</span></span>
<span class="line"><span>    821.3 ms  ✓ FilePathsBase → FilePathsBaseMmapExt</span></span>
<span class="line"><span>   1123.5 ms  ✓ ColorVectorSpace → SpecialFunctionsExt</span></span>
<span class="line"><span>   1688.9 ms  ✓ FilePathsBase → FilePathsBaseTestExt</span></span>
<span class="line"><span>   2010.4 ms  ✓ NPZ</span></span>
<span class="line"><span>   2477.0 ms  ✓ HDF5_jll</span></span>
<span class="line"><span>  12910.6 ms  ✓ JSON3</span></span>
<span class="line"><span>   2962.2 ms  ✓ Chemfiles</span></span>
<span class="line"><span>   4583.9 ms  ✓ HTTP</span></span>
<span class="line"><span>   2850.3 ms  ✓ Pickle</span></span>
<span class="line"><span>   4436.7 ms  ✓ ColorSchemes</span></span>
<span class="line"><span>  20284.9 ms  ✓ ImageCore</span></span>
<span class="line"><span>  21584.6 ms  ✓ CSV</span></span>
<span class="line"><span>  37404.6 ms  ✓ JLD2</span></span>
<span class="line"><span>   3684.7 ms  ✓ DataDeps</span></span>
<span class="line"><span>   8968.9 ms  ✓ HDF5</span></span>
<span class="line"><span>   2709.7 ms  ✓ ImageBase</span></span>
<span class="line"><span>   2337.2 ms  ✓ ImageShow</span></span>
<span class="line"><span>   3177.7 ms  ✓ MAT</span></span>
<span class="line"><span>  10311.4 ms  ✓ MLDatasets</span></span>
<span class="line"><span>  58 dependencies successfully precompiled in 76 seconds. 189 already precompiled.</span></span>
<span class="line"><span>Precompiling OneHotArrays...</span></span>
<span class="line"><span>   2251.9 ms  ✓ OneHotArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 134 already precompiled.</span></span></code></pre></div><h2 id="Loading-Datasets" tabindex="-1">Loading Datasets <a class="header-anchor" href="#Loading-Datasets" aria-label="Permalink to &quot;Loading Datasets {#Loading-Datasets}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> load_dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{dset}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_train</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_eval</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {dset}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    imgs, labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n_train]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_train, y_train </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_train), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(labels, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    imgs, labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:test</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n_eval]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_test, y_test </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_eval), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(labels, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_train, y_train); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">min</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, n_train), shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_test, y_test); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">min</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, n_eval), shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> load_datasets</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(n_train</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1024</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_eval</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> load_dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((MNIST, FashionMNIST), n_train, n_eval, batchsize)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>load_datasets (generic function with 4 methods)</span></span></code></pre></div><h2 id="Implement-a-HyperNet-Layer" tabindex="-1">Implement a HyperNet Layer <a class="header-anchor" href="#Implement-a-HyperNet-Layer" aria-label="Permalink to &quot;Implement a HyperNet Layer {#Implement-a-HyperNet-Layer}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> HyperNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        weight_generator</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Lux.AbstractLuxLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, core_network</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Lux.AbstractLuxLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ca_axes </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">initialparameters</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), core_network) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">              ComponentArray </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">              getaxes</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; ca_axes, weight_generator, core_network, dispatch</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:HyperNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Generate the weights</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ps_new </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ComponentArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">weight_generator</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), ca_axes)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> core_network</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps_new)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>HyperNet (generic function with 1 method)</span></span></code></pre></div><p>Defining functions on the CompactLuxLayer requires some understanding of how the layer is structured, as such we don&#39;t recommend doing it unless you are familiar with the internals. In this case, we simply write it to ignore the initialization of the <code>core_network</code> parameters.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">initialparameters</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractRNG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, hn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">CompactLuxLayer{:HyperNet}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (; weight_generator</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">initialparameters</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, hn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">weight_generator),)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Create-and-Initialize-the-HyperNet" tabindex="-1">Create and Initialize the HyperNet <a class="header-anchor" href="#Create-and-Initialize-the-HyperNet" aria-label="Permalink to &quot;Create and Initialize the HyperNet {#Create-and-Initialize-the-HyperNet}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> create_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Doesn&#39;t need to be a MLP can have any Lux Layer</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    core_network </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">FlattenLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">784</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    weight_generator </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Embedding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">parameterlength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(core_network)))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> HyperNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(weight_generator, core_network)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>create_model (generic function with 1 method)</span></span></code></pre></div><h2 id="Define-Utility-Functions" tabindex="-1">Define Utility Functions <a class="header-anchor" href="#Define-Utility-Functions" aria-label="Permalink to &quot;Define Utility Functions {#Define-Utility-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; logits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, dataloader, data_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    total_correct, total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        predicted_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((data_idx, x), ps, st)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> predicted_class)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(target_class)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="training" tabindex="-1">Training <a class="header-anchor" href="#training" aria-label="Permalink to &quot;Training&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> create_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dataloaders </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> load_datasets</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3.0f-4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    ### Lets train the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    nepochs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 25</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nepochs, data_idx </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_dataloader, test_dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloaders[data_idx] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        stime </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> time</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, _, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), loss, ((data_idx, x), y), train_state)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ttime </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> time</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> stime</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> round</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, train_dataloader, data_idx) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            digits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        test_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> round</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, test_dataloader, data_idx) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            digits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        data_name </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data_idx </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> ?</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;MNIST&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> :</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;FashionMNIST&quot;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;[%3d/%3d] </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> %12s </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Time %.5fs </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Training Accuracy: %.2f%% </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Test \\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 Accuracy: %.2f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch nepochs data_name ttime train_acc test_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    test_acc_list </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data_idx </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_dataloader, test_dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloaders[data_idx] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> round</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, train_dataloader, data_idx) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            digits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        test_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> round</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, test_dataloader, data_idx) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            digits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        data_name </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data_idx </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> ?</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;MNIST&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> :</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;FashionMNIST&quot;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;[FINAL] </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> %12s </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Training Accuracy: %.2f%% </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Test Accuracy: \\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 %.2f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data_name train_acc test_acc</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        test_acc_list[data_idx] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> test_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> test_acc_list</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">test_acc_list </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[  1/ 25] 	        MNIST 	 Time 89.28104s 	 Training Accuracy: 24.12% 	 Test Accuracy: 25.00%</span></span>
<span class="line"><span>[  1/ 25] 	 FashionMNIST 	 Time 0.02714s 	 Training Accuracy: 28.61% 	 Test Accuracy: 28.12%</span></span>
<span class="line"><span>[  2/ 25] 	        MNIST 	 Time 0.02739s 	 Training Accuracy: 49.32% 	 Test Accuracy: 34.38%</span></span>
<span class="line"><span>[  2/ 25] 	 FashionMNIST 	 Time 0.04006s 	 Training Accuracy: 52.34% 	 Test Accuracy: 43.75%</span></span>
<span class="line"><span>[  3/ 25] 	        MNIST 	 Time 0.09945s 	 Training Accuracy: 60.06% 	 Test Accuracy: 53.12%</span></span>
<span class="line"><span>[  3/ 25] 	 FashionMNIST 	 Time 0.03278s 	 Training Accuracy: 56.54% 	 Test Accuracy: 50.00%</span></span>
<span class="line"><span>[  4/ 25] 	        MNIST 	 Time 0.07086s 	 Training Accuracy: 66.11% 	 Test Accuracy: 40.62%</span></span>
<span class="line"><span>[  4/ 25] 	 FashionMNIST 	 Time 0.05720s 	 Training Accuracy: 68.75% 	 Test Accuracy: 53.12%</span></span>
<span class="line"><span>[  5/ 25] 	        MNIST 	 Time 0.02500s 	 Training Accuracy: 76.17% 	 Test Accuracy: 56.25%</span></span>
<span class="line"><span>[  5/ 25] 	 FashionMNIST 	 Time 0.03614s 	 Training Accuracy: 74.12% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[  6/ 25] 	        MNIST 	 Time 0.02294s 	 Training Accuracy: 78.91% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[  6/ 25] 	 FashionMNIST 	 Time 0.02205s 	 Training Accuracy: 76.07% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  7/ 25] 	        MNIST 	 Time 0.02275s 	 Training Accuracy: 82.81% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  7/ 25] 	 FashionMNIST 	 Time 0.02286s 	 Training Accuracy: 78.42% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  8/ 25] 	        MNIST 	 Time 0.02284s 	 Training Accuracy: 85.84% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  8/ 25] 	 FashionMNIST 	 Time 0.02140s 	 Training Accuracy: 79.39% 	 Test Accuracy: 62.50%</span></span>
<span class="line"><span>[  9/ 25] 	        MNIST 	 Time 0.02223s 	 Training Accuracy: 88.77% 	 Test Accuracy: 71.88%</span></span>
<span class="line"><span>[  9/ 25] 	 FashionMNIST 	 Time 0.02195s 	 Training Accuracy: 81.05% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 10/ 25] 	        MNIST 	 Time 0.02153s 	 Training Accuracy: 89.75% 	 Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 10/ 25] 	 FashionMNIST 	 Time 0.02085s 	 Training Accuracy: 81.25% 	 Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 11/ 25] 	        MNIST 	 Time 0.02128s 	 Training Accuracy: 92.77% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 11/ 25] 	 FashionMNIST 	 Time 0.02127s 	 Training Accuracy: 83.69% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 12/ 25] 	        MNIST 	 Time 0.02056s 	 Training Accuracy: 92.77% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 12/ 25] 	 FashionMNIST 	 Time 0.02195s 	 Training Accuracy: 84.57% 	 Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 13/ 25] 	        MNIST 	 Time 0.02354s 	 Training Accuracy: 95.12% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 13/ 25] 	 FashionMNIST 	 Time 0.02163s 	 Training Accuracy: 84.08% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 14/ 25] 	        MNIST 	 Time 0.02127s 	 Training Accuracy: 96.00% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 14/ 25] 	 FashionMNIST 	 Time 0.02118s 	 Training Accuracy: 84.38% 	 Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 15/ 25] 	        MNIST 	 Time 0.02130s 	 Training Accuracy: 97.56% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 15/ 25] 	 FashionMNIST 	 Time 0.02325s 	 Training Accuracy: 87.50% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 16/ 25] 	        MNIST 	 Time 0.02143s 	 Training Accuracy: 97.27% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 16/ 25] 	 FashionMNIST 	 Time 0.02085s 	 Training Accuracy: 86.52% 	 Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 17/ 25] 	        MNIST 	 Time 0.02590s 	 Training Accuracy: 97.46% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 17/ 25] 	 FashionMNIST 	 Time 0.02214s 	 Training Accuracy: 86.13% 	 Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 18/ 25] 	        MNIST 	 Time 0.02112s 	 Training Accuracy: 98.24% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 18/ 25] 	 FashionMNIST 	 Time 0.02459s 	 Training Accuracy: 87.01% 	 Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 19/ 25] 	        MNIST 	 Time 0.02174s 	 Training Accuracy: 98.73% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 19/ 25] 	 FashionMNIST 	 Time 0.02193s 	 Training Accuracy: 90.72% 	 Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 20/ 25] 	        MNIST 	 Time 0.02184s 	 Training Accuracy: 99.51% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 20/ 25] 	 FashionMNIST 	 Time 0.02299s 	 Training Accuracy: 91.60% 	 Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 21/ 25] 	        MNIST 	 Time 0.02111s 	 Training Accuracy: 99.80% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 21/ 25] 	 FashionMNIST 	 Time 0.02235s 	 Training Accuracy: 92.68% 	 Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 22/ 25] 	        MNIST 	 Time 0.02271s 	 Training Accuracy: 100.00% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 22/ 25] 	 FashionMNIST 	 Time 0.02149s 	 Training Accuracy: 92.58% 	 Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 23/ 25] 	        MNIST 	 Time 0.02145s 	 Training Accuracy: 100.00% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 23/ 25] 	 FashionMNIST 	 Time 0.02229s 	 Training Accuracy: 92.77% 	 Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 24/ 25] 	        MNIST 	 Time 0.02116s 	 Training Accuracy: 99.90% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 24/ 25] 	 FashionMNIST 	 Time 0.02242s 	 Training Accuracy: 92.38% 	 Test Accuracy: 56.25%</span></span>
<span class="line"><span>[ 25/ 25] 	        MNIST 	 Time 0.02100s 	 Training Accuracy: 99.71% 	 Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 25/ 25] 	 FashionMNIST 	 Time 0.02285s 	 Training Accuracy: 92.97% 	 Test Accuracy: 59.38%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>[FINAL] 	        MNIST 	 Training Accuracy: 100.00% 	 Test Accuracy: 65.62%</span></span>
<span class="line"><span>[FINAL] 	 FashionMNIST 	 Training Accuracy: 92.97% 	 Test Accuracy: 59.38%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.0</span></span>
<span class="line"><span>Commit 501a4f25c2b (2024-10-07 11:40 UTC)</span></span>
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
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.6.3</span></span>
<span class="line"><span>- CURAND: 10.3.7</span></span>
<span class="line"><span>- CUFFT: 11.3.0</span></span>
<span class="line"><span>- CUSOLVER: 11.7.1</span></span>
<span class="line"><span>- CUSPARSE: 12.5.4</span></span>
<span class="line"><span>- CUPTI: 2024.3.2 (API 24.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.5.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.10.3+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.15.3+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.11.0</span></span>
<span class="line"><span>- LLVM: 16.0.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Preferences:</span></span>
<span class="line"><span>- CUDA_Driver_jll.compat: false</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 3.232 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,26)]))}const y=a(l,[["render",t]]);export{d as __pageData,y as default};
