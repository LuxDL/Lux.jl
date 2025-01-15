import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.bV3h_rQg.js";const E=JSON.parse('{"title":"Training a HyperNetwork on MNIST and FashionMNIST","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/3_HyperNet.md","filePath":"tutorials/intermediate/3_HyperNet.md","lastUpdated":null}'),l={name:"tutorials/intermediate/3_HyperNet.md"};function e(t,s,h,c,k,r){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" tabindex="-1">Training a HyperNetwork on MNIST and FashionMNIST <a class="header-anchor" href="#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" aria-label="Permalink to &quot;Training a HyperNetwork on MNIST and FashionMNIST {#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST}&quot;">​</a></h1><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, ADTypes, ComponentArrays, LuxCUDA, MLDatasets, MLUtils, OneHotArrays, Optimisers,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Printf, Random, Setfield, Statistics, Zygote</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">allowscalar</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    543.6 ms  ✓ Requires</span></span>
<span class="line"><span>    710.6 ms  ✓ CpuId</span></span>
<span class="line"><span>    588.5 ms  ✓ Compat</span></span>
<span class="line"><span>    639.5 ms  ✓ DocStringExtensions</span></span>
<span class="line"><span>    485.4 ms  ✓ JLLWrappers</span></span>
<span class="line"><span>    783.9 ms  ✓ Static</span></span>
<span class="line"><span>   2506.1 ms  ✓ MacroTools</span></span>
<span class="line"><span>    386.8 ms  ✓ Compat → CompatLinearAlgebraExt</span></span>
<span class="line"><span>    594.7 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    601.3 ms  ✓ Hwloc_jll</span></span>
<span class="line"><span>    643.5 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>    426.7 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>    686.6 ms  ✓ CommonSubexpressions</span></span>
<span class="line"><span>    998.0 ms  ✓ CPUSummary</span></span>
<span class="line"><span>   1603.3 ms  ✓ DispatchDoctor</span></span>
<span class="line"><span>   1441.7 ms  ✓ Setfield</span></span>
<span class="line"><span>    613.2 ms  ✓ Functors</span></span>
<span class="line"><span>   1257.4 ms  ✓ ChainRulesCore</span></span>
<span class="line"><span>   1546.2 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>   7598.8 ms  ✓ StaticArrays</span></span>
<span class="line"><span>   2169.8 ms  ✓ Hwloc</span></span>
<span class="line"><span>    659.7 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>    454.4 ms  ✓ DispatchDoctor → DispatchDoctorEnzymeCoreExt</span></span>
<span class="line"><span>    899.5 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>   1238.5 ms  ✓ LuxCore</span></span>
<span class="line"><span>   2803.9 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    414.5 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    401.9 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>   1232.4 ms  ✓ Optimisers</span></span>
<span class="line"><span>    704.0 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>    464.2 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>   1392.9 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    610.9 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    645.6 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    622.5 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    620.5 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    605.8 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    683.9 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    641.0 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>    633.6 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>    497.2 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    454.0 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    479.7 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    497.7 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>   1754.4 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    443.8 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    429.2 ms  ✓ Optimisers → OptimisersAdaptExt</span></span>
<span class="line"><span>   2741.3 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    966.6 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>   3792.8 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    978.7 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    802.9 ms  ✓ Polyester</span></span>
<span class="line"><span>   1039.2 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   4007.1 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    677.5 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    730.1 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5209.0 ms  ✓ NNlib</span></span>
<span class="line"><span>    871.4 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    971.1 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   6050.0 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9108.4 ms  ✓ Lux</span></span>
<span class="line"><span>  61 dependencies successfully precompiled in 44 seconds. 48 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArrays...</span></span>
<span class="line"><span>    911.9 ms  ✓ ComponentArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesComponentArraysExt...</span></span>
<span class="line"><span>    608.1 ms  ✓ MLDataDevices → MLDataDevicesComponentArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 49 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxComponentArraysExt...</span></span>
<span class="line"><span>    522.8 ms  ✓ ComponentArrays → ComponentArraysOptimisersExt</span></span>
<span class="line"><span>   1627.6 ms  ✓ Lux → LuxComponentArraysExt</span></span>
<span class="line"><span>   2109.0 ms  ✓ ComponentArrays → ComponentArraysKernelAbstractionsExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 2 seconds. 111 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>    318.2 ms  ✓ LLVMLoopInfo</span></span>
<span class="line"><span>    612.1 ms  ✓ InlineStrings</span></span>
<span class="line"><span>    385.6 ms  ✓ InvertedIndices</span></span>
<span class="line"><span>   1503.2 ms  ✓ CUDA_Runtime_Discovery</span></span>
<span class="line"><span>   1156.1 ms  ✓ Crayons</span></span>
<span class="line"><span>    492.3 ms  ✓ Scratch</span></span>
<span class="line"><span>   1391.3 ms  ✓ SentinelArrays</span></span>
<span class="line"><span>   2869.8 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>    906.1 ms  ✓ CUDA_Driver_jll</span></span>
<span class="line"><span>    595.4 ms  ✓ NVTX_jll</span></span>
<span class="line"><span>    611.1 ms  ✓ demumble_jll</span></span>
<span class="line"><span>    591.8 ms  ✓ JuliaNVTXCallbacks_jll</span></span>
<span class="line"><span>   1659.9 ms  ✓ DataStructures</span></span>
<span class="line"><span>   1960.3 ms  ✓ StringManipulation</span></span>
<span class="line"><span>   4069.8 ms  ✓ Test</span></span>
<span class="line"><span>    953.9 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   1070.2 ms  ✓ LazyArtifacts</span></span>
<span class="line"><span>   1312.6 ms  ✓ NVTX</span></span>
<span class="line"><span>    562.2 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>    513.8 ms  ✓ BFloat16s</span></span>
<span class="line"><span>   1379.2 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>   1378.4 ms  ✓ LLVMExtra_jll</span></span>
<span class="line"><span>   2577.2 ms  ✓ CUDA_Runtime_jll</span></span>
<span class="line"><span>   2030.5 ms  ✓ CUDNN_jll</span></span>
<span class="line"><span>   6720.9 ms  ✓ LLVM</span></span>
<span class="line"><span>   1329.0 ms  ✓ LLVM → BFloat16sExt</span></span>
<span class="line"><span>   1855.0 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   2134.6 ms  ✓ GPUArrays</span></span>
<span class="line"><span>  20451.6 ms  ✓ PrettyTables</span></span>
<span class="line"><span>  27293.6 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>  47629.2 ms  ✓ DataFrames</span></span>
<span class="line"><span>  52709.9 ms  ✓ CUDA</span></span>
<span class="line"><span>   5002.3 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8262.1 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5381.5 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  35 dependencies successfully precompiled in 147 seconds. 65 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesGPUArraysExt...</span></span>
<span class="line"><span>   1359.0 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 42 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersGPUArraysExt...</span></span>
<span class="line"><span>   1451.4 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArraysGPUArraysExt...</span></span>
<span class="line"><span>   1611.3 ms  ✓ ComponentArrays → ComponentArraysGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 68 already precompiled.</span></span>
<span class="line"><span>Precompiling ParsersExt...</span></span>
<span class="line"><span>    476.9 ms  ✓ InlineStrings → ParsersExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 9 already precompiled.</span></span>
<span class="line"><span>Precompiling ChainRulesCoreSparseArraysExt...</span></span>
<span class="line"><span>    637.4 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 11 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    680.8 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 17 already precompiled.</span></span>
<span class="line"><span>Precompiling AbstractFFTsChainRulesCoreExt...</span></span>
<span class="line"><span>    430.8 ms  ✓ AbstractFFTs → AbstractFFTsChainRulesCoreExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 9 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   4949.0 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   5322.5 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5447.0 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   5090.7 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5324.6 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   5398.9 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5940.3 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 6 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   5063.8 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5455.8 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5202.5 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 107 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   6557.9 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 174 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDatasets...</span></span>
<span class="line"><span>    356.1 ms  ✓ Glob</span></span>
<span class="line"><span>    369.1 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    409.6 ms  ✓ WorkerUtilities</span></span>
<span class="line"><span>    423.7 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    432.6 ms  ✓ BufferedStreams</span></span>
<span class="line"><span>    824.2 ms  ✓ InitialValues</span></span>
<span class="line"><span>    362.4 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    416.9 ms  ✓ ShowCases</span></span>
<span class="line"><span>    327.9 ms  ✓ SimpleBufferStream</span></span>
<span class="line"><span>    338.4 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    585.7 ms  ✓ URIs</span></span>
<span class="line"><span>    336.8 ms  ✓ PtrArrays</span></span>
<span class="line"><span>    333.6 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    541.6 ms  ✓ TranscodingStreams</span></span>
<span class="line"><span>    437.9 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    333.8 ms  ✓ PackageExtensionCompat</span></span>
<span class="line"><span>    356.7 ms  ✓ BitFlags</span></span>
<span class="line"><span>    578.2 ms  ✓ ZipFile</span></span>
<span class="line"><span>   1102.8 ms  ✓ Baselet</span></span>
<span class="line"><span>    787.1 ms  ✓ StructTypes</span></span>
<span class="line"><span>   1126.2 ms  ✓ MbedTLS</span></span>
<span class="line"><span>    542.9 ms  ✓ LoggingExtras</span></span>
<span class="line"><span>   1061.7 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>    597.1 ms  ✓ MPIPreferences</span></span>
<span class="line"><span>    405.1 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>   1171.7 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    338.5 ms  ✓ InternedStrings</span></span>
<span class="line"><span>    504.7 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>    723.8 ms  ✓ OpenSSL_jll</span></span>
<span class="line"><span>    631.1 ms  ✓ Chemfiles_jll</span></span>
<span class="line"><span>    610.2 ms  ✓ libaec_jll</span></span>
<span class="line"><span>    460.2 ms  ✓ MicrosoftMPI_jll</span></span>
<span class="line"><span>    572.9 ms  ✓ Libiconv_jll</span></span>
<span class="line"><span>  18469.7 ms  ✓ MLStyle</span></span>
<span class="line"><span>   3540.9 ms  ✓ ColorSchemes</span></span>
<span class="line"><span>   1027.4 ms  ✓ FilePathsBase</span></span>
<span class="line"><span>  21863.3 ms  ✓ Unitful</span></span>
<span class="line"><span>    783.4 ms  ✓ WeakRefStrings</span></span>
<span class="line"><span>    611.9 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>    395.1 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>    452.7 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    390.1 ms  ✓ NameResolution</span></span>
<span class="line"><span>    365.6 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>    442.0 ms  ✓ AliasTables</span></span>
<span class="line"><span>   4508.0 ms  ✓ FileIO</span></span>
<span class="line"><span>    448.5 ms  ✓ CodecZlib</span></span>
<span class="line"><span>    434.0 ms  ✓ StridedViews</span></span>
<span class="line"><span>   1438.1 ms  ✓ MPICH_jll</span></span>
<span class="line"><span>  19788.9 ms  ✓ ImageCore</span></span>
<span class="line"><span>   1196.1 ms  ✓ MPItrampoline_jll</span></span>
<span class="line"><span>    603.0 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>   1198.6 ms  ✓ OpenMPI_jll</span></span>
<span class="line"><span>    528.6 ms  ✓ StringEncodings</span></span>
<span class="line"><span>    510.8 ms  ✓ FilePathsBase → FilePathsBaseMmapExt</span></span>
<span class="line"><span>   1939.3 ms  ✓ OpenSSL</span></span>
<span class="line"><span>   1337.3 ms  ✓ FilePathsBase → FilePathsBaseTestExt</span></span>
<span class="line"><span>    612.8 ms  ✓ Unitful → ConstructionBaseUnitfulExt</span></span>
<span class="line"><span>    614.0 ms  ✓ Unitful → InverseFunctionsUnitfulExt</span></span>
<span class="line"><span>   2756.5 ms  ✓ UnitfulAtomic</span></span>
<span class="line"><span>   2442.0 ms  ✓ PeriodicTable</span></span>
<span class="line"><span>  11347.7 ms  ✓ JSON3</span></span>
<span class="line"><span>   2952.1 ms  ✓ Accessors</span></span>
<span class="line"><span>   4636.8 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   1603.5 ms  ✓ NPZ</span></span>
<span class="line"><span>   2300.5 ms  ✓ StatsBase</span></span>
<span class="line"><span>   1819.1 ms  ✓ HDF5_jll</span></span>
<span class="line"><span>   2167.1 ms  ✓ ImageBase</span></span>
<span class="line"><span>   2401.1 ms  ✓ Pickle</span></span>
<span class="line"><span>  19737.5 ms  ✓ CSV</span></span>
<span class="line"><span>  19538.9 ms  ✓ HTTP</span></span>
<span class="line"><span>   2362.7 ms  ✓ AtomsBase</span></span>
<span class="line"><span>    619.9 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>    678.2 ms  ✓ Accessors → AccessorsUnitfulExt</span></span>
<span class="line"><span>    851.8 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>    718.8 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>    860.4 ms  ✓ BangBang</span></span>
<span class="line"><span>   2004.1 ms  ✓ ImageShow</span></span>
<span class="line"><span>   3325.3 ms  ✓ DataDeps</span></span>
<span class="line"><span>   2224.6 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>   7896.5 ms  ✓ HDF5</span></span>
<span class="line"><span>    756.1 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>  35208.6 ms  ✓ JLD2</span></span>
<span class="line"><span>    542.6 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>   2614.9 ms  ✓ Chemfiles</span></span>
<span class="line"><span>    666.5 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>   1937.9 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>   1040.9 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2406.4 ms  ✓ MAT</span></span>
<span class="line"><span>   2659.4 ms  ✓ Transducers</span></span>
<span class="line"><span>    684.3 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   1519.9 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>   5406.6 ms  ✓ FLoops</span></span>
<span class="line"><span>   6440.7 ms  ✓ MLUtils</span></span>
<span class="line"><span>   9066.8 ms  ✓ MLDatasets</span></span>
<span class="line"><span>  94 dependencies successfully precompiled in 104 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1630.5 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2297.5 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling OneHotArrays...</span></span>
<span class="line"><span>    950.1 ms  ✓ OneHotArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 28 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesOneHotArraysExt...</span></span>
<span class="line"><span>    753.4 ms  ✓ MLDataDevices → MLDataDevicesOneHotArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 35 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>   1045.6 ms  ✓ ZygoteRules</span></span>
<span class="line"><span>   1960.0 ms  ✓ IRTools</span></span>
<span class="line"><span>   5593.5 ms  ✓ ChainRules</span></span>
<span class="line"><span>  34893.3 ms  ✓ Zygote</span></span>
<span class="line"><span>  4 dependencies successfully precompiled in 41 seconds. 82 already precompiled.</span></span>
<span class="line"><span>Precompiling AccessorsStructArraysExt...</span></span>
<span class="line"><span>    488.8 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 16 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    504.2 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 22 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysStaticArraysExt...</span></span>
<span class="line"><span>    664.6 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    809.9 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 39 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    881.0 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    445.8 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1643.7 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   3049.1 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 163 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArraysZygoteExt...</span></span>
<span class="line"><span>   1737.2 ms  ✓ ComponentArrays → ComponentArraysZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 99 already precompiled.</span></span>
<span class="line"><span>Precompiling ZygoteColorsExt...</span></span>
<span class="line"><span>   1900.2 ms  ✓ Zygote → ZygoteColorsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 89 already precompiled.</span></span></code></pre></div><h2 id="Loading-Datasets" tabindex="-1">Loading Datasets <a class="header-anchor" href="#Loading-Datasets" aria-label="Permalink to &quot;Loading Datasets {#Loading-Datasets}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> load_dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{dset}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_train</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Nothing, Int}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        n_eval</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Nothing, Int}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {dset}</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> n_train </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">===</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        imgs, labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        imgs, labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n_train]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_train, y_train </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_train), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(labels, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> n_eval </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">===</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        imgs, labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:test</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        imgs, labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:test</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n_eval]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_test, y_test </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_eval), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(labels, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_train, y_train); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">min</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, n_train), shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_test, y_test); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">min</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, n_eval), shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> load_datasets</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    n_train </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> parse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Bool, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ENV</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;CI&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;false&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1024</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> :</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    n_eval </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> parse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Bool, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ENV</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;CI&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;false&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 32</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> :</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> load_dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((MNIST, FashionMNIST), n_train, n_eval, batchsize)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>load_datasets (generic function with 2 methods)</span></span></code></pre></div><h2 id="Implement-a-HyperNet-Layer" tabindex="-1">Implement a HyperNet Layer <a class="header-anchor" href="#Implement-a-HyperNet-Layer" aria-label="Permalink to &quot;Implement a HyperNet Layer {#Implement-a-HyperNet-Layer}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> HyperNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    weight_generator </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Embedding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">parameterlength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(core_network))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.001f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    ### Lets train the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    nepochs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 50</span></span>
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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;[%3d/%3d]</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">%12s</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Time %3.5fs</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Training Accuracy: %3.2f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Test \\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 Accuracy: %3.2f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch nepochs data_name ttime train_acc test_acc</span></span>
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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;[FINAL]</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">%12s</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Training Accuracy: %3.2f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Test Accuracy: \\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 %3.2f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data_name train_acc test_acc</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        test_acc_list[data_idx] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> test_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> test_acc_list</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">test_acc_list </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[  1/ 50]	       MNIST	Time 88.66682s	Training Accuracy: 59.47%	Test Accuracy: 46.88%</span></span>
<span class="line"><span>[  1/ 50]	FashionMNIST	Time 0.03187s	Training Accuracy: 48.93%	Test Accuracy: 46.88%</span></span>
<span class="line"><span>[  2/ 50]	       MNIST	Time 0.02846s	Training Accuracy: 69.14%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  2/ 50]	FashionMNIST	Time 0.02836s	Training Accuracy: 64.45%	Test Accuracy: 56.25%</span></span>
<span class="line"><span>[  3/ 50]	       MNIST	Time 0.02319s	Training Accuracy: 79.59%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[  3/ 50]	FashionMNIST	Time 0.02140s	Training Accuracy: 67.68%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[  4/ 50]	       MNIST	Time 0.02217s	Training Accuracy: 78.32%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[  4/ 50]	FashionMNIST	Time 0.02375s	Training Accuracy: 65.72%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[  5/ 50]	       MNIST	Time 0.02758s	Training Accuracy: 79.98%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[  5/ 50]	FashionMNIST	Time 0.03723s	Training Accuracy: 71.09%	Test Accuracy: 56.25%</span></span>
<span class="line"><span>[  6/ 50]	       MNIST	Time 0.02309s	Training Accuracy: 87.79%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[  6/ 50]	FashionMNIST	Time 0.02134s	Training Accuracy: 71.88%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[  7/ 50]	       MNIST	Time 0.02080s	Training Accuracy: 87.89%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[  7/ 50]	FashionMNIST	Time 0.02157s	Training Accuracy: 70.90%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  8/ 50]	       MNIST	Time 0.02117s	Training Accuracy: 91.80%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[  8/ 50]	FashionMNIST	Time 0.02095s	Training Accuracy: 73.83%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[  9/ 50]	       MNIST	Time 0.02107s	Training Accuracy: 93.65%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[  9/ 50]	FashionMNIST	Time 0.02032s	Training Accuracy: 76.86%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 10/ 50]	       MNIST	Time 0.02135s	Training Accuracy: 95.70%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 10/ 50]	FashionMNIST	Time 0.02073s	Training Accuracy: 79.30%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 11/ 50]	       MNIST	Time 0.02084s	Training Accuracy: 97.75%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 11/ 50]	FashionMNIST	Time 0.02082s	Training Accuracy: 80.86%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 12/ 50]	       MNIST	Time 0.02268s	Training Accuracy: 98.54%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 12/ 50]	FashionMNIST	Time 0.02251s	Training Accuracy: 81.74%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 13/ 50]	       MNIST	Time 0.02012s	Training Accuracy: 98.93%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 13/ 50]	FashionMNIST	Time 0.01983s	Training Accuracy: 80.76%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 14/ 50]	       MNIST	Time 0.02065s	Training Accuracy: 99.22%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 14/ 50]	FashionMNIST	Time 0.02009s	Training Accuracy: 82.71%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 15/ 50]	       MNIST	Time 0.02003s	Training Accuracy: 99.32%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 15/ 50]	FashionMNIST	Time 0.02451s	Training Accuracy: 83.98%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 16/ 50]	       MNIST	Time 0.02831s	Training Accuracy: 99.51%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 16/ 50]	FashionMNIST	Time 0.01958s	Training Accuracy: 85.55%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 17/ 50]	       MNIST	Time 0.01997s	Training Accuracy: 99.90%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 17/ 50]	FashionMNIST	Time 0.02447s	Training Accuracy: 85.16%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 18/ 50]	       MNIST	Time 0.02742s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 18/ 50]	FashionMNIST	Time 0.02154s	Training Accuracy: 84.96%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 19/ 50]	       MNIST	Time 0.02036s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 19/ 50]	FashionMNIST	Time 0.02286s	Training Accuracy: 86.04%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 20/ 50]	       MNIST	Time 0.02207s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 20/ 50]	FashionMNIST	Time 0.02385s	Training Accuracy: 86.72%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 21/ 50]	       MNIST	Time 0.02773s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 21/ 50]	FashionMNIST	Time 0.02148s	Training Accuracy: 87.11%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 22/ 50]	       MNIST	Time 0.02177s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 22/ 50]	FashionMNIST	Time 0.02219s	Training Accuracy: 87.60%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 23/ 50]	       MNIST	Time 0.02591s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 23/ 50]	FashionMNIST	Time 0.02209s	Training Accuracy: 88.48%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 24/ 50]	       MNIST	Time 0.02209s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 24/ 50]	FashionMNIST	Time 0.02181s	Training Accuracy: 88.77%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 25/ 50]	       MNIST	Time 0.02596s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 25/ 50]	FashionMNIST	Time 0.02189s	Training Accuracy: 89.06%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 26/ 50]	       MNIST	Time 0.02080s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 26/ 50]	FashionMNIST	Time 0.02891s	Training Accuracy: 90.04%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 27/ 50]	       MNIST	Time 0.02120s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 27/ 50]	FashionMNIST	Time 0.02074s	Training Accuracy: 89.84%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 28/ 50]	       MNIST	Time 0.02146s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 28/ 50]	FashionMNIST	Time 0.02332s	Training Accuracy: 90.23%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 29/ 50]	       MNIST	Time 0.02089s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 29/ 50]	FashionMNIST	Time 0.02084s	Training Accuracy: 90.33%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 30/ 50]	       MNIST	Time 0.02113s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 30/ 50]	FashionMNIST	Time 0.02321s	Training Accuracy: 90.72%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 31/ 50]	       MNIST	Time 0.02106s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 31/ 50]	FashionMNIST	Time 0.02116s	Training Accuracy: 90.92%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 32/ 50]	       MNIST	Time 0.02667s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 32/ 50]	FashionMNIST	Time 0.02244s	Training Accuracy: 91.50%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 33/ 50]	       MNIST	Time 0.02115s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 33/ 50]	FashionMNIST	Time 0.02088s	Training Accuracy: 91.11%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 34/ 50]	       MNIST	Time 0.02179s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 34/ 50]	FashionMNIST	Time 0.02223s	Training Accuracy: 91.80%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 35/ 50]	       MNIST	Time 0.02239s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 35/ 50]	FashionMNIST	Time 0.02110s	Training Accuracy: 92.58%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 36/ 50]	       MNIST	Time 0.02196s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 36/ 50]	FashionMNIST	Time 0.02104s	Training Accuracy: 92.19%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 37/ 50]	       MNIST	Time 0.02194s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 37/ 50]	FashionMNIST	Time 0.02691s	Training Accuracy: 92.87%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 38/ 50]	       MNIST	Time 0.02119s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 38/ 50]	FashionMNIST	Time 0.02103s	Training Accuracy: 92.09%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 39/ 50]	       MNIST	Time 0.02184s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 39/ 50]	FashionMNIST	Time 0.02116s	Training Accuracy: 93.26%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 40/ 50]	       MNIST	Time 0.02258s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 40/ 50]	FashionMNIST	Time 0.02175s	Training Accuracy: 92.77%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 41/ 50]	       MNIST	Time 0.02175s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 41/ 50]	FashionMNIST	Time 0.02191s	Training Accuracy: 93.07%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 42/ 50]	       MNIST	Time 0.02116s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 42/ 50]	FashionMNIST	Time 0.02059s	Training Accuracy: 93.46%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 43/ 50]	       MNIST	Time 0.02533s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 43/ 50]	FashionMNIST	Time 0.02106s	Training Accuracy: 93.65%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 44/ 50]	       MNIST	Time 0.02313s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 44/ 50]	FashionMNIST	Time 0.02102s	Training Accuracy: 93.36%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 45/ 50]	       MNIST	Time 0.02249s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 45/ 50]	FashionMNIST	Time 0.02092s	Training Accuracy: 93.36%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 46/ 50]	       MNIST	Time 0.02170s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 46/ 50]	FashionMNIST	Time 0.02259s	Training Accuracy: 93.16%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 47/ 50]	       MNIST	Time 0.02142s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 47/ 50]	FashionMNIST	Time 0.02260s	Training Accuracy: 94.53%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 48/ 50]	       MNIST	Time 0.02200s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 48/ 50]	FashionMNIST	Time 0.02694s	Training Accuracy: 94.53%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 49/ 50]	       MNIST	Time 0.02129s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 49/ 50]	FashionMNIST	Time 0.02028s	Training Accuracy: 94.53%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 50/ 50]	       MNIST	Time 0.02142s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 50/ 50]	FashionMNIST	Time 0.02170s	Training Accuracy: 93.75%	Test Accuracy: 78.12%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[FINAL]	FashionMNIST	Training Accuracy: 93.75%	Test Accuracy: 78.12%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 3.201 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,26)]))}const y=a(l,[["render",e]]);export{E as __pageData,y as default};
