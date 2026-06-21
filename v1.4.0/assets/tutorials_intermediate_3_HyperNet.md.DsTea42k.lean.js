import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.MfBj6Zyp.js";const d=JSON.parse('{"title":"Training a HyperNetwork on MNIST and FashionMNIST","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/3_HyperNet.md","filePath":"tutorials/intermediate/3_HyperNet.md","lastUpdated":null}'),t={name:"tutorials/intermediate/3_HyperNet.md"};function l(e,s,h,k,r,c){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" tabindex="-1">Training a HyperNetwork on MNIST and FashionMNIST <a class="header-anchor" href="#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" aria-label="Permalink to &quot;Training a HyperNetwork on MNIST and FashionMNIST {#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST}&quot;">​</a></h1><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, ADTypes, ComponentArrays, LuxCUDA, MLDatasets, MLUtils, OneHotArrays, Optimisers,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Printf, Random, Setfield, Statistics, Zygote</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">allowscalar</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    674.6 ms  ✓ ConcreteStructs</span></span>
<span class="line"><span>    581.7 ms  ✓ ExprTools</span></span>
<span class="line"><span>    726.5 ms  ✓ AbstractFFTs</span></span>
<span class="line"><span>    496.3 ms  ✓ IteratorInterfaceExtensions</span></span>
<span class="line"><span>    583.7 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    565.1 ms  ✓ Future</span></span>
<span class="line"><span>   1480.7 ms  ✓ UnsafeAtomics</span></span>
<span class="line"><span>    842.5 ms  ✓ ADTypes</span></span>
<span class="line"><span>   1037.9 ms  ✓ InitialValues</span></span>
<span class="line"><span>    578.3 ms  ✓ CEnum</span></span>
<span class="line"><span>    796.6 ms  ✓ Serialization</span></span>
<span class="line"><span>    630.0 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    785.8 ms  ✓ Statistics</span></span>
<span class="line"><span>    573.2 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    595.2 ms  ✓ ArgCheck</span></span>
<span class="line"><span>    633.3 ms  ✓ ShowCases</span></span>
<span class="line"><span>    656.4 ms  ✓ CompilerSupportLibraries_jll</span></span>
<span class="line"><span>    632.6 ms  ✓ SuiteSparse_jll</span></span>
<span class="line"><span>    505.1 ms  ✓ DataValueInterfaces</span></span>
<span class="line"><span>    710.4 ms  ✓ OrderedCollections</span></span>
<span class="line"><span>    510.4 ms  ✓ Reexport</span></span>
<span class="line"><span>    584.9 ms  ✓ Zlib_jll</span></span>
<span class="line"><span>    610.4 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    539.5 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    645.6 ms  ✓ Adapt</span></span>
<span class="line"><span>    565.2 ms  ✓ DataAPI</span></span>
<span class="line"><span>    601.0 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>   1433.5 ms  ✓ Baselet</span></span>
<span class="line"><span>    640.5 ms  ✓ AbstractFFTs → AbstractFFTsChainRulesCoreExt</span></span>
<span class="line"><span>    553.0 ms  ✓ TableTraits</span></span>
<span class="line"><span>    720.5 ms  ✓ Atomix</span></span>
<span class="line"><span>   3042.8 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>    580.7 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    546.3 ms  ✓ ADTypes → ADTypesEnzymeCoreExt</span></span>
<span class="line"><span>    542.1 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>   2455.8 ms  ✓ Hwloc</span></span>
<span class="line"><span>   2291.0 ms  ✓ Distributed</span></span>
<span class="line"><span>    608.6 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>    762.4 ms  ✓ Unitful → InverseFunctionsUnitfulExt</span></span>
<span class="line"><span>    645.6 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    622.7 ms  ✓ FillArrays → FillArraysStatisticsExt</span></span>
<span class="line"><span>    621.9 ms  ✓ NameResolution</span></span>
<span class="line"><span>   4194.2 ms  ✓ Test</span></span>
<span class="line"><span>   2120.3 ms  ✓ DataStructures</span></span>
<span class="line"><span>    834.8 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>   4569.0 ms  ✓ SparseArrays</span></span>
<span class="line"><span>    571.5 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>    631.2 ms  ✓ OffsetArrays → OffsetArraysAdaptExt</span></span>
<span class="line"><span>    572.6 ms  ✓ EnzymeCore → AdaptExt</span></span>
<span class="line"><span>    702.8 ms  ✓ PooledArrays</span></span>
<span class="line"><span>    651.2 ms  ✓ Missings</span></span>
<span class="line"><span>    598.6 ms  ✓ DiffResults</span></span>
<span class="line"><span>  19789.7 ms  ✓ MLStyle</span></span>
<span class="line"><span>   1148.3 ms  ✓ Tables</span></span>
<span class="line"><span>   7400.5 ms  ✓ LLVM</span></span>
<span class="line"><span>   1554.2 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>    844.1 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>    722.3 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>    866.6 ms  ✓ FillArrays → FillArraysSparseArraysExt</span></span>
<span class="line"><span>    845.8 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>   2876.3 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    813.4 ms  ✓ SuiteSparse</span></span>
<span class="line"><span>    820.5 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>   7907.2 ms  ✓ StaticArrays</span></span>
<span class="line"><span>   1469.4 ms  ✓ LLVM → BFloat16sExt</span></span>
<span class="line"><span>   4773.7 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   2030.3 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    805.4 ms  ✓ SparseInverseSubset</span></span>
<span class="line"><span>   2597.4 ms  ✓ FixedPointNumbers</span></span>
<span class="line"><span>    796.8 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>   2768.1 ms  ✓ StatsBase</span></span>
<span class="line"><span>    768.0 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    808.9 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    782.5 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>   2152.7 ms  ✓ UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   4238.3 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>   2634.4 ms  ✓ ColorTypes</span></span>
<span class="line"><span>   3172.1 ms  ✓ Accessors</span></span>
<span class="line"><span>  20809.5 ms  ✓ PrettyTables</span></span>
<span class="line"><span>   1049.1 ms  ✓ StructArrays</span></span>
<span class="line"><span>   1658.1 ms  ✓ Setfield</span></span>
<span class="line"><span>    710.6 ms  ✓ GPUArraysCore</span></span>
<span class="line"><span>   1060.1 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>    818.2 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>    556.5 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>   1116.5 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   5170.8 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    801.9 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>   4609.6 ms  ✓ Colors</span></span>
<span class="line"><span>    987.5 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>    833.3 ms  ✓ Accessors → AccessorsUnitfulExt</span></span>
<span class="line"><span>    899.4 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>    591.2 ms  ✓ StructArrays → StructArraysAdaptExt</span></span>
<span class="line"><span>    860.2 ms  ✓ StructArrays → StructArraysSparseArraysExt</span></span>
<span class="line"><span>    851.6 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>    631.8 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>   1416.0 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>    654.2 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>  28446.1 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>   2683.0 ms  ✓ GPUArrays</span></span>
<span class="line"><span>    574.5 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>    586.8 ms  ✓ StructArrays → StructArraysGPUArraysCoreExt</span></span>
<span class="line"><span>   3144.6 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    805.5 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>    646.9 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>    873.9 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>    709.2 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>   1793.0 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>   2195.7 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   1626.1 ms  ✓ NVTX</span></span>
<span class="line"><span>   1949.4 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   1010.3 ms  ✓ BangBang</span></span>
<span class="line"><span>   1598.2 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>   1945.3 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>   1335.6 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>   6445.9 ms  ✓ ChainRules</span></span>
<span class="line"><span>    981.5 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>    684.2 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    687.3 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>    672.3 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>   7169.6 ms  ✓ NNlib</span></span>
<span class="line"><span>   1010.9 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>   1094.7 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>   2260.1 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>   2141.6 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>  50523.7 ms  ✓ DataFrames</span></span>
<span class="line"><span>   2115.9 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>   1088.0 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   3325.8 ms  ✓ Transducers</span></span>
<span class="line"><span>   1653.9 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>    827.9 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>  36763.5 ms  ✓ Zygote</span></span>
<span class="line"><span>   2299.6 ms  ✓ Zygote → ZygoteColorsExt</span></span>
<span class="line"><span>   5637.7 ms  ✓ FLoops</span></span>
<span class="line"><span>   1932.9 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  60851.6 ms  ✓ CUDA</span></span>
<span class="line"><span>   5607.3 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5713.0 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5755.6 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   5565.3 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>   5532.3 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>   5731.8 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>    860.0 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>   5764.6 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>   9266.4 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5695.7 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>   6169.7 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>   2150.6 ms  ✓ OneHotArrays</span></span>
<span class="line"><span>   2000.2 ms  ✓ MLDataDevices → MLDataDevicesOneHotArraysExt</span></span>
<span class="line"><span>   7667.2 ms  ✓ MLUtils</span></span>
<span class="line"><span>   2773.1 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>   6991.5 ms  ✓ LuxLib</span></span>
<span class="line"><span>   6184.8 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>   6310.6 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  10804.5 ms  ✓ Lux</span></span>
<span class="line"><span>   3758.0 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>   4346.6 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  157 dependencies successfully precompiled in 240 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArrays...</span></span>
<span class="line"><span>   1522.1 ms  ✓ ComponentArrays</span></span>
<span class="line"><span>   2167.8 ms  ✓ ComponentArrays → ComponentArraysGPUArraysExt</span></span>
<span class="line"><span>   3364.0 ms  ✓ ComponentArrays → ComponentArraysKernelAbstractionsExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 6 seconds. 156 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxComponentArraysExt...</span></span>
<span class="line"><span>    717.3 ms  ✓ ComponentArrays → ComponentArraysOptimisersExt</span></span>
<span class="line"><span>   1848.9 ms  ✓ ComponentArrays → ComponentArraysZygoteExt</span></span>
<span class="line"><span>   2468.5 ms  ✓ Lux → LuxComponentArraysExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 4 seconds. 266 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>   5812.7 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 123 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDatasets...</span></span>
<span class="line"><span>    748.0 ms  ✓ TranscodingStreams</span></span>
<span class="line"><span>    867.4 ms  ✓ GZip</span></span>
<span class="line"><span>    999.3 ms  ✓ ConcurrentUtilities</span></span>
<span class="line"><span>    831.7 ms  ✓ ZipFile</span></span>
<span class="line"><span>    707.2 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>    995.3 ms  ✓ WeakRefStrings</span></span>
<span class="line"><span>   2410.2 ms  ✓ ColorVectorSpace</span></span>
<span class="line"><span>   1772.1 ms  ✓ MPICH_jll</span></span>
<span class="line"><span>   1488.5 ms  ✓ FilePathsBase → FilePathsBaseTestExt</span></span>
<span class="line"><span>   1570.1 ms  ✓ CodecZlib</span></span>
<span class="line"><span>   2542.0 ms  ✓ AtomsBase</span></span>
<span class="line"><span>    944.4 ms  ✓ ColorVectorSpace → SpecialFunctionsExt</span></span>
<span class="line"><span>   2266.1 ms  ✓ HDF5_jll</span></span>
<span class="line"><span>   5365.0 ms  ✓ StridedViews → StridedViewsCUDAExt</span></span>
<span class="line"><span>   2652.3 ms  ✓ Chemfiles</span></span>
<span class="line"><span>  20864.9 ms  ✓ CSV</span></span>
<span class="line"><span>  20584.1 ms  ✓ HTTP</span></span>
<span class="line"><span>  19962.3 ms  ✓ ImageCore</span></span>
<span class="line"><span>   4499.2 ms  ✓ ColorSchemes</span></span>
<span class="line"><span>   2801.7 ms  ✓ Pickle</span></span>
<span class="line"><span>   3777.2 ms  ✓ DataDeps</span></span>
<span class="line"><span>   2365.0 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>   8569.2 ms  ✓ HDF5</span></span>
<span class="line"><span>   2488.5 ms  ✓ ImageBase</span></span>
<span class="line"><span>   1835.7 ms  ✓ NPZ</span></span>
<span class="line"><span>   2704.4 ms  ✓ MAT</span></span>
<span class="line"><span>   2303.0 ms  ✓ ImageShow</span></span>
<span class="line"><span>  35275.9 ms  ✓ JLD2</span></span>
<span class="line"><span>  10826.6 ms  ✓ MLDatasets</span></span>
<span class="line"><span>  29 dependencies successfully precompiled in 83 seconds. 221 already precompiled.</span></span></code></pre></div><h2 id="Loading-Datasets" tabindex="-1">Loading Datasets <a class="header-anchor" href="#Loading-Datasets" aria-label="Permalink to &quot;Loading Datasets {#Loading-Datasets}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> load_dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{dset}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_train</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_eval</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {dset}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    imgs, labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n_train]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_train, y_train </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_train), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(labels, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    imgs, labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:test</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n_eval]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_test, y_test </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_eval), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(labels, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_train, y_train); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">min</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, n_train), shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_test, y_test); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">min</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, n_eval), shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">test_acc_list </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[  1/ 50]	       MNIST	Time 88.45370s	Training Accuracy: 58.01%	Test Accuracy: 46.88%</span></span>
<span class="line"><span>[  1/ 50]	FashionMNIST	Time 0.02975s	Training Accuracy: 50.98%	Test Accuracy: 46.88%</span></span>
<span class="line"><span>[  2/ 50]	       MNIST	Time 0.06977s	Training Accuracy: 67.19%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  2/ 50]	FashionMNIST	Time 0.02853s	Training Accuracy: 59.47%	Test Accuracy: 53.12%</span></span>
<span class="line"><span>[  3/ 50]	       MNIST	Time 0.02840s	Training Accuracy: 78.12%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[  3/ 50]	FashionMNIST	Time 0.02953s	Training Accuracy: 69.63%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[  4/ 50]	       MNIST	Time 0.03371s	Training Accuracy: 78.32%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[  4/ 50]	FashionMNIST	Time 0.02054s	Training Accuracy: 65.62%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[  5/ 50]	       MNIST	Time 0.02133s	Training Accuracy: 77.25%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  5/ 50]	FashionMNIST	Time 0.02112s	Training Accuracy: 72.07%	Test Accuracy: 56.25%</span></span>
<span class="line"><span>[  6/ 50]	       MNIST	Time 0.02102s	Training Accuracy: 86.62%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[  6/ 50]	FashionMNIST	Time 0.02481s	Training Accuracy: 74.41%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[  7/ 50]	       MNIST	Time 0.02066s	Training Accuracy: 88.57%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[  7/ 50]	FashionMNIST	Time 0.02103s	Training Accuracy: 73.24%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[  8/ 50]	       MNIST	Time 0.02169s	Training Accuracy: 90.92%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[  8/ 50]	FashionMNIST	Time 0.02057s	Training Accuracy: 76.95%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[  9/ 50]	       MNIST	Time 0.02215s	Training Accuracy: 93.07%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[  9/ 50]	FashionMNIST	Time 0.02106s	Training Accuracy: 80.66%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 10/ 50]	       MNIST	Time 0.02085s	Training Accuracy: 95.70%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 10/ 50]	FashionMNIST	Time 0.02199s	Training Accuracy: 81.25%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 11/ 50]	       MNIST	Time 0.02279s	Training Accuracy: 97.36%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 11/ 50]	FashionMNIST	Time 0.02128s	Training Accuracy: 81.45%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 12/ 50]	       MNIST	Time 0.02074s	Training Accuracy: 98.05%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 12/ 50]	FashionMNIST	Time 0.02565s	Training Accuracy: 81.84%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 13/ 50]	       MNIST	Time 0.02064s	Training Accuracy: 98.54%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 13/ 50]	FashionMNIST	Time 0.02058s	Training Accuracy: 81.93%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 14/ 50]	       MNIST	Time 0.02040s	Training Accuracy: 99.02%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 14/ 50]	FashionMNIST	Time 0.02248s	Training Accuracy: 85.16%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 15/ 50]	       MNIST	Time 0.02229s	Training Accuracy: 99.32%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 15/ 50]	FashionMNIST	Time 0.02102s	Training Accuracy: 85.84%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 16/ 50]	       MNIST	Time 0.02095s	Training Accuracy: 99.32%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 16/ 50]	FashionMNIST	Time 0.02068s	Training Accuracy: 86.43%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 17/ 50]	       MNIST	Time 0.02081s	Training Accuracy: 99.51%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 17/ 50]	FashionMNIST	Time 0.02611s	Training Accuracy: 86.72%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 18/ 50]	       MNIST	Time 0.02004s	Training Accuracy: 99.80%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 18/ 50]	FashionMNIST	Time 0.02113s	Training Accuracy: 88.48%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 19/ 50]	       MNIST	Time 0.02119s	Training Accuracy: 99.80%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 19/ 50]	FashionMNIST	Time 0.02131s	Training Accuracy: 88.57%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 20/ 50]	       MNIST	Time 0.02227s	Training Accuracy: 99.90%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 20/ 50]	FashionMNIST	Time 0.02106s	Training Accuracy: 89.06%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 21/ 50]	       MNIST	Time 0.02123s	Training Accuracy: 99.90%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 21/ 50]	FashionMNIST	Time 0.02278s	Training Accuracy: 89.45%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 22/ 50]	       MNIST	Time 0.02143s	Training Accuracy: 99.90%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 22/ 50]	FashionMNIST	Time 0.02074s	Training Accuracy: 89.45%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 23/ 50]	       MNIST	Time 0.02552s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 23/ 50]	FashionMNIST	Time 0.02061s	Training Accuracy: 90.14%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 24/ 50]	       MNIST	Time 0.02316s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 24/ 50]	FashionMNIST	Time 0.02123s	Training Accuracy: 91.70%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 25/ 50]	       MNIST	Time 0.02055s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 25/ 50]	FashionMNIST	Time 0.02182s	Training Accuracy: 91.21%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 26/ 50]	       MNIST	Time 0.02166s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 26/ 50]	FashionMNIST	Time 0.02073s	Training Accuracy: 91.70%	Test Accuracy: 90.62%</span></span>
<span class="line"><span>[ 27/ 50]	       MNIST	Time 0.02021s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 27/ 50]	FashionMNIST	Time 0.02205s	Training Accuracy: 92.19%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 28/ 50]	       MNIST	Time 0.02247s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 28/ 50]	FashionMNIST	Time 0.02149s	Training Accuracy: 92.68%	Test Accuracy: 90.62%</span></span>
<span class="line"><span>[ 29/ 50]	       MNIST	Time 0.02136s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 29/ 50]	FashionMNIST	Time 0.02383s	Training Accuracy: 92.77%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 30/ 50]	       MNIST	Time 0.02186s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 30/ 50]	FashionMNIST	Time 0.02068s	Training Accuracy: 93.85%	Test Accuracy: 90.62%</span></span>
<span class="line"><span>[ 31/ 50]	       MNIST	Time 0.02048s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 31/ 50]	FashionMNIST	Time 0.02130s	Training Accuracy: 93.85%	Test Accuracy: 90.62%</span></span>
<span class="line"><span>[ 32/ 50]	       MNIST	Time 0.02268s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 32/ 50]	FashionMNIST	Time 0.02070s	Training Accuracy: 94.63%	Test Accuracy: 90.62%</span></span>
<span class="line"><span>[ 33/ 50]	       MNIST	Time 0.02138s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 33/ 50]	FashionMNIST	Time 0.02640s	Training Accuracy: 94.24%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 34/ 50]	       MNIST	Time 0.02085s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 34/ 50]	FashionMNIST	Time 0.02125s	Training Accuracy: 95.02%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 35/ 50]	       MNIST	Time 0.02136s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 35/ 50]	FashionMNIST	Time 0.02131s	Training Accuracy: 95.41%	Test Accuracy: 90.62%</span></span>
<span class="line"><span>[ 36/ 50]	       MNIST	Time 0.02093s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 36/ 50]	FashionMNIST	Time 0.02084s	Training Accuracy: 95.51%	Test Accuracy: 90.62%</span></span>
<span class="line"><span>[ 37/ 50]	       MNIST	Time 0.02103s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 37/ 50]	FashionMNIST	Time 0.02101s	Training Accuracy: 95.61%	Test Accuracy: 90.62%</span></span>
<span class="line"><span>[ 38/ 50]	       MNIST	Time 0.02128s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 38/ 50]	FashionMNIST	Time 0.02100s	Training Accuracy: 95.80%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 39/ 50]	       MNIST	Time 0.02025s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 39/ 50]	FashionMNIST	Time 0.02092s	Training Accuracy: 96.00%	Test Accuracy: 90.62%</span></span>
<span class="line"><span>[ 40/ 50]	       MNIST	Time 0.02272s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 40/ 50]	FashionMNIST	Time 0.02348s	Training Accuracy: 96.29%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 41/ 50]	       MNIST	Time 0.02093s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 41/ 50]	FashionMNIST	Time 0.02076s	Training Accuracy: 96.19%	Test Accuracy: 90.62%</span></span>
<span class="line"><span>[ 42/ 50]	       MNIST	Time 0.02086s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 42/ 50]	FashionMNIST	Time 0.02919s	Training Accuracy: 96.68%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 43/ 50]	       MNIST	Time 0.02177s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 43/ 50]	FashionMNIST	Time 0.02187s	Training Accuracy: 96.58%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 44/ 50]	       MNIST	Time 0.02563s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 44/ 50]	FashionMNIST	Time 0.02175s	Training Accuracy: 96.58%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 45/ 50]	       MNIST	Time 0.02170s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 45/ 50]	FashionMNIST	Time 0.02085s	Training Accuracy: 96.68%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 46/ 50]	       MNIST	Time 0.02218s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 46/ 50]	FashionMNIST	Time 0.02262s	Training Accuracy: 96.78%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 47/ 50]	       MNIST	Time 0.02142s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 47/ 50]	FashionMNIST	Time 0.02121s	Training Accuracy: 96.88%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 48/ 50]	       MNIST	Time 0.02073s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 48/ 50]	FashionMNIST	Time 0.02216s	Training Accuracy: 97.07%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 49/ 50]	       MNIST	Time 0.02142s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 49/ 50]	FashionMNIST	Time 0.02106s	Training Accuracy: 97.36%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 50/ 50]	       MNIST	Time 0.02152s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 50/ 50]	FashionMNIST	Time 0.02239s	Training Accuracy: 97.46%	Test Accuracy: 84.38%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[FINAL]	FashionMNIST	Training Accuracy: 97.46%	Test Accuracy: 84.38%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.1</span></span>
<span class="line"><span>Commit 8f5b7ca12ad (2024-10-16 10:53 UTC)</span></span>
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
<span class="line"><span>- Julia: 1.11.1</span></span>
<span class="line"><span>- LLVM: 16.0.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 3.170 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,26)]))}const y=a(t,[["render",l]]);export{d as __pageData,y as default};
