import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.bV3h_rQg.js";const E=JSON.parse('{"title":"Training a HyperNetwork on MNIST and FashionMNIST","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/3_HyperNet.md","filePath":"tutorials/intermediate/3_HyperNet.md","lastUpdated":null}'),l={name:"tutorials/intermediate/3_HyperNet.md"};function e(t,s,h,c,k,r){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" tabindex="-1">Training a HyperNetwork on MNIST and FashionMNIST <a class="header-anchor" href="#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" aria-label="Permalink to &quot;Training a HyperNetwork on MNIST and FashionMNIST {#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST}&quot;">​</a></h1><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, ADTypes, ComponentArrays, LuxCUDA, MLDatasets, MLUtils, OneHotArrays, Optimisers,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Printf, Random, Setfield, Statistics, Zygote</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">allowscalar</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    334.4 ms  ✓ Future</span></span>
<span class="line"><span>    389.8 ms  ✓ ConcreteStructs</span></span>
<span class="line"><span>    355.9 ms  ✓ CEnum</span></span>
<span class="line"><span>    554.8 ms  ✓ ADTypes</span></span>
<span class="line"><span>    382.4 ms  ✓ OpenLibm_jll</span></span>
<span class="line"><span>    496.7 ms  ✓ Statistics</span></span>
<span class="line"><span>    368.1 ms  ✓ ArgCheck</span></span>
<span class="line"><span>   1820.6 ms  ✓ UnsafeAtomics</span></span>
<span class="line"><span>    374.7 ms  ✓ ManualMemory</span></span>
<span class="line"><span>    457.2 ms  ✓ CompilerSupportLibraries_jll</span></span>
<span class="line"><span>    297.7 ms  ✓ SIMDTypes</span></span>
<span class="line"><span>    316.2 ms  ✓ Reexport</span></span>
<span class="line"><span>    304.3 ms  ✓ IfElse</span></span>
<span class="line"><span>    425.4 ms  ✓ ConstructionBase</span></span>
<span class="line"><span>    884.4 ms  ✓ IrrationalConstants</span></span>
<span class="line"><span>    436.1 ms  ✓ Adapt</span></span>
<span class="line"><span>    320.7 ms  ✓ CommonWorldInvalidations</span></span>
<span class="line"><span>    325.7 ms  ✓ FastClosures</span></span>
<span class="line"><span>    385.0 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>    404.1 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    374.2 ms  ✓ ADTypes → ADTypesEnzymeCoreExt</span></span>
<span class="line"><span>    414.7 ms  ✓ NaNMath</span></span>
<span class="line"><span>    510.9 ms  ✓ Atomix</span></span>
<span class="line"><span>    802.0 ms  ✓ ThreadingUtilities</span></span>
<span class="line"><span>    626.1 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>    368.4 ms  ✓ ConstructionBase → ConstructionBaseLinearAlgebraExt</span></span>
<span class="line"><span>   2157.5 ms  ✓ Hwloc</span></span>
<span class="line"><span>    368.6 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>    505.6 ms  ✓ ArrayInterface</span></span>
<span class="line"><span>    458.9 ms  ✓ GPUArraysCore</span></span>
<span class="line"><span>    598.9 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    381.2 ms  ✓ EnzymeCore → AdaptExt</span></span>
<span class="line"><span>    393.1 ms  ✓ DiffResults</span></span>
<span class="line"><span>    805.7 ms  ✓ Static</span></span>
<span class="line"><span>    604.9 ms  ✓ Functors</span></span>
<span class="line"><span>    394.0 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>    347.6 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>    375.5 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>   1465.1 ms  ✓ Setfield</span></span>
<span class="line"><span>   1360.3 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    415.1 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>   2630.8 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    997.6 ms  ✓ CPUSummary</span></span>
<span class="line"><span>   1121.9 ms  ✓ Optimisers</span></span>
<span class="line"><span>   1579.7 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>    475.8 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    492.4 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    831.2 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>   7655.2 ms  ✓ StaticArrays</span></span>
<span class="line"><span>    635.6 ms  ✓ DiffRules</span></span>
<span class="line"><span>   1837.8 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    422.2 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    666.0 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>    451.9 ms  ✓ Optimisers → OptimisersAdaptExt</span></span>
<span class="line"><span>    473.8 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>   2801.7 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    593.3 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    640.9 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>    520.9 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    647.3 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    612.5 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    609.8 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    602.7 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    663.6 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    964.3 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    941.7 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>    768.9 ms  ✓ Polyester</span></span>
<span class="line"><span>   3839.0 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    920.5 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   3984.8 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    668.5 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    745.7 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5345.4 ms  ✓ NNlib</span></span>
<span class="line"><span>    884.6 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    971.8 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5633.7 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9130.7 ms  ✓ Lux</span></span>
<span class="line"><span>  77 dependencies successfully precompiled in 44 seconds. 32 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArrays...</span></span>
<span class="line"><span>    878.6 ms  ✓ ComponentArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesComponentArraysExt...</span></span>
<span class="line"><span>    530.0 ms  ✓ MLDataDevices → MLDataDevicesComponentArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 49 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxComponentArraysExt...</span></span>
<span class="line"><span>    541.2 ms  ✓ ComponentArrays → ComponentArraysOptimisersExt</span></span>
<span class="line"><span>   1660.1 ms  ✓ Lux → LuxComponentArraysExt</span></span>
<span class="line"><span>   2003.5 ms  ✓ ComponentArrays → ComponentArraysKernelAbstractionsExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 2 seconds. 111 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>    388.2 ms  ✓ Zlib_jll</span></span>
<span class="line"><span>    347.6 ms  ✓ InvertedIndices</span></span>
<span class="line"><span>    648.0 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>    469.5 ms  ✓ PooledArrays</span></span>
<span class="line"><span>    982.0 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>   2438.9 ms  ✓ FixedPointNumbers</span></span>
<span class="line"><span>   1439.4 ms  ✓ ColorTypes</span></span>
<span class="line"><span>   6741.1 ms  ✓ LLVM</span></span>
<span class="line"><span>   4864.7 ms  ✓ Colors</span></span>
<span class="line"><span>   1867.7 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>   1466.7 ms  ✓ LLVM → BFloat16sExt</span></span>
<span class="line"><span>   2188.4 ms  ✓ GPUArrays</span></span>
<span class="line"><span>   1300.3 ms  ✓ NVTX</span></span>
<span class="line"><span>  20827.7 ms  ✓ PrettyTables</span></span>
<span class="line"><span>  27074.5 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>  46537.8 ms  ✓ DataFrames</span></span>
<span class="line"><span>  51448.3 ms  ✓ CUDA</span></span>
<span class="line"><span>   5218.6 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8256.1 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5511.1 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  20 dependencies successfully precompiled in 139 seconds. 80 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesGPUArraysExt...</span></span>
<span class="line"><span>   1318.3 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 42 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersGPUArraysExt...</span></span>
<span class="line"><span>   1385.6 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArraysGPUArraysExt...</span></span>
<span class="line"><span>   1562.2 ms  ✓ ComponentArrays → ComponentArraysGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 68 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceSparseArraysExt...</span></span>
<span class="line"><span>    601.3 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 7 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    654.1 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 17 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   4853.7 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   5010.6 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5508.5 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   4914.2 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5178.8 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   5324.9 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5755.5 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 6 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   5111.2 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5552.4 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5155.8 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 107 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   5799.2 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 174 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDatasets...</span></span>
<span class="line"><span>    363.8 ms  ✓ Glob</span></span>
<span class="line"><span>    408.3 ms  ✓ WorkerUtilities</span></span>
<span class="line"><span>    758.7 ms  ✓ InitialValues</span></span>
<span class="line"><span>    359.4 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    444.0 ms  ✓ BufferedStreams</span></span>
<span class="line"><span>    339.1 ms  ✓ SimpleBufferStream</span></span>
<span class="line"><span>    412.9 ms  ✓ ShowCases</span></span>
<span class="line"><span>    560.2 ms  ✓ URIs</span></span>
<span class="line"><span>    324.0 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    525.0 ms  ✓ TranscodingStreams</span></span>
<span class="line"><span>    433.1 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    315.3 ms  ✓ PackageExtensionCompat</span></span>
<span class="line"><span>    346.4 ms  ✓ BitFlags</span></span>
<span class="line"><span>    437.3 ms  ✓ OffsetArrays → OffsetArraysAdaptExt</span></span>
<span class="line"><span>    690.3 ms  ✓ GZip</span></span>
<span class="line"><span>   1054.1 ms  ✓ Baselet</span></span>
<span class="line"><span>    595.6 ms  ✓ ZipFile</span></span>
<span class="line"><span>    807.3 ms  ✓ StructTypes</span></span>
<span class="line"><span>   1044.9 ms  ✓ MbedTLS</span></span>
<span class="line"><span>    561.6 ms  ✓ Unitful → ConstructionBaseUnitfulExt</span></span>
<span class="line"><span>   2060.2 ms  ✓ ColorVectorSpace</span></span>
<span class="line"><span>   2335.6 ms  ✓ PeriodicTable</span></span>
<span class="line"><span>   2679.9 ms  ✓ UnitfulAtomic</span></span>
<span class="line"><span>    572.7 ms  ✓ MPIPreferences</span></span>
<span class="line"><span>    336.3 ms  ✓ InternedStrings</span></span>
<span class="line"><span>    518.6 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>   1125.7 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>   2863.5 ms  ✓ Accessors</span></span>
<span class="line"><span>    563.1 ms  ✓ Chemfiles_jll</span></span>
<span class="line"><span>    593.1 ms  ✓ libaec_jll</span></span>
<span class="line"><span>    487.0 ms  ✓ MicrosoftMPI_jll</span></span>
<span class="line"><span>    452.6 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    548.4 ms  ✓ StringEncodings</span></span>
<span class="line"><span>    796.7 ms  ✓ WeakRefStrings</span></span>
<span class="line"><span>    506.8 ms  ✓ FilePathsBase → FilePathsBaseMmapExt</span></span>
<span class="line"><span>    378.0 ms  ✓ NameResolution</span></span>
<span class="line"><span>    434.8 ms  ✓ CodecZlib</span></span>
<span class="line"><span>    414.1 ms  ✓ StridedViews</span></span>
<span class="line"><span>   2241.5 ms  ✓ StatsBase</span></span>
<span class="line"><span>   1838.6 ms  ✓ OpenSSL</span></span>
<span class="line"><span>   1525.0 ms  ✓ NPZ</span></span>
<span class="line"><span>  10969.2 ms  ✓ JSON3</span></span>
<span class="line"><span>   3402.1 ms  ✓ ColorSchemes</span></span>
<span class="line"><span>   2265.6 ms  ✓ AtomsBase</span></span>
<span class="line"><span>   1431.8 ms  ✓ MPICH_jll</span></span>
<span class="line"><span>   1157.9 ms  ✓ MPItrampoline_jll</span></span>
<span class="line"><span>  19418.6 ms  ✓ ImageCore</span></span>
<span class="line"><span>   1161.0 ms  ✓ OpenMPI_jll</span></span>
<span class="line"><span>    649.5 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>    674.5 ms  ✓ Accessors → AccessorsUnitfulExt</span></span>
<span class="line"><span>    831.9 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>    708.7 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>    763.5 ms  ✓ BangBang</span></span>
<span class="line"><span>   4518.9 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   2396.0 ms  ✓ Pickle</span></span>
<span class="line"><span>  34275.4 ms  ✓ JLD2</span></span>
<span class="line"><span>   2336.4 ms  ✓ Chemfiles</span></span>
<span class="line"><span>   2219.3 ms  ✓ ImageBase</span></span>
<span class="line"><span>   1971.9 ms  ✓ HDF5_jll</span></span>
<span class="line"><span>    719.0 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>   1822.1 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>    502.3 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>  19711.3 ms  ✓ CSV</span></span>
<span class="line"><span>    510.9 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    835.3 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2042.1 ms  ✓ ImageShow</span></span>
<span class="line"><span>   2855.4 ms  ✓ Transducers</span></span>
<span class="line"><span>   1504.5 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>  19593.6 ms  ✓ HTTP</span></span>
<span class="line"><span>    633.2 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   7857.4 ms  ✓ HDF5</span></span>
<span class="line"><span>   1869.7 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>   3331.8 ms  ✓ DataDeps</span></span>
<span class="line"><span>   5416.0 ms  ✓ FLoops</span></span>
<span class="line"><span>   2370.4 ms  ✓ MAT</span></span>
<span class="line"><span>   6677.9 ms  ✓ MLUtils</span></span>
<span class="line"><span>   9341.9 ms  ✓ MLDatasets</span></span>
<span class="line"><span>  77 dependencies successfully precompiled in 82 seconds. 121 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1628.8 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2182.6 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling OneHotArrays...</span></span>
<span class="line"><span>    970.4 ms  ✓ OneHotArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 28 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesOneHotArraysExt...</span></span>
<span class="line"><span>    749.1 ms  ✓ MLDataDevices → MLDataDevicesOneHotArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 35 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    408.5 ms  ✓ FillArrays → FillArraysStatisticsExt</span></span>
<span class="line"><span>    760.2 ms  ✓ StructArrays</span></span>
<span class="line"><span>    409.9 ms  ✓ StructArrays → StructArraysAdaptExt</span></span>
<span class="line"><span>    422.8 ms  ✓ StructArrays → StructArraysGPUArraysCoreExt</span></span>
<span class="line"><span>    664.9 ms  ✓ StructArrays → StructArraysSparseArraysExt</span></span>
<span class="line"><span>   5422.9 ms  ✓ ChainRules</span></span>
<span class="line"><span>  33229.0 ms  ✓ Zygote</span></span>
<span class="line"><span>  7 dependencies successfully precompiled in 40 seconds. 79 already precompiled.</span></span>
<span class="line"><span>Precompiling AccessorsStructArraysExt...</span></span>
<span class="line"><span>    464.3 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 16 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    491.8 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 22 already precompiled.</span></span>
<span class="line"><span>Precompiling StructArraysStaticArraysExt...</span></span>
<span class="line"><span>    650.9 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    780.9 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 39 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    822.9 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    460.2 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1633.5 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   2787.2 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 163 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArraysZygoteExt...</span></span>
<span class="line"><span>   1594.2 ms  ✓ ComponentArrays → ComponentArraysZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 99 already precompiled.</span></span>
<span class="line"><span>Precompiling ZygoteColorsExt...</span></span>
<span class="line"><span>   1775.9 ms  ✓ Zygote → ZygoteColorsExt</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">test_acc_list </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[  1/ 50]	       MNIST	Time 87.25041s	Training Accuracy: 56.54%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[  1/ 50]	FashionMNIST	Time 0.01223s	Training Accuracy: 48.05%	Test Accuracy: 46.88%</span></span>
<span class="line"><span>[  2/ 50]	       MNIST	Time 0.01235s	Training Accuracy: 69.14%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[  2/ 50]	FashionMNIST	Time 0.01225s	Training Accuracy: 62.01%	Test Accuracy: 53.12%</span></span>
<span class="line"><span>[  3/ 50]	       MNIST	Time 0.01229s	Training Accuracy: 77.25%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[  3/ 50]	FashionMNIST	Time 0.01222s	Training Accuracy: 63.96%	Test Accuracy: 56.25%</span></span>
<span class="line"><span>[  4/ 50]	       MNIST	Time 0.01070s	Training Accuracy: 76.56%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[  4/ 50]	FashionMNIST	Time 0.01193s	Training Accuracy: 64.84%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[  5/ 50]	       MNIST	Time 0.01117s	Training Accuracy: 80.86%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[  5/ 50]	FashionMNIST	Time 0.01363s	Training Accuracy: 67.77%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  6/ 50]	       MNIST	Time 0.01247s	Training Accuracy: 86.62%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  6/ 50]	FashionMNIST	Time 0.01048s	Training Accuracy: 71.29%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[  7/ 50]	       MNIST	Time 0.01114s	Training Accuracy: 86.91%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[  7/ 50]	FashionMNIST	Time 0.01155s	Training Accuracy: 69.92%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[  8/ 50]	       MNIST	Time 0.01164s	Training Accuracy: 91.21%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[  8/ 50]	FashionMNIST	Time 0.01060s	Training Accuracy: 69.53%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  9/ 50]	       MNIST	Time 0.01181s	Training Accuracy: 93.46%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[  9/ 50]	FashionMNIST	Time 0.01047s	Training Accuracy: 69.04%	Test Accuracy: 53.12%</span></span>
<span class="line"><span>[ 10/ 50]	       MNIST	Time 0.01540s	Training Accuracy: 94.92%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 10/ 50]	FashionMNIST	Time 0.01049s	Training Accuracy: 72.07%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 11/ 50]	       MNIST	Time 0.01177s	Training Accuracy: 97.07%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 11/ 50]	FashionMNIST	Time 0.01040s	Training Accuracy: 75.59%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 12/ 50]	       MNIST	Time 0.01642s	Training Accuracy: 97.95%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 12/ 50]	FashionMNIST	Time 0.01065s	Training Accuracy: 75.49%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 13/ 50]	       MNIST	Time 0.01052s	Training Accuracy: 98.93%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 13/ 50]	FashionMNIST	Time 0.01057s	Training Accuracy: 76.66%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 14/ 50]	       MNIST	Time 0.01681s	Training Accuracy: 99.22%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 14/ 50]	FashionMNIST	Time 0.01005s	Training Accuracy: 78.61%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 15/ 50]	       MNIST	Time 0.01124s	Training Accuracy: 99.71%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 15/ 50]	FashionMNIST	Time 0.01555s	Training Accuracy: 78.71%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 16/ 50]	       MNIST	Time 0.01153s	Training Accuracy: 99.80%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 16/ 50]	FashionMNIST	Time 0.01208s	Training Accuracy: 80.08%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 17/ 50]	       MNIST	Time 0.01060s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 17/ 50]	FashionMNIST	Time 0.01210s	Training Accuracy: 81.05%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 18/ 50]	       MNIST	Time 0.01098s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 18/ 50]	FashionMNIST	Time 0.01059s	Training Accuracy: 82.13%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 19/ 50]	       MNIST	Time 0.01176s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 19/ 50]	FashionMNIST	Time 0.01718s	Training Accuracy: 82.81%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 20/ 50]	       MNIST	Time 0.01098s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 20/ 50]	FashionMNIST	Time 0.01180s	Training Accuracy: 83.69%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 21/ 50]	       MNIST	Time 0.01544s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 21/ 50]	FashionMNIST	Time 0.01121s	Training Accuracy: 83.89%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 22/ 50]	       MNIST	Time 0.01173s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 22/ 50]	FashionMNIST	Time 0.01044s	Training Accuracy: 84.96%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 23/ 50]	       MNIST	Time 0.01125s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 23/ 50]	FashionMNIST	Time 0.01073s	Training Accuracy: 85.25%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 24/ 50]	       MNIST	Time 0.01126s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 24/ 50]	FashionMNIST	Time 0.01175s	Training Accuracy: 85.74%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 25/ 50]	       MNIST	Time 0.01826s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 25/ 50]	FashionMNIST	Time 0.01103s	Training Accuracy: 86.13%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 26/ 50]	       MNIST	Time 0.01150s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 26/ 50]	FashionMNIST	Time 0.01517s	Training Accuracy: 86.72%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 27/ 50]	       MNIST	Time 0.01175s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 27/ 50]	FashionMNIST	Time 0.01211s	Training Accuracy: 87.01%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 28/ 50]	       MNIST	Time 0.01111s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 28/ 50]	FashionMNIST	Time 0.01145s	Training Accuracy: 87.50%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 29/ 50]	       MNIST	Time 0.01050s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 29/ 50]	FashionMNIST	Time 0.01201s	Training Accuracy: 87.30%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 30/ 50]	       MNIST	Time 0.01151s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 30/ 50]	FashionMNIST	Time 0.01796s	Training Accuracy: 88.18%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 31/ 50]	       MNIST	Time 0.01143s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 31/ 50]	FashionMNIST	Time 0.01111s	Training Accuracy: 88.28%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 32/ 50]	       MNIST	Time 0.01192s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 32/ 50]	FashionMNIST	Time 0.01064s	Training Accuracy: 88.48%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 33/ 50]	       MNIST	Time 0.01172s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 33/ 50]	FashionMNIST	Time 0.01185s	Training Accuracy: 88.48%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 34/ 50]	       MNIST	Time 0.01047s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 34/ 50]	FashionMNIST	Time 0.01131s	Training Accuracy: 89.06%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 35/ 50]	       MNIST	Time 0.01534s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 35/ 50]	FashionMNIST	Time 0.01130s	Training Accuracy: 90.23%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 36/ 50]	       MNIST	Time 0.01176s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 36/ 50]	FashionMNIST	Time 0.01611s	Training Accuracy: 90.72%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 37/ 50]	       MNIST	Time 0.01191s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 37/ 50]	FashionMNIST	Time 0.01169s	Training Accuracy: 90.72%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 38/ 50]	       MNIST	Time 0.01005s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 38/ 50]	FashionMNIST	Time 0.01170s	Training Accuracy: 89.94%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 39/ 50]	       MNIST	Time 0.01084s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 39/ 50]	FashionMNIST	Time 0.01055s	Training Accuracy: 90.23%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 40/ 50]	       MNIST	Time 0.01177s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 40/ 50]	FashionMNIST	Time 0.01578s	Training Accuracy: 89.75%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 41/ 50]	       MNIST	Time 0.01126s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 41/ 50]	FashionMNIST	Time 0.01176s	Training Accuracy: 91.02%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 42/ 50]	       MNIST	Time 0.01648s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 42/ 50]	FashionMNIST	Time 0.01127s	Training Accuracy: 91.80%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 43/ 50]	       MNIST	Time 0.01169s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 43/ 50]	FashionMNIST	Time 0.01084s	Training Accuracy: 91.50%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 44/ 50]	       MNIST	Time 0.01126s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 44/ 50]	FashionMNIST	Time 0.01170s	Training Accuracy: 91.31%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 45/ 50]	       MNIST	Time 0.01074s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 45/ 50]	FashionMNIST	Time 0.01047s	Training Accuracy: 92.38%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 46/ 50]	       MNIST	Time 0.01490s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 46/ 50]	FashionMNIST	Time 0.01086s	Training Accuracy: 93.46%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 47/ 50]	       MNIST	Time 0.01123s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 47/ 50]	FashionMNIST	Time 0.01625s	Training Accuracy: 93.16%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 48/ 50]	       MNIST	Time 0.01082s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 48/ 50]	FashionMNIST	Time 0.01151s	Training Accuracy: 93.07%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 49/ 50]	       MNIST	Time 0.01049s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 49/ 50]	FashionMNIST	Time 0.01085s	Training Accuracy: 94.04%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 50/ 50]	       MNIST	Time 0.01119s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 50/ 50]	FashionMNIST	Time 0.01060s	Training Accuracy: 94.14%	Test Accuracy: 65.62%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[FINAL]	FashionMNIST	Training Accuracy: 94.14%	Test Accuracy: 65.62%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  0: Quadro RTX 5000 (sm_75, 13.881 GiB / 16.000 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,26)]))}const y=a(l,[["render",e]]);export{E as __pageData,y as default};
