import{_ as a,c as i,a2 as n,o as p}from"./chunks/framework.C24IT_go.js";const E=JSON.parse('{"title":"Training a HyperNetwork on MNIST and FashionMNIST","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/3_HyperNet.md","filePath":"tutorials/intermediate/3_HyperNet.md","lastUpdated":null}'),t={name:"tutorials/intermediate/3_HyperNet.md"};function l(e,s,h,k,c,r){return p(),i("div",null,s[0]||(s[0]=[n(`<h1 id="Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" tabindex="-1">Training a HyperNetwork on MNIST and FashionMNIST <a class="header-anchor" href="#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" aria-label="Permalink to &quot;Training a HyperNetwork on MNIST and FashionMNIST {#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST}&quot;">​</a></h1><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, ComponentArrays, LuxCUDA, MLDatasets, MLUtils, OneHotArrays, Optimisers,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Printf, Random, Zygote</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">allowscalar</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling LuxComponentArraysExt...</span></span>
<span class="line"><span>   1540.4 ms  ✓ Lux → LuxComponentArraysExt</span></span>
<span class="line"><span>   2152.8 ms  ✓ ComponentArrays → ComponentArraysKernelAbstractionsExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 2 seconds. 112 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>  26103.8 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>  51357.4 ms  ✓ CUDA</span></span>
<span class="line"><span>   5182.4 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8159.0 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5259.5 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  5 dependencies successfully precompiled in 97 seconds. 95 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersGPUArraysExt...</span></span>
<span class="line"><span>   1407.8 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 46 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   4922.4 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 101 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   4973.8 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5368.8 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   5034.3 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5212.8 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   5234.5 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5683.4 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 6 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   5051.9 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5331.8 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5064.3 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 107 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   5799.5 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 174 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDatasets...</span></span>
<span class="line"><span>    359.5 ms  ✓ Glob</span></span>
<span class="line"><span>    381.9 ms  ✓ WorkerUtilities</span></span>
<span class="line"><span>    420.8 ms  ✓ BufferedStreams</span></span>
<span class="line"><span>    315.5 ms  ✓ SimpleBufferStream</span></span>
<span class="line"><span>    435.1 ms  ✓ CodecZlib</span></span>
<span class="line"><span>    560.7 ms  ✓ URIs</span></span>
<span class="line"><span>    425.1 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    308.7 ms  ✓ PackageExtensionCompat</span></span>
<span class="line"><span>    340.6 ms  ✓ BitFlags</span></span>
<span class="line"><span>    570.6 ms  ✓ ZipFile</span></span>
<span class="line"><span>    663.8 ms  ✓ GZip</span></span>
<span class="line"><span>    659.9 ms  ✓ ConcurrentUtilities</span></span>
<span class="line"><span>    491.0 ms  ✓ LoggingExtras</span></span>
<span class="line"><span>    378.3 ms  ✓ ContextVariablesX</span></span>
<span class="line"><span>    318.7 ms  ✓ InternedStrings</span></span>
<span class="line"><span>    470.1 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>   2069.5 ms  ✓ ColorVectorSpace</span></span>
<span class="line"><span>   1084.2 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>   2874.3 ms  ✓ Accessors</span></span>
<span class="line"><span>   1373.7 ms  ✓ MPICH_jll</span></span>
<span class="line"><span>    758.5 ms  ✓ WeakRefStrings</span></span>
<span class="line"><span>   1120.0 ms  ✓ MPItrampoline_jll</span></span>
<span class="line"><span>   2206.8 ms  ✓ AtomsBase</span></span>
<span class="line"><span>   1096.9 ms  ✓ OpenMPI_jll</span></span>
<span class="line"><span>    400.1 ms  ✓ StridedViews</span></span>
<span class="line"><span>   1825.4 ms  ✓ OpenSSL</span></span>
<span class="line"><span>   1529.2 ms  ✓ NPZ</span></span>
<span class="line"><span>    582.7 ms  ✓ FLoopsBase</span></span>
<span class="line"><span>  11077.4 ms  ✓ JSON3</span></span>
<span class="line"><span>   3450.4 ms  ✓ ColorSchemes</span></span>
<span class="line"><span>    605.0 ms  ✓ Accessors → AccessorsTestExt</span></span>
<span class="line"><span>    759.1 ms  ✓ Accessors → AccessorsDatesExt</span></span>
<span class="line"><span>    634.0 ms  ✓ Accessors → AccessorsUnitfulExt</span></span>
<span class="line"><span>    741.6 ms  ✓ BangBang</span></span>
<span class="line"><span>    683.5 ms  ✓ Accessors → AccessorsStaticArraysExt</span></span>
<span class="line"><span>  18510.4 ms  ✓ ImageCore</span></span>
<span class="line"><span>   2226.2 ms  ✓ Chemfiles</span></span>
<span class="line"><span>   1871.0 ms  ✓ HDF5_jll</span></span>
<span class="line"><span>   2346.6 ms  ✓ Pickle</span></span>
<span class="line"><span>  33069.8 ms  ✓ JLD2</span></span>
<span class="line"><span>    694.3 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>  18725.0 ms  ✓ CSV</span></span>
<span class="line"><span>   1629.3 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>    495.8 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    477.0 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    832.1 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   2071.7 ms  ✓ ImageBase</span></span>
<span class="line"><span>   2589.0 ms  ✓ Transducers</span></span>
<span class="line"><span>   1853.5 ms  ✓ ImageShow</span></span>
<span class="line"><span>   7107.9 ms  ✓ HDF5</span></span>
<span class="line"><span>   1397.9 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>    633.4 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   2301.0 ms  ✓ MAT</span></span>
<span class="line"><span>   4950.5 ms  ✓ FLoops</span></span>
<span class="line"><span>  19188.9 ms  ✓ HTTP</span></span>
<span class="line"><span>   1819.2 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>   3050.2 ms  ✓ DataDeps</span></span>
<span class="line"><span>   6202.1 ms  ✓ MLUtils</span></span>
<span class="line"><span>   8856.3 ms  ✓ MLDatasets</span></span>
<span class="line"><span>  59 dependencies successfully precompiled in 71 seconds. 139 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1592.5 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 102 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2167.6 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 167 already precompiled.</span></span>
<span class="line"><span>Precompiling OneHotArrays...</span></span>
<span class="line"><span>    934.1 ms  ✓ OneHotArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 28 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesOneHotArraysExt...</span></span>
<span class="line"><span>    727.2 ms  ✓ MLDataDevices → MLDataDevicesOneHotArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 35 already precompiled.</span></span>
<span class="line"><span>Precompiling Zygote...</span></span>
<span class="line"><span>    324.8 ms  ✓ RealDot</span></span>
<span class="line"><span>   5301.8 ms  ✓ ChainRules</span></span>
<span class="line"><span>  32822.6 ms  ✓ Zygote</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 39 seconds. 83 already precompiled.</span></span>
<span class="line"><span>Precompiling AccessorsStructArraysExt...</span></span>
<span class="line"><span>    473.3 ms  ✓ Accessors → AccessorsStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 16 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    481.8 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 22 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceChainRulesExt...</span></span>
<span class="line"><span>    864.4 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 39 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    833.2 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 40 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1604.0 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 93 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   2761.2 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 163 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArraysZygoteExt...</span></span>
<span class="line"><span>   1620.3 ms  ✓ ComponentArrays → ComponentArraysZygoteExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 99 already precompiled.</span></span>
<span class="line"><span>Precompiling ZygoteColorsExt...</span></span>
<span class="line"><span>   1776.5 ms  ✓ Zygote → ZygoteColorsExt</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">test_acc_list </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[  1/ 50]	       MNIST	Time 88.27585s	Training Accuracy: 58.01%	Test Accuracy: 46.88%</span></span>
<span class="line"><span>[  1/ 50]	FashionMNIST	Time 0.02446s	Training Accuracy: 52.15%	Test Accuracy: 46.88%</span></span>
<span class="line"><span>[  2/ 50]	       MNIST	Time 0.02460s	Training Accuracy: 66.99%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[  2/ 50]	FashionMNIST	Time 0.02514s	Training Accuracy: 64.94%	Test Accuracy: 50.00%</span></span>
<span class="line"><span>[  3/ 50]	       MNIST	Time 0.02564s	Training Accuracy: 74.51%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[  3/ 50]	FashionMNIST	Time 0.02598s	Training Accuracy: 64.75%	Test Accuracy: 53.12%</span></span>
<span class="line"><span>[  4/ 50]	       MNIST	Time 0.05349s	Training Accuracy: 74.41%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[  4/ 50]	FashionMNIST	Time 0.02134s	Training Accuracy: 63.38%	Test Accuracy: 56.25%</span></span>
<span class="line"><span>[  5/ 50]	       MNIST	Time 0.02132s	Training Accuracy: 77.54%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[  5/ 50]	FashionMNIST	Time 0.02118s	Training Accuracy: 67.87%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[  6/ 50]	       MNIST	Time 0.02120s	Training Accuracy: 84.47%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[  6/ 50]	FashionMNIST	Time 0.03953s	Training Accuracy: 70.61%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[  7/ 50]	       MNIST	Time 0.02204s	Training Accuracy: 87.30%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[  7/ 50]	FashionMNIST	Time 0.02045s	Training Accuracy: 66.41%	Test Accuracy: 46.88%</span></span>
<span class="line"><span>[  8/ 50]	       MNIST	Time 0.02065s	Training Accuracy: 91.11%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[  8/ 50]	FashionMNIST	Time 0.02127s	Training Accuracy: 75.49%	Test Accuracy: 53.12%</span></span>
<span class="line"><span>[  9/ 50]	       MNIST	Time 0.02198s	Training Accuracy: 93.36%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[  9/ 50]	FashionMNIST	Time 0.02122s	Training Accuracy: 75.88%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 10/ 50]	       MNIST	Time 0.02040s	Training Accuracy: 94.82%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 10/ 50]	FashionMNIST	Time 0.02252s	Training Accuracy: 76.66%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 11/ 50]	       MNIST	Time 0.02274s	Training Accuracy: 96.78%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 11/ 50]	FashionMNIST	Time 0.02050s	Training Accuracy: 75.20%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 12/ 50]	       MNIST	Time 0.02255s	Training Accuracy: 97.85%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 12/ 50]	FashionMNIST	Time 0.02147s	Training Accuracy: 78.32%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 13/ 50]	       MNIST	Time 0.02286s	Training Accuracy: 98.44%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 13/ 50]	FashionMNIST	Time 0.02389s	Training Accuracy: 80.37%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 14/ 50]	       MNIST	Time 0.02165s	Training Accuracy: 98.34%	Test Accuracy: 87.50%</span></span>
<span class="line"><span>[ 14/ 50]	FashionMNIST	Time 0.02102s	Training Accuracy: 79.98%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 15/ 50]	       MNIST	Time 0.02389s	Training Accuracy: 98.93%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 15/ 50]	FashionMNIST	Time 0.02076s	Training Accuracy: 80.86%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 16/ 50]	       MNIST	Time 0.02198s	Training Accuracy: 99.22%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 16/ 50]	FashionMNIST	Time 0.02283s	Training Accuracy: 82.91%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 17/ 50]	       MNIST	Time 0.02087s	Training Accuracy: 99.41%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 17/ 50]	FashionMNIST	Time 0.02075s	Training Accuracy: 84.57%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 18/ 50]	       MNIST	Time 0.02213s	Training Accuracy: 99.71%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 18/ 50]	FashionMNIST	Time 0.02183s	Training Accuracy: 85.25%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 19/ 50]	       MNIST	Time 0.02074s	Training Accuracy: 99.80%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 19/ 50]	FashionMNIST	Time 0.02277s	Training Accuracy: 85.84%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 20/ 50]	       MNIST	Time 0.02216s	Training Accuracy: 99.80%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 20/ 50]	FashionMNIST	Time 0.02070s	Training Accuracy: 86.91%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 21/ 50]	       MNIST	Time 0.02558s	Training Accuracy: 99.90%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 21/ 50]	FashionMNIST	Time 0.02074s	Training Accuracy: 87.30%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 22/ 50]	       MNIST	Time 0.02022s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 22/ 50]	FashionMNIST	Time 0.02249s	Training Accuracy: 87.99%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 23/ 50]	       MNIST	Time 0.02069s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 23/ 50]	FashionMNIST	Time 0.02081s	Training Accuracy: 87.60%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 24/ 50]	       MNIST	Time 0.02344s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 24/ 50]	FashionMNIST	Time 0.02160s	Training Accuracy: 88.67%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 25/ 50]	       MNIST	Time 0.02121s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 25/ 50]	FashionMNIST	Time 0.02252s	Training Accuracy: 88.96%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 26/ 50]	       MNIST	Time 0.02087s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 26/ 50]	FashionMNIST	Time 0.02066s	Training Accuracy: 88.96%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 27/ 50]	       MNIST	Time 0.02329s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 27/ 50]	FashionMNIST	Time 0.02078s	Training Accuracy: 89.84%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 28/ 50]	       MNIST	Time 0.02276s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 28/ 50]	FashionMNIST	Time 0.02285s	Training Accuracy: 89.75%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 29/ 50]	       MNIST	Time 0.02087s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 29/ 50]	FashionMNIST	Time 0.02064s	Training Accuracy: 89.65%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 30/ 50]	       MNIST	Time 0.02719s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 30/ 50]	FashionMNIST	Time 0.02065s	Training Accuracy: 88.87%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 31/ 50]	       MNIST	Time 0.02154s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 31/ 50]	FashionMNIST	Time 0.02290s	Training Accuracy: 89.36%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 32/ 50]	       MNIST	Time 0.02069s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 32/ 50]	FashionMNIST	Time 0.02048s	Training Accuracy: 89.65%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 33/ 50]	       MNIST	Time 0.02328s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 33/ 50]	FashionMNIST	Time 0.02044s	Training Accuracy: 90.23%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 34/ 50]	       MNIST	Time 0.02069s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 34/ 50]	FashionMNIST	Time 0.02398s	Training Accuracy: 90.72%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 35/ 50]	       MNIST	Time 0.02258s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 35/ 50]	FashionMNIST	Time 0.02103s	Training Accuracy: 91.02%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 36/ 50]	       MNIST	Time 0.02304s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 36/ 50]	FashionMNIST	Time 0.02057s	Training Accuracy: 90.43%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 37/ 50]	       MNIST	Time 0.02040s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 37/ 50]	FashionMNIST	Time 0.02203s	Training Accuracy: 91.50%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 38/ 50]	       MNIST	Time 0.02213s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 38/ 50]	FashionMNIST	Time 0.02045s	Training Accuracy: 91.41%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 39/ 50]	       MNIST	Time 0.02225s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 39/ 50]	FashionMNIST	Time 0.02042s	Training Accuracy: 91.89%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 40/ 50]	       MNIST	Time 0.02254s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 40/ 50]	FashionMNIST	Time 0.02756s	Training Accuracy: 92.19%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 41/ 50]	       MNIST	Time 0.02421s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 41/ 50]	FashionMNIST	Time 0.02099s	Training Accuracy: 92.38%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 42/ 50]	       MNIST	Time 0.02322s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 42/ 50]	FashionMNIST	Time 0.02150s	Training Accuracy: 93.26%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 43/ 50]	       MNIST	Time 0.02165s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 43/ 50]	FashionMNIST	Time 0.02323s	Training Accuracy: 92.48%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 44/ 50]	       MNIST	Time 0.02272s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 44/ 50]	FashionMNIST	Time 0.02271s	Training Accuracy: 93.07%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 45/ 50]	       MNIST	Time 0.02384s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 45/ 50]	FashionMNIST	Time 0.02160s	Training Accuracy: 92.19%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 46/ 50]	       MNIST	Time 0.02293s	Training Accuracy: 100.00%	Test Accuracy: 81.25%</span></span>
<span class="line"><span>[ 46/ 50]	FashionMNIST	Time 0.02301s	Training Accuracy: 88.77%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 47/ 50]	       MNIST	Time 0.02104s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 47/ 50]	FashionMNIST	Time 0.02125s	Training Accuracy: 90.53%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 48/ 50]	       MNIST	Time 0.02407s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 48/ 50]	FashionMNIST	Time 0.02186s	Training Accuracy: 90.14%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 49/ 50]	       MNIST	Time 0.02145s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 49/ 50]	FashionMNIST	Time 0.02530s	Training Accuracy: 90.82%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 50/ 50]	       MNIST	Time 0.02194s	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[ 50/ 50]	FashionMNIST	Time 0.02140s	Training Accuracy: 92.09%	Test Accuracy: 75.00%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 84.38%</span></span>
<span class="line"><span>[FINAL]	FashionMNIST	Training Accuracy: 92.09%	Test Accuracy: 75.00%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 3.170 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,26)]))}const y=a(t,[["render",l]]);export{E as __pageData,y as default};
