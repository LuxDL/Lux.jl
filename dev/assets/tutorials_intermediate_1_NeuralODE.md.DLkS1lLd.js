import{_ as a,c as n,o as i,al as e}from"./chunks/framework.BCN3FD2k.js";const o=JSON.parse('{"title":"MNIST Classification using Neural ODEs","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/1_NeuralODE.md","filePath":"tutorials/intermediate/1_NeuralODE.md","lastUpdated":null}'),p={name:"tutorials/intermediate/1_NeuralODE.md"};function l(t,s,r,h,k,c){return i(),n("div",null,s[0]||(s[0]=[e(`<h1 id="MNIST-Classification-using-Neural-ODEs" tabindex="-1">MNIST Classification using Neural ODEs <a class="header-anchor" href="#MNIST-Classification-using-Neural-ODEs" aria-label="Permalink to &quot;MNIST Classification using Neural ODEs {#MNIST-Classification-using-Neural-ODEs}&quot;">​</a></h1><p>To understand Neural ODEs, users should look up <a href="https://book.sciml.ai/notes/11-Differentiable_Programming_and_Neural_Differential_Equations/" target="_blank" rel="noreferrer">these lecture notes</a>. We recommend users to directly use <a href="https://docs.sciml.ai/DiffEqFlux/stable/" target="_blank" rel="noreferrer">DiffEqFlux.jl</a>, instead of implementing Neural ODEs from scratch.</p><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ComponentArrays,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    SciMLSensitivity,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    LuxCUDA,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Optimisers,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    OrdinaryDiffEqTsit5,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Statistics,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Zygote,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    OneHotArrays,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    InteractiveUtils,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Printf</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDatasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MNIST</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> DataLoader, splitobs</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">allowscalar</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    354.4 ms  ✓ Future</span></span>
<span class="line"><span>    406.6 ms  ✓ ConcreteStructs</span></span>
<span class="line"><span>    368.6 ms  ✓ OpenLibm_jll</span></span>
<span class="line"><span>    586.7 ms  ✓ ADTypes</span></span>
<span class="line"><span>    513.0 ms  ✓ Statistics</span></span>
<span class="line"><span>    382.5 ms  ✓ ArgCheck</span></span>
<span class="line"><span>    464.2 ms  ✓ CompilerSupportLibraries_jll</span></span>
<span class="line"><span>   1752.3 ms  ✓ UnsafeAtomics</span></span>
<span class="line"><span>    329.5 ms  ✓ Reexport</span></span>
<span class="line"><span>    381.4 ms  ✓ ManualMemory</span></span>
<span class="line"><span>    325.3 ms  ✓ SIMDTypes</span></span>
<span class="line"><span>    557.0 ms  ✓ EnzymeCore</span></span>
<span class="line"><span>   1069.7 ms  ✓ IrrationalConstants</span></span>
<span class="line"><span>    312.9 ms  ✓ IfElse</span></span>
<span class="line"><span>    433.2 ms  ✓ ConstructionBase</span></span>
<span class="line"><span>    336.5 ms  ✓ CommonWorldInvalidations</span></span>
<span class="line"><span>   2318.3 ms  ✓ MacroTools</span></span>
<span class="line"><span>    342.0 ms  ✓ FastClosures</span></span>
<span class="line"><span>    397.7 ms  ✓ StaticArraysCore</span></span>
<span class="line"><span>    519.2 ms  ✓ ArrayInterface</span></span>
<span class="line"><span>    449.5 ms  ✓ NaNMath</span></span>
<span class="line"><span>    438.0 ms  ✓ ADTypes → ADTypesChainRulesCoreExt</span></span>
<span class="line"><span>    642.5 ms  ✓ OpenSpecFun_jll</span></span>
<span class="line"><span>    507.9 ms  ✓ Atomix</span></span>
<span class="line"><span>    389.5 ms  ✓ EnzymeCore → AdaptExt</span></span>
<span class="line"><span>    810.7 ms  ✓ ThreadingUtilities</span></span>
<span class="line"><span>    397.0 ms  ✓ ADTypes → ADTypesEnzymeCoreExt</span></span>
<span class="line"><span>    377.7 ms  ✓ ConstructionBase → ConstructionBaseLinearAlgebraExt</span></span>
<span class="line"><span>    400.1 ms  ✓ ADTypes → ADTypesConstructionBaseExt</span></span>
<span class="line"><span>    636.8 ms  ✓ LogExpFunctions</span></span>
<span class="line"><span>    677.7 ms  ✓ CommonSubexpressions</span></span>
<span class="line"><span>    786.7 ms  ✓ Static</span></span>
<span class="line"><span>    406.0 ms  ✓ DiffResults</span></span>
<span class="line"><span>    471.5 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>   1574.2 ms  ✓ DispatchDoctor</span></span>
<span class="line"><span>    390.0 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>    380.9 ms  ✓ ArrayInterface → ArrayInterfaceStaticArraysCoreExt</span></span>
<span class="line"><span>    620.2 ms  ✓ Functors</span></span>
<span class="line"><span>   1480.3 ms  ✓ Setfield</span></span>
<span class="line"><span>   1347.7 ms  ✓ LogExpFunctions → LogExpFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    412.9 ms  ✓ BitTwiddlingConvenienceFunctions</span></span>
<span class="line"><span>   1055.6 ms  ✓ CPUSummary</span></span>
<span class="line"><span>   2618.1 ms  ✓ SpecialFunctions</span></span>
<span class="line"><span>    442.7 ms  ✓ DispatchDoctor → DispatchDoctorEnzymeCoreExt</span></span>
<span class="line"><span>   1545.3 ms  ✓ StaticArrayInterface</span></span>
<span class="line"><span>    678.5 ms  ✓ DispatchDoctor → DispatchDoctorChainRulesCoreExt</span></span>
<span class="line"><span>    788.7 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>   1191.2 ms  ✓ LuxCore</span></span>
<span class="line"><span>   7448.9 ms  ✓ StaticArrays</span></span>
<span class="line"><span>    629.1 ms  ✓ PolyesterWeave</span></span>
<span class="line"><span>   1291.7 ms  ✓ Optimisers</span></span>
<span class="line"><span>   1755.7 ms  ✓ SpecialFunctions → SpecialFunctionsChainRulesCoreExt</span></span>
<span class="line"><span>    654.2 ms  ✓ DiffRules</span></span>
<span class="line"><span>    480.9 ms  ✓ CloseOpenIntervals</span></span>
<span class="line"><span>    617.7 ms  ✓ LayoutPointers</span></span>
<span class="line"><span>    656.8 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>    664.4 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>    493.2 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>   2870.2 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    450.9 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    459.3 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    509.4 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    647.7 ms  ✓ StaticArrays → StaticArraysChainRulesCoreExt</span></span>
<span class="line"><span>    631.6 ms  ✓ StaticArrays → StaticArraysStatisticsExt</span></span>
<span class="line"><span>    607.6 ms  ✓ Adapt → AdaptStaticArraysExt</span></span>
<span class="line"><span>    643.7 ms  ✓ ConstructionBase → ConstructionBaseStaticArraysExt</span></span>
<span class="line"><span>    442.9 ms  ✓ Optimisers → OptimisersEnzymeCoreExt</span></span>
<span class="line"><span>    663.1 ms  ✓ StaticArrayInterface → StaticArrayInterfaceStaticArraysExt</span></span>
<span class="line"><span>    440.9 ms  ✓ Optimisers → OptimisersAdaptExt</span></span>
<span class="line"><span>    931.8 ms  ✓ StrideArraysCore</span></span>
<span class="line"><span>   1012.2 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>    769.7 ms  ✓ Polyester</span></span>
<span class="line"><span>   3572.3 ms  ✓ ForwardDiff</span></span>
<span class="line"><span>    867.4 ms  ✓ ForwardDiff → ForwardDiffStaticArraysExt</span></span>
<span class="line"><span>   4174.3 ms  ✓ KernelAbstractions</span></span>
<span class="line"><span>    671.2 ms  ✓ KernelAbstractions → LinearAlgebraExt</span></span>
<span class="line"><span>    747.3 ms  ✓ KernelAbstractions → EnzymeExt</span></span>
<span class="line"><span>   5687.3 ms  ✓ NNlib</span></span>
<span class="line"><span>    854.4 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    934.3 ms  ✓ NNlib → NNlibSpecialFunctionsExt</span></span>
<span class="line"><span>    950.2 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5966.6 ms  ✓ LuxLib</span></span>
<span class="line"><span>   9267.6 ms  ✓ Lux</span></span>
<span class="line"><span>  83 dependencies successfully precompiled in 47 seconds. 24 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArrays...</span></span>
<span class="line"><span>    891.2 ms  ✓ ComponentArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 45 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesComponentArraysExt...</span></span>
<span class="line"><span>    511.7 ms  ✓ MLDataDevices → MLDataDevicesComponentArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 48 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxComponentArraysExt...</span></span>
<span class="line"><span>    508.1 ms  ✓ ComponentArrays → ComponentArraysOptimisersExt</span></span>
<span class="line"><span>   1615.2 ms  ✓ Lux → LuxComponentArraysExt</span></span>
<span class="line"><span>   2181.5 ms  ✓ ComponentArrays → ComponentArraysKernelAbstractionsExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 3 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling SciMLSensitivity...</span></span>
<span class="line"><span>    333.7 ms  ✓ IteratorInterfaceExtensions</span></span>
<span class="line"><span>    394.9 ms  ✓ ExprTools</span></span>
<span class="line"><span>    540.3 ms  ✓ AbstractFFTs</span></span>
<span class="line"><span>    405.1 ms  ✓ StatsAPI</span></span>
<span class="line"><span>    370.1 ms  ✓ CEnum</span></span>
<span class="line"><span>    433.5 ms  ✓ InverseFunctions</span></span>
<span class="line"><span>    578.2 ms  ✓ Serialization</span></span>
<span class="line"><span>    917.4 ms  ✓ FillArrays</span></span>
<span class="line"><span>    448.1 ms  ✓ SuiteSparse_jll</span></span>
<span class="line"><span>    311.3 ms  ✓ DataValueInterfaces</span></span>
<span class="line"><span>    572.3 ms  ✓ OrderedCollections</span></span>
<span class="line"><span>    368.2 ms  ✓ EnumX</span></span>
<span class="line"><span>    379.9 ms  ✓ StructIO</span></span>
<span class="line"><span>    371.1 ms  ✓ RealDot</span></span>
<span class="line"><span>    402.2 ms  ✓ Zlib_jll</span></span>
<span class="line"><span>    356.3 ms  ✓ CompositionsBase</span></span>
<span class="line"><span>    347.4 ms  ✓ PtrArrays</span></span>
<span class="line"><span>    418.1 ms  ✓ HashArrayMappedTries</span></span>
<span class="line"><span>    376.5 ms  ✓ DataAPI</span></span>
<span class="line"><span>    414.6 ms  ✓ SciMLStructures</span></span>
<span class="line"><span>    658.4 ms  ✓ FiniteDiff</span></span>
<span class="line"><span>    980.3 ms  ✓ DifferentiationInterface</span></span>
<span class="line"><span>    538.0 ms  ✓ TruncatedStacktraces</span></span>
<span class="line"><span>   1090.5 ms  ✓ ZygoteRules</span></span>
<span class="line"><span>   1743.0 ms  ✓ RecipesBase</span></span>
<span class="line"><span>    626.9 ms  ✓ ResettableStacks</span></span>
<span class="line"><span>   1996.2 ms  ✓ IRTools</span></span>
<span class="line"><span>    516.5 ms  ✓ FunctionProperties</span></span>
<span class="line"><span>    703.0 ms  ✓ FastPower → FastPowerForwardDiffExt</span></span>
<span class="line"><span>   1161.8 ms  ✓ HypergeometricFunctions</span></span>
<span class="line"><span>    779.4 ms  ✓ PreallocationTools</span></span>
<span class="line"><span>    737.2 ms  ✓ FastBroadcast</span></span>
<span class="line"><span>    357.5 ms  ✓ TableTraits</span></span>
<span class="line"><span>   2729.3 ms  ✓ TimerOutputs</span></span>
<span class="line"><span>    462.7 ms  ✓ AbstractFFTs → AbstractFFTsChainRulesCoreExt</span></span>
<span class="line"><span>    428.9 ms  ✓ InverseFunctions → InverseFunctionsDatesExt</span></span>
<span class="line"><span>   5384.8 ms  ✓ Tracker</span></span>
<span class="line"><span>    457.9 ms  ✓ LogExpFunctions → LogExpFunctionsInverseFunctionsExt</span></span>
<span class="line"><span>    429.3 ms  ✓ RuntimeGeneratedFunctions</span></span>
<span class="line"><span>   1940.2 ms  ✓ Distributed</span></span>
<span class="line"><span>    467.4 ms  ✓ FillArrays → FillArraysStatisticsExt</span></span>
<span class="line"><span>   3811.4 ms  ✓ SparseArrays</span></span>
<span class="line"><span>    485.8 ms  ✓ Parameters</span></span>
<span class="line"><span>   1768.3 ms  ✓ DataStructures</span></span>
<span class="line"><span>  15275.5 ms  ✓ ReverseDiff</span></span>
<span class="line"><span>   2060.9 ms  ✓ ObjectFile</span></span>
<span class="line"><span>    455.6 ms  ✓ CompositionsBase → CompositionsBaseInverseFunctionsExt</span></span>
<span class="line"><span>    466.4 ms  ✓ AliasTables</span></span>
<span class="line"><span>    362.5 ms  ✓ ScopedValues</span></span>
<span class="line"><span>    449.1 ms  ✓ Missings</span></span>
<span class="line"><span>  12168.5 ms  ✓ ArrayLayouts</span></span>
<span class="line"><span>    602.4 ms  ✓ FiniteDiff → FiniteDiffStaticArraysExt</span></span>
<span class="line"><span>    636.1 ms  ✓ DifferentiationInterface → DifferentiationInterfaceStaticArraysExt</span></span>
<span class="line"><span>    502.6 ms  ✓ DifferentiationInterface → DifferentiationInterfaceFiniteDiffExt</span></span>
<span class="line"><span>    437.0 ms  ✓ DifferentiationInterface → DifferentiationInterfaceChainRulesCoreExt</span></span>
<span class="line"><span>    850.1 ms  ✓ DifferentiationInterface → DifferentiationInterfaceForwardDiffExt</span></span>
<span class="line"><span>    810.1 ms  ✓ Tables</span></span>
<span class="line"><span>   1973.6 ms  ✓ StatsFuns</span></span>
<span class="line"><span>   1151.2 ms  ✓ FastPower → FastPowerTrackerExt</span></span>
<span class="line"><span>   6636.6 ms  ✓ LLVM</span></span>
<span class="line"><span>   1147.3 ms  ✓ ArrayInterface → ArrayInterfaceTrackerExt</span></span>
<span class="line"><span>   1144.6 ms  ✓ DifferentiationInterface → DifferentiationInterfaceTrackerExt</span></span>
<span class="line"><span>   1124.1 ms  ✓ NLSolversBase</span></span>
<span class="line"><span>    685.4 ms  ✓ Statistics → SparseArraysExt</span></span>
<span class="line"><span>    664.5 ms  ✓ Adapt → AdaptSparseArraysExt</span></span>
<span class="line"><span>    660.6 ms  ✓ ArrayInterface → ArrayInterfaceSparseArraysExt</span></span>
<span class="line"><span>    677.0 ms  ✓ ChainRulesCore → ChainRulesCoreSparseArraysExt</span></span>
<span class="line"><span>    976.4 ms  ✓ KernelAbstractions → SparseArraysExt</span></span>
<span class="line"><span>    631.0 ms  ✓ SuiteSparse</span></span>
<span class="line"><span>    773.1 ms  ✓ FillArrays → FillArraysSparseArraysExt</span></span>
<span class="line"><span>    651.8 ms  ✓ FiniteDiff → FiniteDiffSparseArraysExt</span></span>
<span class="line"><span>    523.8 ms  ✓ SortingAlgorithms</span></span>
<span class="line"><span>    681.8 ms  ✓ DifferentiationInterface → DifferentiationInterfaceSparseArraysExt</span></span>
<span class="line"><span>   1001.9 ms  ✓ QuadGK</span></span>
<span class="line"><span>   6349.8 ms  ✓ Krylov</span></span>
<span class="line"><span>   3490.1 ms  ✓ FastPower → FastPowerReverseDiffExt</span></span>
<span class="line"><span>   3480.9 ms  ✓ ArrayInterface → ArrayInterfaceReverseDiffExt</span></span>
<span class="line"><span>   3646.9 ms  ✓ DifferentiationInterface → DifferentiationInterfaceReverseDiffExt</span></span>
<span class="line"><span>   3484.1 ms  ✓ PreallocationTools → PreallocationToolsReverseDiffExt</span></span>
<span class="line"><span>   2670.1 ms  ✓ Accessors</span></span>
<span class="line"><span>    834.5 ms  ✓ ArrayLayouts → ArrayLayoutsSparseArraysExt</span></span>
<span class="line"><span>    748.6 ms  ✓ StructArrays</span></span>
<span class="line"><span>    767.3 ms  ✓ StatsFuns → StatsFunsInverseFunctionsExt</span></span>
<span class="line"><span>   1577.6 ms  ✓ StatsFuns → StatsFunsChainRulesCoreExt</span></span>
<span class="line"><span>   1780.2 ms  ✓ UnsafeAtomics → UnsafeAtomicsLLVM</span></span>
<span class="line"><span>    886.7 ms  ✓ PDMats</span></span>
<span class="line"><span>   1816.5 ms  ✓ LineSearches</span></span>
<span class="line"><span>    652.7 ms  ✓ SparseInverseSubset</span></span>
<span class="line"><span>    905.6 ms  ✓ Accessors → LinearAlgebraExt</span></span>
<span class="line"><span>    683.9 ms  ✓ Accessors → StaticArraysExt</span></span>
<span class="line"><span>   2319.3 ms  ✓ StatsBase</span></span>
<span class="line"><span>    409.2 ms  ✓ StructArrays → StructArraysAdaptExt</span></span>
<span class="line"><span>    720.5 ms  ✓ StructArrays → StructArraysSparseArraysExt</span></span>
<span class="line"><span>    696.4 ms  ✓ StructArrays → StructArraysStaticArraysExt</span></span>
<span class="line"><span>   2616.9 ms  ✓ LazyArrays</span></span>
<span class="line"><span>    425.8 ms  ✓ StructArrays → StructArraysLinearAlgebraExt</span></span>
<span class="line"><span>    742.1 ms  ✓ StructArrays → StructArraysGPUArraysCoreExt</span></span>
<span class="line"><span>    487.1 ms  ✓ Accessors → StructArraysExt</span></span>
<span class="line"><span>    695.3 ms  ✓ FillArrays → FillArraysPDMatsExt</span></span>
<span class="line"><span>   1437.1 ms  ✓ Tracker → TrackerPDMatsExt</span></span>
<span class="line"><span>   1859.4 ms  ✓ SciMLOperators</span></span>
<span class="line"><span>   4919.5 ms  ✓ GPUArrays</span></span>
<span class="line"><span>   1549.4 ms  ✓ SymbolicIndexingInterface</span></span>
<span class="line"><span>   1333.3 ms  ✓ LazyArrays → LazyArraysStaticArraysExt</span></span>
<span class="line"><span>   3217.0 ms  ✓ Optim</span></span>
<span class="line"><span>   5551.3 ms  ✓ ChainRules</span></span>
<span class="line"><span>   5239.9 ms  ✓ Distributions</span></span>
<span class="line"><span>    621.3 ms  ✓ SciMLOperators → SciMLOperatorsStaticArraysCoreExt</span></span>
<span class="line"><span>    841.8 ms  ✓ SciMLOperators → SciMLOperatorsSparseArraysExt</span></span>
<span class="line"><span>    836.6 ms  ✓ ArrayInterface → ArrayInterfaceChainRulesExt</span></span>
<span class="line"><span>   2217.2 ms  ✓ RecursiveArrayTools</span></span>
<span class="line"><span>   1486.6 ms  ✓ Distributions → DistributionsChainRulesCoreExt</span></span>
<span class="line"><span>    808.3 ms  ✓ RecursiveArrayTools → RecursiveArrayToolsFastBroadcastExt</span></span>
<span class="line"><span>    679.2 ms  ✓ RecursiveArrayTools → RecursiveArrayToolsStructArraysExt</span></span>
<span class="line"><span>    906.6 ms  ✓ RecursiveArrayTools → RecursiveArrayToolsSparseArraysExt</span></span>
<span class="line"><span>  27800.1 ms  ✓ GPUCompiler</span></span>
<span class="line"><span>    808.8 ms  ✓ RecursiveArrayTools → RecursiveArrayToolsForwardDiffExt</span></span>
<span class="line"><span>   1268.6 ms  ✓ RecursiveArrayTools → RecursiveArrayToolsTrackerExt</span></span>
<span class="line"><span>  11257.5 ms  ✓ SciMLBase</span></span>
<span class="line"><span>   1138.8 ms  ✓ SciMLBase → SciMLBaseChainRulesCoreExt</span></span>
<span class="line"><span>   2880.2 ms  ✓ SciMLJacobianOperators</span></span>
<span class="line"><span>   6163.8 ms  ✓ DiffEqBase</span></span>
<span class="line"><span>  34996.5 ms  ✓ Zygote</span></span>
<span class="line"><span>   1548.5 ms  ✓ DiffEqBase → DiffEqBaseChainRulesCoreExt</span></span>
<span class="line"><span>   2428.2 ms  ✓ DiffEqBase → DiffEqBaseTrackerExt</span></span>
<span class="line"><span>   4930.3 ms  ✓ DiffEqBase → DiffEqBaseReverseDiffExt</span></span>
<span class="line"><span>  17185.4 ms  ✓ LinearSolve</span></span>
<span class="line"><span>   1694.2 ms  ✓ DiffEqBase → DiffEqBaseForwardDiffExt</span></span>
<span class="line"><span>   1919.6 ms  ✓ DiffEqBase → DiffEqBaseDistributionsExt</span></span>
<span class="line"><span>   1481.8 ms  ✓ DiffEqBase → DiffEqBaseSparseArraysExt</span></span>
<span class="line"><span>   1943.2 ms  ✓ Zygote → ZygoteTrackerExt</span></span>
<span class="line"><span>   4372.4 ms  ✓ DiffEqCallbacks</span></span>
<span class="line"><span>   1648.5 ms  ✓ DifferentiationInterface → DifferentiationInterfaceZygoteExt</span></span>
<span class="line"><span>   3288.9 ms  ✓ RecursiveArrayTools → RecursiveArrayToolsZygoteExt</span></span>
<span class="line"><span>   3682.1 ms  ✓ SciMLBase → SciMLBaseZygoteExt</span></span>
<span class="line"><span>   1809.4 ms  ✓ LinearSolve → LinearSolveEnzymeExt</span></span>
<span class="line"><span>   3786.0 ms  ✓ LinearSolve → LinearSolveKernelAbstractionsExt</span></span>
<span class="line"><span>   4716.6 ms  ✓ LinearSolve → LinearSolveSparseArraysExt</span></span>
<span class="line"><span>   4228.1 ms  ✓ DiffEqNoiseProcess</span></span>
<span class="line"><span>   5919.0 ms  ✓ RecursiveArrayTools → RecursiveArrayToolsReverseDiffExt</span></span>
<span class="line"><span>   5223.9 ms  ✓ DiffEqNoiseProcess → DiffEqNoiseProcessReverseDiffExt</span></span>
<span class="line"><span> 218879.5 ms  ✓ Enzyme</span></span>
<span class="line"><span>   6242.4 ms  ✓ Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   8410.0 ms  ✓ Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>  11065.2 ms  ✓ Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   5877.1 ms  ✓ Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>   5919.3 ms  ✓ Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   5786.7 ms  ✓ FastPower → FastPowerEnzymeExt</span></span>
<span class="line"><span>   6005.6 ms  ✓ DifferentiationInterface → DifferentiationInterfaceEnzymeExt</span></span>
<span class="line"><span>   6066.3 ms  ✓ QuadGK → QuadGKEnzymeExt</span></span>
<span class="line"><span>   8213.8 ms  ✓ DiffEqBase → DiffEqBaseEnzymeExt</span></span>
<span class="line"><span>  21157.9 ms  ✓ SciMLSensitivity</span></span>
<span class="line"><span>  152 dependencies successfully precompiled in 330 seconds. 124 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesRecursiveArrayToolsExt...</span></span>
<span class="line"><span>    607.3 ms  ✓ MLDataDevices → MLDataDevicesRecursiveArrayToolsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 47 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArraysRecursiveArrayToolsExt...</span></span>
<span class="line"><span>    714.4 ms  ✓ ComponentArrays → ComponentArraysRecursiveArrayToolsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 69 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArraysSciMLBaseExt...</span></span>
<span class="line"><span>   1089.8 ms  ✓ ComponentArrays → ComponentArraysSciMLBaseExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 89 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesSparseArraysExt...</span></span>
<span class="line"><span>    676.8 ms  ✓ MLDataDevices → MLDataDevicesSparseArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 18 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesFillArraysExt...</span></span>
<span class="line"><span>    467.2 ms  ✓ MLDataDevices → MLDataDevicesFillArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 15 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibEnzymeExt...</span></span>
<span class="line"><span>   1301.3 ms  ✓ LuxLib → LuxLibEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 130 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   6785.0 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 146 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesTrackerExt...</span></span>
<span class="line"><span>   1227.3 ms  ✓ MLDataDevices → MLDataDevicesTrackerExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 60 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibTrackerExt...</span></span>
<span class="line"><span>   1136.3 ms  ✓ LuxCore → LuxCoreArrayInterfaceTrackerExt</span></span>
<span class="line"><span>   3368.9 ms  ✓ LuxLib → LuxLibTrackerExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 4 seconds. 98 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxTrackerExt...</span></span>
<span class="line"><span>   2034.9 ms  ✓ Lux → LuxTrackerExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 112 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArraysTrackerExt...</span></span>
<span class="line"><span>   1212.6 ms  ✓ ComponentArrays → ComponentArraysTrackerExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 71 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesReverseDiffExt...</span></span>
<span class="line"><span>   3565.7 ms  ✓ MLDataDevices → MLDataDevicesReverseDiffExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 49 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibReverseDiffExt...</span></span>
<span class="line"><span>   3508.1 ms  ✓ LuxCore → LuxCoreArrayInterfaceReverseDiffExt</span></span>
<span class="line"><span>   4285.7 ms  ✓ LuxLib → LuxLibReverseDiffExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 5 seconds. 96 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArraysReverseDiffExt...</span></span>
<span class="line"><span>   3628.7 ms  ✓ ComponentArrays → ComponentArraysReverseDiffExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 4 seconds. 57 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReverseDiffExt...</span></span>
<span class="line"><span>   4368.7 ms  ✓ Lux → LuxReverseDiffExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 113 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesChainRulesExt...</span></span>
<span class="line"><span>    859.3 ms  ✓ MLDataDevices → MLDataDevicesChainRulesExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 41 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesZygoteExt...</span></span>
<span class="line"><span>   1608.1 ms  ✓ MLDataDevices → MLDataDevicesGPUArraysExt</span></span>
<span class="line"><span>   1628.6 ms  ✓ MLDataDevices → MLDataDevicesZygoteExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 2 seconds. 109 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxZygoteExt...</span></span>
<span class="line"><span>   1717.5 ms  ✓ WeightInitializers → WeightInitializersGPUArraysExt</span></span>
<span class="line"><span>   2790.9 ms  ✓ Lux → LuxZygoteExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 3 seconds. 165 already precompiled.</span></span>
<span class="line"><span>Precompiling ComponentArraysZygoteExt...</span></span>
<span class="line"><span>   1646.8 ms  ✓ ComponentArrays → ComponentArraysZygoteExt</span></span>
<span class="line"><span>   1860.8 ms  ✓ ComponentArrays → ComponentArraysGPUArraysExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 2 seconds. 117 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxCUDA...</span></span>
<span class="line"><span>    393.5 ms  ✓ LaTeXStrings</span></span>
<span class="line"><span>    397.9 ms  ✓ InvertedIndices</span></span>
<span class="line"><span>    630.9 ms  ✓ InlineStrings</span></span>
<span class="line"><span>    497.2 ms  ✓ PooledArrays</span></span>
<span class="line"><span>   1139.3 ms  ✓ Crayons</span></span>
<span class="line"><span>   2242.8 ms  ✓ FixedPointNumbers</span></span>
<span class="line"><span>   1423.3 ms  ✓ ColorTypes</span></span>
<span class="line"><span>   3583.9 ms  ✓ Test</span></span>
<span class="line"><span>   1395.5 ms  ✓ AbstractFFTs → AbstractFFTsTestExt</span></span>
<span class="line"><span>   1270.5 ms  ✓ LLVM → BFloat16sExt</span></span>
<span class="line"><span>   4586.1 ms  ✓ Colors</span></span>
<span class="line"><span>   1260.7 ms  ✓ NVTX</span></span>
<span class="line"><span>  19347.9 ms  ✓ PrettyTables</span></span>
<span class="line"><span>  43928.9 ms  ✓ DataFrames</span></span>
<span class="line"><span>  46340.1 ms  ✓ CUDA</span></span>
<span class="line"><span>   5844.8 ms  ✓ Atomix → AtomixCUDAExt</span></span>
<span class="line"><span>   8639.1 ms  ✓ cuDNN</span></span>
<span class="line"><span>   5771.9 ms  ✓ LuxCUDA</span></span>
<span class="line"><span>  18 dependencies successfully precompiled in 132 seconds. 86 already precompiled.</span></span>
<span class="line"><span>Precompiling EnzymeBFloat16sExt...</span></span>
<span class="line"><span>   5595.4 ms  ✓ Enzyme → EnzymeBFloat16sExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 47 already precompiled.</span></span>
<span class="line"><span>Precompiling ZygoteColorsExt...</span></span>
<span class="line"><span>   1864.8 ms  ✓ Zygote → ZygoteColorsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling ParsersExt...</span></span>
<span class="line"><span>    547.6 ms  ✓ InlineStrings → ParsersExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 9 already precompiled.</span></span>
<span class="line"><span>Precompiling ArrayInterfaceCUDAExt...</span></span>
<span class="line"><span>   5343.3 ms  ✓ ArrayInterface → ArrayInterfaceCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 105 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDAExt...</span></span>
<span class="line"><span>   5357.5 ms  ✓ CUDA → ChainRulesCoreExt</span></span>
<span class="line"><span>   5813.7 ms  ✓ NNlib → NNlibCUDAExt</span></span>
<span class="line"><span>  2 dependencies successfully precompiled in 6 seconds. 106 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesCUDAExt...</span></span>
<span class="line"><span>   5585.0 ms  ✓ MLDataDevices → MLDataDevicesCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 108 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibCUDAExt...</span></span>
<span class="line"><span>   5561.1 ms  ✓ CUDA → SpecialFunctionsExt</span></span>
<span class="line"><span>   5621.5 ms  ✓ CUDA → EnzymeCoreExt</span></span>
<span class="line"><span>   6026.2 ms  ✓ LuxLib → LuxLibCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 7 seconds. 170 already precompiled.</span></span>
<span class="line"><span>Precompiling DiffEqBaseCUDAExt...</span></span>
<span class="line"><span>    622.1 ms  ✓ InverseFunctions → InverseFunctionsTestExt</span></span>
<span class="line"><span>    658.5 ms  ✓ Accessors → TestExt</span></span>
<span class="line"><span>   6076.4 ms  ✓ DiffEqBase → DiffEqBaseCUDAExt</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 7 seconds. 170 already precompiled.</span></span>
<span class="line"><span>Precompiling LinearSolveCUDAExt...</span></span>
<span class="line"><span>   6591.8 ms  ✓ LinearSolve → LinearSolveCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 164 already precompiled.</span></span>
<span class="line"><span>Precompiling WeightInitializersCUDAExt...</span></span>
<span class="line"><span>   5464.1 ms  ✓ WeightInitializers → WeightInitializersCUDAExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 113 already precompiled.</span></span>
<span class="line"><span>Precompiling NNlibCUDACUDNNExt...</span></span>
<span class="line"><span>   5856.9 ms  ✓ NNlib → NNlibCUDACUDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 110 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicescuDNNExt...</span></span>
<span class="line"><span>   5522.9 ms  ✓ MLDataDevices → MLDataDevicescuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 6 seconds. 111 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxLibcuDNNExt...</span></span>
<span class="line"><span>   6298.7 ms  ✓ LuxLib → LuxLibcuDNNExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 7 seconds. 177 already precompiled.</span></span>
<span class="line"><span>Precompiling OrdinaryDiffEqTsit5...</span></span>
<span class="line"><span>   4052.4 ms  ✓ OrdinaryDiffEqCore</span></span>
<span class="line"><span>   1273.8 ms  ✓ OrdinaryDiffEqCore → OrdinaryDiffEqCoreEnzymeCoreExt</span></span>
<span class="line"><span>   7014.3 ms  ✓ OrdinaryDiffEqTsit5</span></span>
<span class="line"><span>  3 dependencies successfully precompiled in 13 seconds. 97 already precompiled.</span></span>
<span class="line"><span>Precompiling OneHotArrays...</span></span>
<span class="line"><span>    997.9 ms  ✓ OneHotArrays</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 28 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesOneHotArraysExt...</span></span>
<span class="line"><span>    821.0 ms  ✓ MLDataDevices → MLDataDevicesOneHotArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 35 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDatasets...</span></span>
<span class="line"><span>    410.3 ms  ✓ Glob</span></span>
<span class="line"><span>    445.2 ms  ✓ WorkerUtilities</span></span>
<span class="line"><span>    452.5 ms  ✓ TensorCore</span></span>
<span class="line"><span>    469.3 ms  ✓ BufferedStreams</span></span>
<span class="line"><span>    813.3 ms  ✓ InitialValues</span></span>
<span class="line"><span>    402.2 ms  ✓ PrettyPrint</span></span>
<span class="line"><span>    985.3 ms  ✓ OffsetArrays</span></span>
<span class="line"><span>    456.4 ms  ✓ ShowCases</span></span>
<span class="line"><span>    386.7 ms  ✓ SimpleBufferStream</span></span>
<span class="line"><span>    687.8 ms  ✓ URIs</span></span>
<span class="line"><span>    551.5 ms  ✓ TranscodingStreams</span></span>
<span class="line"><span>    377.9 ms  ✓ DefineSingletons</span></span>
<span class="line"><span>    406.1 ms  ✓ LazyModules</span></span>
<span class="line"><span>    474.9 ms  ✓ DelimitedFiles</span></span>
<span class="line"><span>    361.0 ms  ✓ PackageExtensionCompat</span></span>
<span class="line"><span>    392.3 ms  ✓ BitFlags</span></span>
<span class="line"><span>    424.7 ms  ✓ MappedArrays</span></span>
<span class="line"><span>    701.2 ms  ✓ GZip</span></span>
<span class="line"><span>   1067.6 ms  ✓ Baselet</span></span>
<span class="line"><span>    729.1 ms  ✓ ConcurrentUtilities</span></span>
<span class="line"><span>    624.5 ms  ✓ ZipFile</span></span>
<span class="line"><span>    616.3 ms  ✓ Unitful → ConstructionBaseUnitfulExt</span></span>
<span class="line"><span>    610.1 ms  ✓ Unitful → InverseFunctionsUnitfulExt</span></span>
<span class="line"><span>    644.2 ms  ✓ Accessors → UnitfulExt</span></span>
<span class="line"><span>    383.2 ms  ✓ InternedStrings</span></span>
<span class="line"><span>   1185.0 ms  ✓ SimpleTraits</span></span>
<span class="line"><span>    520.2 ms  ✓ ExceptionUnwrapping</span></span>
<span class="line"><span>   1242.3 ms  ✓ SplittablesBase</span></span>
<span class="line"><span>   1468.6 ms  ✓ MPICH_jll</span></span>
<span class="line"><span>    775.7 ms  ✓ WeakRefStrings</span></span>
<span class="line"><span>   2199.6 ms  ✓ AtomsBase</span></span>
<span class="line"><span>    553.9 ms  ✓ FilePathsBase → FilePathsBaseMmapExt</span></span>
<span class="line"><span>   1234.3 ms  ✓ FilePathsBase → FilePathsBaseTestExt</span></span>
<span class="line"><span>   2139.1 ms  ✓ ColorVectorSpace</span></span>
<span class="line"><span>    770.9 ms  ✓ BangBang</span></span>
<span class="line"><span>    436.1 ms  ✓ NameResolution</span></span>
<span class="line"><span>    432.7 ms  ✓ OffsetArrays → OffsetArraysAdaptExt</span></span>
<span class="line"><span>    449.2 ms  ✓ StackViews</span></span>
<span class="line"><span>    504.0 ms  ✓ PaddedViews</span></span>
<span class="line"><span>    520.7 ms  ✓ CodecZlib</span></span>
<span class="line"><span>   9628.3 ms  ✓ JSON3</span></span>
<span class="line"><span>  16692.9 ms  ✓ MLStyle</span></span>
<span class="line"><span>    455.9 ms  ✓ StridedViews</span></span>
<span class="line"><span>   1533.5 ms  ✓ NPZ</span></span>
<span class="line"><span>   1901.1 ms  ✓ OpenSSL</span></span>
<span class="line"><span>   1102.5 ms  ✓ MLCore</span></span>
<span class="line"><span>   1824.8 ms  ✓ HDF5_jll</span></span>
<span class="line"><span>   2285.7 ms  ✓ Chemfiles</span></span>
<span class="line"><span>    709.6 ms  ✓ BangBang → BangBangStaticArraysExt</span></span>
<span class="line"><span>   3488.1 ms  ✓ ColorSchemes</span></span>
<span class="line"><span>   1598.3 ms  ✓ BangBang → BangBangDataFramesExt</span></span>
<span class="line"><span>    521.8 ms  ✓ BangBang → BangBangChainRulesCoreExt</span></span>
<span class="line"><span>    508.2 ms  ✓ BangBang → BangBangTablesExt</span></span>
<span class="line"><span>    502.3 ms  ✓ MosaicViews</span></span>
<span class="line"><span>   1070.6 ms  ✓ MicroCollections</span></span>
<span class="line"><span>   4454.6 ms  ✓ JuliaVariables</span></span>
<span class="line"><span>   2301.7 ms  ✓ Pickle</span></span>
<span class="line"><span>  16179.6 ms  ✓ CSV</span></span>
<span class="line"><span>  31865.8 ms  ✓ JLD2</span></span>
<span class="line"><span>   7194.0 ms  ✓ HDF5</span></span>
<span class="line"><span>  18032.8 ms  ✓ HTTP</span></span>
<span class="line"><span>   2923.0 ms  ✓ Transducers</span></span>
<span class="line"><span>   2359.5 ms  ✓ MAT</span></span>
<span class="line"><span>   1842.2 ms  ✓ FileIO → HTTPExt</span></span>
<span class="line"><span>   3076.7 ms  ✓ DataDeps</span></span>
<span class="line"><span>    710.5 ms  ✓ Transducers → TransducersAdaptExt</span></span>
<span class="line"><span>   1397.8 ms  ✓ Transducers → TransducersDataFramesExt</span></span>
<span class="line"><span>   5330.0 ms  ✓ FLoops</span></span>
<span class="line"><span>  18298.2 ms  ✓ ImageCore</span></span>
<span class="line"><span>   5927.3 ms  ✓ MLUtils</span></span>
<span class="line"><span>   2070.0 ms  ✓ ImageBase</span></span>
<span class="line"><span>   1871.8 ms  ✓ ImageShow</span></span>
<span class="line"><span>   9057.5 ms  ✓ MLDatasets</span></span>
<span class="line"><span>  73 dependencies successfully precompiled in 81 seconds. 127 already precompiled.</span></span>
<span class="line"><span>Precompiling DistributionsTestExt...</span></span>
<span class="line"><span>   1463.2 ms  ✓ Distributions → DistributionsTestExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 53 already precompiled.</span></span>
<span class="line"><span>Precompiling BangBangStructArraysExt...</span></span>
<span class="line"><span>    512.7 ms  ✓ BangBang → BangBangStructArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 24 already precompiled.</span></span>
<span class="line"><span>Precompiling SciMLBaseMLStyleExt...</span></span>
<span class="line"><span>   1134.6 ms  ✓ SciMLBase → SciMLBaseMLStyleExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 61 already precompiled.</span></span>
<span class="line"><span>Precompiling TransducersLazyArraysExt...</span></span>
<span class="line"><span>   1274.2 ms  ✓ Transducers → TransducersLazyArraysExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 1 seconds. 48 already precompiled.</span></span>
<span class="line"><span>Precompiling MLDataDevicesMLUtilsExt...</span></span>
<span class="line"><span>   1624.7 ms  ✓ MLDataDevices → MLDataDevicesMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2136.6 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 3 seconds. 167 already precompiled.</span></span></code></pre></div><h2 id="Loading-MNIST" tabindex="-1">Loading MNIST <a class="header-anchor" href="#Loading-MNIST" aria-label="Permalink to &quot;Loading MNIST {#Loading-MNIST}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadmnist</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, train_split)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Load MNIST: Only 1500 for demonstration purposes</span></span>
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
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Process images into (H,W,C,BS) batches</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(labels_raw, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_test, y_test) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, y_data); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">train_split)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the test data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_test, y_test)); batchsize, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>loadmnist (generic function with 1 method)</span></span></code></pre></div><h2 id="Define-the-Neural-ODE-Layer" tabindex="-1">Define the Neural ODE Layer <a class="header-anchor" href="#Define-the-Neural-ODE-Layer" aria-label="Permalink to &quot;Define the Neural ODE Layer {#Define-the-Neural-ODE-Layer}&quot;">​</a></h2><p>First we will use the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>@compact</code></a> macro to define the Neural ODE Layer.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> NeuralODECompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Lux.AbstractLuxLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; solver</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Tsit5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), tspan</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; model, solver, tspan, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x, p</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">        dudt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(u, p, t) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(u, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), p))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Note the \`p.model\` here</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        prob </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ODEProblem</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ODEFunction{false}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dudt), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x), tspan, p</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> solve</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(prob, solver; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>NeuralODECompact (generic function with 1 method)</span></span></code></pre></div><p>We recommend using the compact macro for creating custom layers. The below implementation exists mostly for historical reasons when <code>@compact</code> was not part of the stable API. Also, it helps users understand how the layer interface of Lux works.</p><p>The NeuralODE is a ContainerLayer, which stores a <code>model</code>. The parameters and states of the NeuralODE are same as those of the underlying model.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> NeuralODE{M</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Lux.AbstractLuxLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,So,T,K} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxWrapperLayer{:model}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">M</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    solver</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">So</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    tspan</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">T</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">K</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> NeuralODE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Lux.AbstractLuxLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; solver</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Tsit5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), tspan</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> NeuralODE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, solver, tspan, kwargs)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##230&quot;.NeuralODE</span></span></code></pre></div><p>OrdinaryDiffEq.jl can deal with non-Vector Inputs! However, certain discrete sensitivities like <code>ReverseDiffAdjoint</code> can&#39;t handle non-Vector inputs. Hence, we need to convert the input and output of the ODE solver to a Vector.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NeuralODE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> dudt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(u, p, t)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        u_, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(u, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), p, st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(u_)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    prob </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ODEProblem{false}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ODEFunction{false}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dudt), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x), n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tspan, ps)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> solve</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(prob, n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">solver; n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@views</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> diffeqsol_to_array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(l</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ODESolution</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">last</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">u), (l, :))</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@views</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> diffeqsol_to_array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(l</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractMatrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x[:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], (l, :))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>diffeqsol_to_array (generic function with 2 methods)</span></span></code></pre></div><h2 id="Create-and-Initialize-the-Neural-ODE-Layer" tabindex="-1">Create and Initialize the Neural ODE Layer <a class="header-anchor" href="#Create-and-Initialize-the-Neural-ODE-Layer" aria-label="Permalink to &quot;Create and Initialize the Neural ODE Layer {#Create-and-Initialize-the-Neural-ODE-Layer}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> create_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model_fn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">NeuralODE;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    use_named_tuple</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    sensealg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">InterpolatingAdjoint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; autojacvec</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ZygoteVJP</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Construct the Neural ODE Model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        FlattenLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">784</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tanh),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        model_fn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tanh), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tanh), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tanh));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            save_everystep</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            reltol</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f-3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            abstol</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f-3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            save_start</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            sensealg,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        Base</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Fix1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(diffeqsol_to_array, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">seed!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((use_named_tuple </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ComponentArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ps)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> model, ps, st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>create_model (generic function with 2 methods)</span></span></code></pre></div><h2 id="Define-Utility-Functions" tabindex="-1">Define Utility Functions <a class="header-anchor" href="#Define-Utility-Functions" aria-label="Permalink to &quot;Define Utility Functions {#Define-Utility-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> logitcrossentropy </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; logits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, dataloader)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    total_correct, total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        predicted_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> predicted_class)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(target_class)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="training" tabindex="-1">Training <a class="header-anchor" href="#training" aria-label="Permalink to &quot;Training&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_function; cpu</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cpu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model, ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> create_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_function; dev, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Training</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_dataloader, test_dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loadmnist</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    tstate </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.001f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    ### Lets train the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    nepochs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 9</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nepochs</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        stime </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> time</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            _, _, _, tstate </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), logitcrossentropy, (x, y), tstate</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ttime </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> time</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> stime</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        tr_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, tstate</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, tstate</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, train_dataloader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, tstate</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, tstate</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, test_dataloader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;[%d/%d]</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Time %.4fs</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Training Accuracy: %.5f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Test \\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 Accuracy: %.5f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch nepochs ttime tr_acc te_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(NeuralODECompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[1/9]	Time 147.0607s	Training Accuracy: 37.48148%	Test Accuracy: 40.00000%</span></span>
<span class="line"><span>[2/9]	Time 0.8177s	Training Accuracy: 58.22222%	Test Accuracy: 57.33333%</span></span>
<span class="line"><span>[3/9]	Time 0.9382s	Training Accuracy: 67.85185%	Test Accuracy: 70.66667%</span></span>
<span class="line"><span>[4/9]	Time 0.7244s	Training Accuracy: 74.29630%	Test Accuracy: 74.66667%</span></span>
<span class="line"><span>[5/9]	Time 0.9347s	Training Accuracy: 76.29630%	Test Accuracy: 76.00000%</span></span>
<span class="line"><span>[6/9]	Time 0.7172s	Training Accuracy: 78.74074%	Test Accuracy: 80.00000%</span></span>
<span class="line"><span>[7/9]	Time 0.9548s	Training Accuracy: 82.22222%	Test Accuracy: 81.33333%</span></span>
<span class="line"><span>[8/9]	Time 0.7181s	Training Accuracy: 83.62963%	Test Accuracy: 83.33333%</span></span>
<span class="line"><span>[9/9]	Time 0.7188s	Training Accuracy: 85.18519%	Test Accuracy: 82.66667%</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(NeuralODE)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[1/9]	Time 39.3721s	Training Accuracy: 37.48148%	Test Accuracy: 40.00000%</span></span>
<span class="line"><span>[2/9]	Time 0.6383s	Training Accuracy: 57.18519%	Test Accuracy: 57.33333%</span></span>
<span class="line"><span>[3/9]	Time 0.8067s	Training Accuracy: 68.37037%	Test Accuracy: 68.00000%</span></span>
<span class="line"><span>[4/9]	Time 0.6434s	Training Accuracy: 73.77778%	Test Accuracy: 75.33333%</span></span>
<span class="line"><span>[5/9]	Time 0.8708s	Training Accuracy: 76.14815%	Test Accuracy: 77.33333%</span></span>
<span class="line"><span>[6/9]	Time 0.6291s	Training Accuracy: 79.48148%	Test Accuracy: 80.66667%</span></span>
<span class="line"><span>[7/9]	Time 0.6619s	Training Accuracy: 81.25926%	Test Accuracy: 80.66667%</span></span>
<span class="line"><span>[8/9]	Time 0.8746s	Training Accuracy: 83.40741%	Test Accuracy: 82.66667%</span></span>
<span class="line"><span>[9/9]	Time 0.6375s	Training Accuracy: 84.81481%	Test Accuracy: 82.00000%</span></span></code></pre></div><p>We can also change the sensealg and train the model! <code>GaussAdjoint</code> allows you to use any arbitrary parameter structure and not just a flat vector (<code>ComponentArray</code>).</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(NeuralODE; sensealg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">GaussAdjoint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; autojacvec</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ZygoteVJP</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()), use_named_tuple</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[1/9]	Time 41.7178s	Training Accuracy: 37.48148%	Test Accuracy: 40.00000%</span></span>
<span class="line"><span>[2/9]	Time 0.6444s	Training Accuracy: 58.44444%	Test Accuracy: 58.00000%</span></span>
<span class="line"><span>[3/9]	Time 0.6414s	Training Accuracy: 66.96296%	Test Accuracy: 68.00000%</span></span>
<span class="line"><span>[4/9]	Time 0.6406s	Training Accuracy: 72.44444%	Test Accuracy: 73.33333%</span></span>
<span class="line"><span>[5/9]	Time 0.8621s	Training Accuracy: 76.37037%	Test Accuracy: 76.00000%</span></span>
<span class="line"><span>[6/9]	Time 0.6321s	Training Accuracy: 78.81481%	Test Accuracy: 79.33333%</span></span>
<span class="line"><span>[7/9]	Time 0.6406s	Training Accuracy: 80.51852%	Test Accuracy: 81.33333%</span></span>
<span class="line"><span>[8/9]	Time 0.8813s	Training Accuracy: 82.74074%	Test Accuracy: 83.33333%</span></span>
<span class="line"><span>[9/9]	Time 0.6277s	Training Accuracy: 85.25926%	Test Accuracy: 82.66667%</span></span></code></pre></div><p>But remember some AD backends like <code>ReverseDiff</code> is not GPU compatible. For a model this size, you will notice that training time is significantly lower for training on CPU than on GPU.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(NeuralODE; sensealg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">InterpolatingAdjoint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; autojacvec</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ReverseDiffVJP</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()), cpu</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[1/9]	Time 40.1717s	Training Accuracy: 37.48148%	Test Accuracy: 40.00000%</span></span>
<span class="line"><span>[2/9]	Time 0.4204s	Training Accuracy: 58.74074%	Test Accuracy: 56.66667%</span></span>
<span class="line"><span>[3/9]	Time 0.3651s	Training Accuracy: 69.92593%	Test Accuracy: 71.33333%</span></span>
<span class="line"><span>[4/9]	Time 0.3689s	Training Accuracy: 72.81481%	Test Accuracy: 74.00000%</span></span>
<span class="line"><span>[5/9]	Time 0.3636s	Training Accuracy: 76.37037%	Test Accuracy: 78.66667%</span></span>
<span class="line"><span>[6/9]	Time 0.3656s	Training Accuracy: 79.03704%	Test Accuracy: 80.66667%</span></span>
<span class="line"><span>[7/9]	Time 0.3647s	Training Accuracy: 81.62963%	Test Accuracy: 80.66667%</span></span>
<span class="line"><span>[8/9]	Time 0.3577s	Training Accuracy: 83.33333%	Test Accuracy: 80.00000%</span></span>
<span class="line"><span>[9/9]	Time 0.3710s	Training Accuracy: 85.40741%	Test Accuracy: 82.00000%</span></span></code></pre></div><p>For completeness, let&#39;s also test out discrete sensitivities!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(NeuralODE; sensealg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ReverseDiffAdjoint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), cpu</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[1/9]	Time 38.6841s	Training Accuracy: 37.48148%	Test Accuracy: 40.00000%</span></span>
<span class="line"><span>[2/9]	Time 11.4830s	Training Accuracy: 58.66667%	Test Accuracy: 57.33333%</span></span>
<span class="line"><span>[3/9]	Time 10.5621s	Training Accuracy: 69.70370%	Test Accuracy: 71.33333%</span></span>
<span class="line"><span>[4/9]	Time 10.5151s	Training Accuracy: 72.74074%	Test Accuracy: 74.00000%</span></span>
<span class="line"><span>[5/9]	Time 10.9738s	Training Accuracy: 76.14815%	Test Accuracy: 78.66667%</span></span>
<span class="line"><span>[6/9]	Time 10.4069s	Training Accuracy: 79.03704%	Test Accuracy: 80.66667%</span></span>
<span class="line"><span>[7/9]	Time 10.5844s	Training Accuracy: 81.55556%	Test Accuracy: 80.66667%</span></span>
<span class="line"><span>[8/9]	Time 11.0439s	Training Accuracy: 83.40741%	Test Accuracy: 80.00000%</span></span>
<span class="line"><span>[9/9]	Time 11.0524s	Training Accuracy: 85.25926%	Test Accuracy: 81.33333%</span></span></code></pre></div><h2 id="Alternate-Implementation-using-Stateful-Layer" tabindex="-1">Alternate Implementation using Stateful Layer <a class="header-anchor" href="#Alternate-Implementation-using-Stateful-Layer" aria-label="Permalink to &quot;Alternate Implementation using Stateful Layer {#Alternate-Implementation-using-Stateful-Layer}&quot;">​</a></h2><p>Starting <code>v0.5.5</code>, Lux provides a <a href="/dev/api/Lux/utilities#Lux.StatefulLuxLayer"><code>StatefulLuxLayer</code></a> which can be used to avoid the <a href="https://github.com/JuliaLang/julia/issues/15276" target="_blank" rel="noreferrer"><code>Box</code>ing of <code>st</code></a>. Using the <code>@compact</code> API avoids this problem entirely.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> StatefulNeuralODE{M</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Lux.AbstractLuxLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,So,T,K} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractLuxWrapperLayer{</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">M</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    solver</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">So</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    tspan</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">T</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">K</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> StatefulNeuralODE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Lux.AbstractLuxLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; solver</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Tsit5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), tspan</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> StatefulNeuralODE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, solver, tspan, kwargs)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">StatefulNeuralODE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> StatefulLuxLayer{true}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    dudt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(u, p, t) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> st_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(u, p)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    prob </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ODEProblem{false}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ODEFunction{false}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dudt), x, n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tspan, ps)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> solve</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(prob, n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">solver; n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), st_model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Train-the-new-Stateful-Neural-ODE" tabindex="-1">Train the new Stateful Neural ODE <a class="header-anchor" href="#Train-the-new-Stateful-Neural-ODE" aria-label="Permalink to &quot;Train the new Stateful Neural ODE {#Train-the-new-Stateful-Neural-ODE}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(StatefulNeuralODE)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[1/9]	Time 36.8740s	Training Accuracy: 37.48148%	Test Accuracy: 40.00000%</span></span>
<span class="line"><span>[2/9]	Time 0.8381s	Training Accuracy: 58.22222%	Test Accuracy: 55.33333%</span></span>
<span class="line"><span>[3/9]	Time 0.6706s	Training Accuracy: 68.29630%	Test Accuracy: 68.66667%</span></span>
<span class="line"><span>[4/9]	Time 0.6514s	Training Accuracy: 73.11111%	Test Accuracy: 76.00000%</span></span>
<span class="line"><span>[5/9]	Time 1.0077s	Training Accuracy: 75.92593%	Test Accuracy: 76.66667%</span></span>
<span class="line"><span>[6/9]	Time 0.6385s	Training Accuracy: 78.96296%	Test Accuracy: 80.66667%</span></span>
<span class="line"><span>[7/9]	Time 0.6386s	Training Accuracy: 80.81481%	Test Accuracy: 81.33333%</span></span>
<span class="line"><span>[8/9]	Time 0.6388s	Training Accuracy: 83.25926%	Test Accuracy: 82.66667%</span></span>
<span class="line"><span>[9/9]	Time 0.6660s	Training Accuracy: 84.59259%	Test Accuracy: 82.00000%</span></span></code></pre></div><p>We might not see a significant difference in the training time, but let us investigate the type stabilities of the layers.</p><h2 id="Type-Stability" tabindex="-1">Type Stability <a class="header-anchor" href="#Type-Stability" aria-label="Permalink to &quot;Type Stability {#Type-Stability}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model, ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> create_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(NeuralODE)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model_stateful, ps_stateful, st_stateful </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> create_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(StatefulNeuralODE)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ones</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">));</span></span></code></pre></div><p>NeuralODE is not type stable due to the boxing of <code>st</code></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@code_warntype</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>MethodInstance for (::Lux.Chain{@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Main.var&quot;##230&quot;.NeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}, OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var&quot;##230&quot;.diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing})(::CUDA.CuArray{Float32, 4, CUDA.DeviceMemory}, ::ComponentArrays.ComponentVector{Float32, CUDA.CuArray{Float32, 1, CUDA.DeviceMemory}, Tuple{ComponentArrays.Axis{(layer_1 = ViewAxis(1:0, Shaped1DAxis((0,))), layer_2 = ViewAxis(1:15700, Axis(weight = ViewAxis(1:15680, ShapedAxis((20, 784))), bias = ViewAxis(15681:15700, Shaped1DAxis((20,))))), layer_3 = ViewAxis(15701:16240, Axis(layer_1 = ViewAxis(1:210, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))), layer_2 = ViewAxis(211:320, Axis(weight = ViewAxis(1:100, ShapedAxis((10, 10))), bias = ViewAxis(101:110, Shaped1DAxis((10,))))), layer_3 = ViewAxis(321:540, Axis(weight = ViewAxis(1:200, ShapedAxis((20, 10))), bias = ViewAxis(201:220, Shaped1DAxis((20,))))))), layer_4 = ViewAxis(16241:16240, Shaped1DAxis((0,))), layer_5 = ViewAxis(16241:16450, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))))}}}, ::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}})</span></span>
<span class="line"><span>  from (c::Lux.Chain)(x, ps, st::NamedTuple) @ Lux /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/src/layers/containers.jl:509</span></span>
<span class="line"><span>Arguments</span></span>
<span class="line"><span>  c::Lux.Chain{@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Main.var&quot;##230&quot;.NeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}, OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var&quot;##230&quot;.diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}</span></span>
<span class="line"><span>  x::CUDA.CuArray{Float32, 4, CUDA.DeviceMemory}</span></span>
<span class="line"><span>  ps::ComponentArrays.ComponentVector{Float32, CUDA.CuArray{Float32, 1, CUDA.DeviceMemory}, Tuple{ComponentArrays.Axis{(layer_1 = ViewAxis(1:0, Shaped1DAxis((0,))), layer_2 = ViewAxis(1:15700, Axis(weight = ViewAxis(1:15680, ShapedAxis((20, 784))), bias = ViewAxis(15681:15700, Shaped1DAxis((20,))))), layer_3 = ViewAxis(15701:16240, Axis(layer_1 = ViewAxis(1:210, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))), layer_2 = ViewAxis(211:320, Axis(weight = ViewAxis(1:100, ShapedAxis((10, 10))), bias = ViewAxis(101:110, Shaped1DAxis((10,))))), layer_3 = ViewAxis(321:540, Axis(weight = ViewAxis(1:200, ShapedAxis((20, 10))), bias = ViewAxis(201:220, Shaped1DAxis((20,))))))), layer_4 = ViewAxis(16241:16240, Shaped1DAxis((0,))), layer_5 = ViewAxis(16241:16450, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))))}}}</span></span>
<span class="line"><span>  st::Core.Const((layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple()), layer_4 = NamedTuple(), layer_5 = NamedTuple()))</span></span>
<span class="line"><span>Body::TUPLE{CUDA.CUARRAY{FLOAT32, 2, CUDA.DEVICEMEMORY}, NAMEDTUPLE{(:LAYER_1, :LAYER_2, :LAYER_3, :LAYER_4, :LAYER_5), &lt;:TUPLE{@NAMEDTUPLE{}, @NAMEDTUPLE{}, ANY, @NAMEDTUPLE{}, @NAMEDTUPLE{}}}}</span></span>
<span class="line"><span>1 ─ %1 = Lux.applychain::Core.Const(Lux.applychain)</span></span>
<span class="line"><span>│   %2 = Base.getproperty(c, :layers)::@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Main.var&quot;##230&quot;.NeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}, OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var&quot;##230&quot;.diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}</span></span>
<span class="line"><span>│   %3 = (%1)(%2, x, ps, st)::TUPLE{CUDA.CUARRAY{FLOAT32, 2, CUDA.DEVICEMEMORY}, NAMEDTUPLE{(:LAYER_1, :LAYER_2, :LAYER_3, :LAYER_4, :LAYER_5), &lt;:TUPLE{@NAMEDTUPLE{}, @NAMEDTUPLE{}, ANY, @NAMEDTUPLE{}, @NAMEDTUPLE{}}}}</span></span>
<span class="line"><span>└──      return %3</span></span></code></pre></div><p>We avoid the problem entirely by using <code>StatefulNeuralODE</code></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@code_warntype</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_stateful</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps_stateful, st_stateful)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>MethodInstance for (::Lux.Chain{@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Main.var&quot;##230&quot;.StatefulNeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}, OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var&quot;##230&quot;.diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing})(::CUDA.CuArray{Float32, 4, CUDA.DeviceMemory}, ::ComponentArrays.ComponentVector{Float32, CUDA.CuArray{Float32, 1, CUDA.DeviceMemory}, Tuple{ComponentArrays.Axis{(layer_1 = ViewAxis(1:0, Shaped1DAxis((0,))), layer_2 = ViewAxis(1:15700, Axis(weight = ViewAxis(1:15680, ShapedAxis((20, 784))), bias = ViewAxis(15681:15700, Shaped1DAxis((20,))))), layer_3 = ViewAxis(15701:16240, Axis(layer_1 = ViewAxis(1:210, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))), layer_2 = ViewAxis(211:320, Axis(weight = ViewAxis(1:100, ShapedAxis((10, 10))), bias = ViewAxis(101:110, Shaped1DAxis((10,))))), layer_3 = ViewAxis(321:540, Axis(weight = ViewAxis(1:200, ShapedAxis((20, 10))), bias = ViewAxis(201:220, Shaped1DAxis((20,))))))), layer_4 = ViewAxis(16241:16240, Shaped1DAxis((0,))), layer_5 = ViewAxis(16241:16450, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))))}}}, ::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}})</span></span>
<span class="line"><span>  from (c::Lux.Chain)(x, ps, st::NamedTuple) @ Lux /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/src/layers/containers.jl:509</span></span>
<span class="line"><span>Arguments</span></span>
<span class="line"><span>  c::Lux.Chain{@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Main.var&quot;##230&quot;.StatefulNeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}, OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var&quot;##230&quot;.diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}</span></span>
<span class="line"><span>  x::CUDA.CuArray{Float32, 4, CUDA.DeviceMemory}</span></span>
<span class="line"><span>  ps::ComponentArrays.ComponentVector{Float32, CUDA.CuArray{Float32, 1, CUDA.DeviceMemory}, Tuple{ComponentArrays.Axis{(layer_1 = ViewAxis(1:0, Shaped1DAxis((0,))), layer_2 = ViewAxis(1:15700, Axis(weight = ViewAxis(1:15680, ShapedAxis((20, 784))), bias = ViewAxis(15681:15700, Shaped1DAxis((20,))))), layer_3 = ViewAxis(15701:16240, Axis(layer_1 = ViewAxis(1:210, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))), layer_2 = ViewAxis(211:320, Axis(weight = ViewAxis(1:100, ShapedAxis((10, 10))), bias = ViewAxis(101:110, Shaped1DAxis((10,))))), layer_3 = ViewAxis(321:540, Axis(weight = ViewAxis(1:200, ShapedAxis((20, 10))), bias = ViewAxis(201:220, Shaped1DAxis((20,))))))), layer_4 = ViewAxis(16241:16240, Shaped1DAxis((0,))), layer_5 = ViewAxis(16241:16450, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))))}}}</span></span>
<span class="line"><span>  st::Core.Const((layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple()), layer_4 = NamedTuple(), layer_5 = NamedTuple()))</span></span>
<span class="line"><span>Body::Tuple{CUDA.CuArray{Float32, 2, CUDA.DeviceMemory}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}}}</span></span>
<span class="line"><span>1 ─ %1 = Lux.applychain::Core.Const(Lux.applychain)</span></span>
<span class="line"><span>│   %2 = Base.getproperty(c, :layers)::@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Main.var&quot;##230&quot;.StatefulNeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}, OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var&quot;##230&quot;.diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}</span></span>
<span class="line"><span>│   %3 = (%1)(%2, x, ps, st)::Tuple{CUDA.CuArray{Float32, 2, CUDA.DeviceMemory}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}}}</span></span>
<span class="line"><span>└──      return %3</span></span></code></pre></div><p>Note, that we still recommend using this layer internally and not exposing this as the default API to the users.</p><p>Finally checking the compact model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model_compact, ps_compact, st_compact </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> create_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(NeuralODECompact)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@code_warntype</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps_compact, st_compact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>MethodInstance for (::Lux.Chain{@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.CompactLuxLayer{:₋₋₋no_special_dispatch₋₋₋, Main.var&quot;##230&quot;.var&quot;#2#3&quot;, Nothing, @NamedTuple{model::Lux.Chain{@NamedTuple{layer_1::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Lux.CompactMacroImpl.ValueStorage{@NamedTuple{}, @NamedTuple{solver::Returns{OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}}, tspan::Returns{Tuple{Float32, Float32}}}}, Tuple{Tuple{Symbol}, Tuple{Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var&quot;##230&quot;.diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing})(::CUDA.CuArray{Float32, 4, CUDA.DeviceMemory}, ::ComponentArrays.ComponentVector{Float32, CUDA.CuArray{Float32, 1, CUDA.DeviceMemory}, Tuple{ComponentArrays.Axis{(layer_1 = ViewAxis(1:0, Shaped1DAxis((0,))), layer_2 = ViewAxis(1:15700, Axis(weight = ViewAxis(1:15680, ShapedAxis((20, 784))), bias = ViewAxis(15681:15700, Shaped1DAxis((20,))))), layer_3 = ViewAxis(15701:16240, Axis(model = ViewAxis(1:540, Axis(layer_1 = ViewAxis(1:210, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))), layer_2 = ViewAxis(211:320, Axis(weight = ViewAxis(1:100, ShapedAxis((10, 10))), bias = ViewAxis(101:110, Shaped1DAxis((10,))))), layer_3 = ViewAxis(321:540, Axis(weight = ViewAxis(1:200, ShapedAxis((20, 10))), bias = ViewAxis(201:220, Shaped1DAxis((20,))))))),)), layer_4 = ViewAxis(16241:16240, Shaped1DAxis((0,))), layer_5 = ViewAxis(16241:16450, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))))}}}, ::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{model::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, solver::OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}, tspan::Tuple{Float32, Float32}, ₋₋₋kwargs₋₋₋::Lux.CompactMacroImpl.KwargsStorage{@NamedTuple{kwargs::Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}})</span></span>
<span class="line"><span>  from (c::Lux.Chain)(x, ps, st::NamedTuple) @ Lux /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/src/layers/containers.jl:509</span></span>
<span class="line"><span>Arguments</span></span>
<span class="line"><span>  c::Lux.Chain{@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.CompactLuxLayer{:₋₋₋no_special_dispatch₋₋₋, Main.var&quot;##230&quot;.var&quot;#2#3&quot;, Nothing, @NamedTuple{model::Lux.Chain{@NamedTuple{layer_1::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Lux.CompactMacroImpl.ValueStorage{@NamedTuple{}, @NamedTuple{solver::Returns{OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}}, tspan::Returns{Tuple{Float32, Float32}}}}, Tuple{Tuple{Symbol}, Tuple{Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var&quot;##230&quot;.diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}</span></span>
<span class="line"><span>  x::CUDA.CuArray{Float32, 4, CUDA.DeviceMemory}</span></span>
<span class="line"><span>  ps::ComponentArrays.ComponentVector{Float32, CUDA.CuArray{Float32, 1, CUDA.DeviceMemory}, Tuple{ComponentArrays.Axis{(layer_1 = ViewAxis(1:0, Shaped1DAxis((0,))), layer_2 = ViewAxis(1:15700, Axis(weight = ViewAxis(1:15680, ShapedAxis((20, 784))), bias = ViewAxis(15681:15700, Shaped1DAxis((20,))))), layer_3 = ViewAxis(15701:16240, Axis(model = ViewAxis(1:540, Axis(layer_1 = ViewAxis(1:210, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))), layer_2 = ViewAxis(211:320, Axis(weight = ViewAxis(1:100, ShapedAxis((10, 10))), bias = ViewAxis(101:110, Shaped1DAxis((10,))))), layer_3 = ViewAxis(321:540, Axis(weight = ViewAxis(1:200, ShapedAxis((20, 10))), bias = ViewAxis(201:220, Shaped1DAxis((20,))))))),)), layer_4 = ViewAxis(16241:16240, Shaped1DAxis((0,))), layer_5 = ViewAxis(16241:16450, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, Shaped1DAxis((10,))))))}}}</span></span>
<span class="line"><span>  st::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{model::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, solver::OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}, tspan::Tuple{Float32, Float32}, ₋₋₋kwargs₋₋₋::Lux.CompactMacroImpl.KwargsStorage{@NamedTuple{kwargs::Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}}</span></span>
<span class="line"><span>Body::Tuple{CUDA.CuArray{Float32, 2, CUDA.DeviceMemory}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{model::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, solver::OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}, tspan::Tuple{Float32, Float32}, ₋₋₋kwargs₋₋₋::Lux.CompactMacroImpl.KwargsStorage{@NamedTuple{kwargs::Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}}}</span></span>
<span class="line"><span>1 ─ %1 = Lux.applychain::Core.Const(Lux.applychain)</span></span>
<span class="line"><span>│   %2 = Base.getproperty(c, :layers)::@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.CompactLuxLayer{:₋₋₋no_special_dispatch₋₋₋, Main.var&quot;##230&quot;.var&quot;#2#3&quot;, Nothing, @NamedTuple{model::Lux.Chain{@NamedTuple{layer_1::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Lux.Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Lux.CompactMacroImpl.ValueStorage{@NamedTuple{}, @NamedTuple{solver::Returns{OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}}, tspan::Returns{Tuple{Float32, Float32}}}}, Tuple{Tuple{Symbol}, Tuple{Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var&quot;##230&quot;.diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}</span></span>
<span class="line"><span>│   %3 = (%1)(%2, x, ps, st)::Tuple{CUDA.CuArray{Float32, 2, CUDA.DeviceMemory}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{model::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, solver::OrdinaryDiffEqTsit5.Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}, tspan::Tuple{Float32, Float32}, ₋₋₋kwargs₋₋₋::Lux.CompactMacroImpl.KwargsStorage{@NamedTuple{kwargs::Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}}}</span></span>
<span class="line"><span>└──      return %3</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.8, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.8</span></span>
<span class="line"><span>NVIDIA driver 560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.8.3</span></span>
<span class="line"><span>- CURAND: 10.3.9</span></span>
<span class="line"><span>- CUFFT: 11.3.3</span></span>
<span class="line"><span>- CUSOLVER: 11.7.2</span></span>
<span class="line"><span>- CUSPARSE: 12.5.7</span></span>
<span class="line"><span>- CUPTI: 2025.1.0 (API 26.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.7.0</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.12.0+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.16.0+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.11.4</span></span>
<span class="line"><span>- LLVM: 16.0.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 3.857 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,63)]))}const y=a(p,[["render",l]]);export{o as __pageData,y as default};
