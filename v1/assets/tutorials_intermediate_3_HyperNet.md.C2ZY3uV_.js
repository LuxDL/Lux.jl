import{_ as a,c as n,o as i,al as p}from"./chunks/framework.D0cb6RzP.js";const d=JSON.parse('{"title":"Training a HyperNetwork on MNIST and FashionMNIST","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/3_HyperNet.md","filePath":"tutorials/intermediate/3_HyperNet.md","lastUpdated":null}'),t={name:"tutorials/intermediate/3_HyperNet.md"};function e(l,s,h,k,c,r){return i(),n("div",null,[...s[0]||(s[0]=[p(`<h1 id="Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" tabindex="-1">Training a HyperNetwork on MNIST and FashionMNIST <a class="header-anchor" href="#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST" aria-label="Permalink to &quot;Training a HyperNetwork on MNIST and FashionMNIST {#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST}&quot;">​</a></h1><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ComponentArrays, MLDatasets, MLUtils, OneHotArrays, Optimisers, Printf, Random, Reactant</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>  13130.2 ms  ? Enzyme</span></span>
<span class="line"><span>  13619.6 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>Info Given Reactant was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0KWARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>\x1B[0KERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>  14845.8 ms  ? Reactant</span></span>
<span class="line"><span>WARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>ERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>Precompiling Enzyme...</span></span>
<span class="line"><span>Info Given Enzyme was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0KWARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>\x1B[0KERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>  13275.6 ms  ? Enzyme</span></span>
<span class="line"><span>WARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>ERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>  13180.5 ms  ? Enzyme</span></span>
<span class="line"><span>    724.8 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    874.8 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    916.3 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    715.8 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>Info Given LuxEnzymeExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    693.0 ms  ? Lux → LuxEnzymeExt</span></span>
<span class="line"><span>    775.0 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeLogExpFunctionsExt...</span></span>
<span class="line"><span>  13171.3 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeLogExpFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    717.4 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeChainRulesCoreExt...</span></span>
<span class="line"><span>  13275.0 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeChainRulesCoreExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    775.8 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeSpecialFunctionsExt...</span></span>
<span class="line"><span>  13043.5 ms  ? Enzyme</span></span>
<span class="line"><span>    747.5 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>Info Given EnzymeSpecialFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    852.1 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeGPUArraysCoreExt...</span></span>
<span class="line"><span>  13126.3 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeGPUArraysCoreExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    705.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeStaticArraysExt...</span></span>
<span class="line"><span>  12986.0 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeStaticArraysExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    961.5 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling OptimisersReactantExt...</span></span>
<span class="line"><span>  13073.6 ms  ? Enzyme</span></span>
<span class="line"><span>    719.7 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    733.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1900.6 ms  ? Reactant</span></span>
<span class="line"><span>    696.3 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>Info Given OptimisersReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    741.9 ms  ? Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  13156.2 ms  ? Enzyme</span></span>
<span class="line"><span>    702.3 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1966.7 ms  ? Reactant</span></span>
<span class="line"><span>Info Given LuxCoreReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    782.1 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  13138.5 ms  ? Enzyme</span></span>
<span class="line"><span>    698.9 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1900.6 ms  ? Reactant</span></span>
<span class="line"><span>Info Given MLDataDevicesReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    749.8 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  13355.2 ms  ? Enzyme</span></span>
<span class="line"><span>    714.9 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    730.7 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    862.6 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   1907.0 ms  ? Reactant</span></span>
<span class="line"><span>    693.9 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>Info Given WeightInitializersReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    737.7 ms  ? WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>    901.1 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ComponentArraysReactantExt...</span></span>
<span class="line"><span>  13706.7 ms  ? Enzyme</span></span>
<span class="line"><span>    722.6 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    755.8 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   1913.3 ms  ? Reactant</span></span>
<span class="line"><span>    705.8 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>Info Given ComponentArraysReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    831.4 ms  ? ComponentArrays → ComponentArraysReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantKernelAbstractionsExt...</span></span>
<span class="line"><span>  13376.0 ms  ? Enzyme</span></span>
<span class="line"><span>    719.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    936.9 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   1964.5 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantKernelAbstractionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    691.2 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantArrayInterfaceExt...</span></span>
<span class="line"><span>  13181.9 ms  ? Enzyme</span></span>
<span class="line"><span>    711.4 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1927.1 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantArrayInterfaceExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    703.3 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantSpecialFunctionsExt...</span></span>
<span class="line"><span>  13104.9 ms  ? Enzyme</span></span>
<span class="line"><span>    728.7 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    734.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    865.4 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   1970.7 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantSpecialFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    888.3 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantStatisticsExt...</span></span>
<span class="line"><span>  13694.5 ms  ? Enzyme</span></span>
<span class="line"><span>    741.0 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1983.3 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantStatisticsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    728.0 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantOneHotArraysExt...</span></span>
<span class="line"><span>  13326.6 ms  ? Enzyme</span></span>
<span class="line"><span>    721.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    737.4 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    935.1 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   2033.3 ms  ? Reactant</span></span>
<span class="line"><span>    689.8 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    778.9 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>Info Given ReactantOneHotArraysExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>   1042.5 ms  ? Reactant → ReactantOneHotArraysExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxLibReactantExt...</span></span>
<span class="line"><span>  13379.0 ms  ? Enzyme</span></span>
<span class="line"><span>    751.7 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    896.8 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    939.8 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    745.6 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    725.8 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1916.9 ms  ? Reactant</span></span>
<span class="line"><span>    701.5 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    785.9 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>    888.3 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>    760.1 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>    785.5 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>    762.7 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>Info Given LuxLibReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    728.0 ms  ? LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  13216.2 ms  ? Enzyme</span></span>
<span class="line"><span>    746.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    760.0 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    916.3 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   2035.0 ms  ? Reactant</span></span>
<span class="line"><span>    699.8 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    725.5 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>Info Given ReactantNNlibExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>   1163.4 ms  ? Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-014a-71b5118e5067 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  13461.3 ms  ? Enzyme</span></span>
<span class="line"><span>    755.8 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    850.0 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    987.7 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    745.2 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    701.1 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    769.1 ms  ? Lux → LuxEnzymeExt</span></span>
<span class="line"><span>   1919.3 ms  ? Reactant</span></span>
<span class="line"><span>    692.9 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    716.1 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>    869.0 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>    783.4 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>    852.8 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>    760.5 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>    771.8 ms  ? Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>    693.7 ms  ? LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>    790.8 ms  ? WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>Info Given LuxReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    719.4 ms  ? Lux → LuxReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-6d16-734699a07e98 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span></code></pre></div><h2 id="Loading-Datasets" tabindex="-1">Loading Datasets <a class="header-anchor" href="#Loading-Datasets" aria-label="Permalink to &quot;Loading Datasets {#Loading-Datasets}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> load_dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    ::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{dset}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_train</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Nothing,Int}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_eval</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Nothing,Int}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {dset}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (; features, targets) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> n_train </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">===</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        tmp </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        tmp[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tmp)]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n_train]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_train, y_train </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(features, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, :), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (; features, targets) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> n_eval </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">===</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        tmp </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:test</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        tmp[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tmp)]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        dset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:test</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n_eval]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_test, y_test </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(features, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, :), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (x_train, y_train);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">min</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_train, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (x_test, y_test);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">min</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_test, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> load_datasets</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    n_train </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> parse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Bool, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ENV</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;CI&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;false&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1024</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> :</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    n_eval </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> parse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Bool, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ENV</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;CI&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;false&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 32</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> :</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> load_dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((MNIST, FashionMNIST), n_train, n_eval, batchsize)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Implement-a-HyperNet-Layer" tabindex="-1">Implement a HyperNet Layer <a class="header-anchor" href="#Implement-a-HyperNet-Layer" aria-label="Permalink to &quot;Implement a HyperNet Layer {#Implement-a-HyperNet-Layer}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> HyperNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(weight_generator</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, core_network</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ca_axes </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> getaxes</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        ComponentArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">initialparameters</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), core_network))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; ca_axes, weight_generator, core_network, dispatch</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:HyperNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Generate the weights</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ps_new </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ComponentArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">weight_generator</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), ca_axes)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> core_network</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps_new)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>Defining functions on the CompactLuxLayer requires some understanding of how the layer is structured, as such we don&#39;t recommend doing it unless you are familiar with the internals. In this case, we simply write it to ignore the initialization of the <code>core_network</code> parameters.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">initialparameters</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractRNG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, hn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">CompactLuxLayer{:HyperNet}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (; weight_generator</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">initialparameters</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, hn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">weight_generator))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Create-and-Initialize-the-HyperNet" tabindex="-1">Create and Initialize the HyperNet <a class="header-anchor" href="#Create-and-Initialize-the-HyperNet" aria-label="Permalink to &quot;Create and Initialize the HyperNet {#Create-and-Initialize-the-HyperNet}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> create_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    core_network </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu; stride</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">16</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu; stride</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu; stride</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        GlobalMeanPool</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        FlattenLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> HyperNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Embedding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">parameterlength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(core_network)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        core_network,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Define-Utility-Functions" tabindex="-1">Define Utility Functions <a class="header-anchor" href="#Define-Utility-Functions" aria-label="Permalink to &quot;Define Utility Functions {#Define-Utility-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, dataloader, data_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    total_correct, total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        predicted_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((data_idx, x), ps, st))))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> predicted_class)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(target_class)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Training" tabindex="-1">Training <a class="header-anchor" href="#Training" aria-label="Permalink to &quot;Training {#Training}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; force</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> create_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dataloaders </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">load_datasets</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">seed!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1234</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), model))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0003f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataloaders[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data_idx </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ConcreteRNumber</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((data_idx, x), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    ### Let&#39;s train the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    nepochs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 50</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nepochs, data_idx </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_dataloader, test_dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(dataloaders[data_idx])</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        ### This allows us to trace the data index, else it will be embedded as a constant</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        ### in the IR</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        concrete_data_idx </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ConcreteRNumber</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data_idx)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        stime </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> time</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, _, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; logits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                ((concrete_data_idx, x), y),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                return_gradients</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ttime </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> time</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> stime</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> round</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                model_compiled,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_dataloader,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                concrete_data_idx,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            digits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        test_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> round</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                model_compiled,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                test_dataloader,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                concrete_data_idx,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            digits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_dataloader, test_dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(dataloaders[data_idx])</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        concrete_data_idx </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ConcreteRNumber</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> round</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                model_compiled,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_dataloader,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                concrete_data_idx,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            digits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        test_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> round</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                model_compiled,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                test_dataloader,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                concrete_data_idx,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            digits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">test_acc_list </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>AssertionError(&quot;Could not find registered platform with name: \\&quot;cuda\\&quot;. Available platform names are: &quot;)</span></span>
<span class="line"><span>[  1/ 50]	       MNIST	Time 47.25133s	Training Accuracy: 34.57%	Test Accuracy: 37.50%</span></span>
<span class="line"><span>[  1/ 50]	FashionMNIST	Time 0.19298s	Training Accuracy: 32.52%	Test Accuracy: 43.75%</span></span>
<span class="line"><span>[  2/ 50]	       MNIST	Time 0.22306s	Training Accuracy: 36.33%	Test Accuracy: 34.38%</span></span>
<span class="line"><span>[  2/ 50]	FashionMNIST	Time 0.23245s	Training Accuracy: 46.19%	Test Accuracy: 46.88%</span></span>
<span class="line"><span>[  3/ 50]	       MNIST	Time 0.16945s	Training Accuracy: 42.68%	Test Accuracy: 28.12%</span></span>
<span class="line"><span>[  3/ 50]	FashionMNIST	Time 0.22751s	Training Accuracy: 56.74%	Test Accuracy: 56.25%</span></span>
<span class="line"><span>[  4/ 50]	       MNIST	Time 0.25773s	Training Accuracy: 51.27%	Test Accuracy: 37.50%</span></span>
<span class="line"><span>[  4/ 50]	FashionMNIST	Time 0.21111s	Training Accuracy: 64.55%	Test Accuracy: 56.25%</span></span>
<span class="line"><span>[  5/ 50]	       MNIST	Time 0.23107s	Training Accuracy: 57.03%	Test Accuracy: 40.62%</span></span>
<span class="line"><span>[  5/ 50]	FashionMNIST	Time 0.23048s	Training Accuracy: 71.19%	Test Accuracy: 56.25%</span></span>
<span class="line"><span>[  6/ 50]	       MNIST	Time 0.18307s	Training Accuracy: 62.70%	Test Accuracy: 34.38%</span></span>
<span class="line"><span>[  6/ 50]	FashionMNIST	Time 0.16869s	Training Accuracy: 75.39%	Test Accuracy: 56.25%</span></span>
<span class="line"><span>[  7/ 50]	       MNIST	Time 0.16684s	Training Accuracy: 69.04%	Test Accuracy: 43.75%</span></span>
<span class="line"><span>[  7/ 50]	FashionMNIST	Time 0.16792s	Training Accuracy: 75.88%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[  8/ 50]	       MNIST	Time 0.17440s	Training Accuracy: 73.93%	Test Accuracy: 46.88%</span></span>
<span class="line"><span>[  8/ 50]	FashionMNIST	Time 0.16443s	Training Accuracy: 81.25%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[  9/ 50]	       MNIST	Time 0.19038s	Training Accuracy: 79.59%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[  9/ 50]	FashionMNIST	Time 0.16802s	Training Accuracy: 84.57%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 10/ 50]	       MNIST	Time 0.17108s	Training Accuracy: 83.20%	Test Accuracy: 53.12%</span></span>
<span class="line"><span>[ 10/ 50]	FashionMNIST	Time 0.17530s	Training Accuracy: 87.70%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 11/ 50]	       MNIST	Time 0.16764s	Training Accuracy: 86.13%	Test Accuracy: 53.12%</span></span>
<span class="line"><span>[ 11/ 50]	FashionMNIST	Time 0.16947s	Training Accuracy: 88.18%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 12/ 50]	       MNIST	Time 0.21124s	Training Accuracy: 90.23%	Test Accuracy: 50.00%</span></span>
<span class="line"><span>[ 12/ 50]	FashionMNIST	Time 0.17310s	Training Accuracy: 90.92%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 13/ 50]	       MNIST	Time 0.17504s	Training Accuracy: 94.34%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 13/ 50]	FashionMNIST	Time 0.17976s	Training Accuracy: 92.87%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 14/ 50]	       MNIST	Time 0.18228s	Training Accuracy: 95.12%	Test Accuracy: 53.12%</span></span>
<span class="line"><span>[ 14/ 50]	FashionMNIST	Time 0.17146s	Training Accuracy: 94.24%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 15/ 50]	       MNIST	Time 0.17092s	Training Accuracy: 96.29%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 15/ 50]	FashionMNIST	Time 0.22091s	Training Accuracy: 94.63%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 16/ 50]	       MNIST	Time 0.16761s	Training Accuracy: 98.14%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 16/ 50]	FashionMNIST	Time 0.20761s	Training Accuracy: 96.39%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 17/ 50]	       MNIST	Time 0.18366s	Training Accuracy: 99.61%	Test Accuracy: 65.62%</span></span>
<span class="line"><span>[ 17/ 50]	FashionMNIST	Time 0.19053s	Training Accuracy: 97.27%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 18/ 50]	       MNIST	Time 0.17661s	Training Accuracy: 99.71%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 18/ 50]	FashionMNIST	Time 0.22203s	Training Accuracy: 96.68%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 19/ 50]	       MNIST	Time 0.20205s	Training Accuracy: 99.80%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 19/ 50]	FashionMNIST	Time 0.17773s	Training Accuracy: 99.02%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 20/ 50]	       MNIST	Time 0.17589s	Training Accuracy: 99.90%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 20/ 50]	FashionMNIST	Time 0.17808s	Training Accuracy: 99.12%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 21/ 50]	       MNIST	Time 0.17368s	Training Accuracy: 99.90%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 21/ 50]	FashionMNIST	Time 0.20439s	Training Accuracy: 98.83%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 22/ 50]	       MNIST	Time 0.18979s	Training Accuracy: 99.90%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 22/ 50]	FashionMNIST	Time 0.18065s	Training Accuracy: 99.51%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 23/ 50]	       MNIST	Time 0.18448s	Training Accuracy: 99.90%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 23/ 50]	FashionMNIST	Time 0.17725s	Training Accuracy: 99.71%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 24/ 50]	       MNIST	Time 0.18730s	Training Accuracy: 99.90%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 24/ 50]	FashionMNIST	Time 0.16720s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 25/ 50]	       MNIST	Time 0.17303s	Training Accuracy: 100.00%	Test Accuracy: 68.75%</span></span>
<span class="line"><span>[ 25/ 50]	FashionMNIST	Time 0.17709s	Training Accuracy: 100.00%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 26/ 50]	       MNIST	Time 0.17719s	Training Accuracy: 100.00%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 26/ 50]	FashionMNIST	Time 0.17276s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 27/ 50]	       MNIST	Time 0.16721s	Training Accuracy: 100.00%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 27/ 50]	FashionMNIST	Time 0.17238s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 28/ 50]	       MNIST	Time 0.16904s	Training Accuracy: 100.00%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 28/ 50]	FashionMNIST	Time 0.17979s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 29/ 50]	       MNIST	Time 0.17906s	Training Accuracy: 100.00%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 29/ 50]	FashionMNIST	Time 0.19616s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 30/ 50]	       MNIST	Time 0.16879s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 30/ 50]	FashionMNIST	Time 0.18608s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 31/ 50]	       MNIST	Time 0.17991s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 31/ 50]	FashionMNIST	Time 0.19045s	Training Accuracy: 100.00%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 32/ 50]	       MNIST	Time 0.17816s	Training Accuracy: 100.00%	Test Accuracy: 62.50%</span></span>
<span class="line"><span>[ 32/ 50]	FashionMNIST	Time 0.17921s	Training Accuracy: 100.00%	Test Accuracy: 78.12%</span></span>
<span class="line"><span>[ 33/ 50]	       MNIST	Time 0.18032s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 33/ 50]	FashionMNIST	Time 0.17738s	Training Accuracy: 100.00%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 34/ 50]	       MNIST	Time 0.17516s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 34/ 50]	FashionMNIST	Time 0.17145s	Training Accuracy: 100.00%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 35/ 50]	       MNIST	Time 0.18243s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 35/ 50]	FashionMNIST	Time 0.18124s	Training Accuracy: 100.00%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 36/ 50]	       MNIST	Time 0.19813s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 36/ 50]	FashionMNIST	Time 0.17606s	Training Accuracy: 100.00%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 37/ 50]	       MNIST	Time 0.18434s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 37/ 50]	FashionMNIST	Time 0.16323s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 38/ 50]	       MNIST	Time 0.17885s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 38/ 50]	FashionMNIST	Time 0.17277s	Training Accuracy: 100.00%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 39/ 50]	       MNIST	Time 0.18511s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 39/ 50]	FashionMNIST	Time 0.17106s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 40/ 50]	       MNIST	Time 0.23123s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 40/ 50]	FashionMNIST	Time 0.17562s	Training Accuracy: 100.00%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 41/ 50]	       MNIST	Time 0.17778s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 41/ 50]	FashionMNIST	Time 0.18432s	Training Accuracy: 100.00%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 42/ 50]	       MNIST	Time 0.18230s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 42/ 50]	FashionMNIST	Time 0.19206s	Training Accuracy: 100.00%	Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 43/ 50]	       MNIST	Time 0.17065s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 43/ 50]	FashionMNIST	Time 0.17754s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 44/ 50]	       MNIST	Time 0.17320s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 44/ 50]	FashionMNIST	Time 0.17923s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 45/ 50]	       MNIST	Time 0.16997s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 45/ 50]	FashionMNIST	Time 0.17484s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 46/ 50]	       MNIST	Time 0.17846s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 46/ 50]	FashionMNIST	Time 0.19340s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 47/ 50]	       MNIST	Time 0.16674s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 47/ 50]	FashionMNIST	Time 0.18571s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 48/ 50]	       MNIST	Time 0.17814s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 48/ 50]	FashionMNIST	Time 0.17444s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 49/ 50]	       MNIST	Time 0.17209s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 49/ 50]	FashionMNIST	Time 0.17431s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span>[ 50/ 50]	       MNIST	Time 0.18021s	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[ 50/ 50]	FashionMNIST	Time 0.16950s	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 59.38%</span></span>
<span class="line"><span>[FINAL]	FashionMNIST	Training Accuracy: 100.00%	Test Accuracy: 71.88%</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.6</span></span>
<span class="line"><span>Commit 9615af0f269 (2025-07-09 12:58 UTC)</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,22)])])}const m=a(t,[["render",e]]);export{d as __pageData,m as default};
