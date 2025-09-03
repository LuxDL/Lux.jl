import{_ as a,c as n,o as i,al as p}from"./chunks/framework.CDztWmMz.js";const d=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),l={name:"tutorials/intermediate/6_GCN_Cora.md"};function t(e,s,h,k,c,E){return i(),n("div",null,[...s[0]||(s[0]=[p(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Reactant,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    MLDatasets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Statistics,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    GNNGraphs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ConcreteStructs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Printf,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    OneHotArrays,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Optimisers</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; force</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>  13328.3 ms  ? Enzyme</span></span>
<span class="line"><span>  13441.2 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>Info Given Reactant was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0KWARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>\x1B[0KERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>  14979.7 ms  ? Reactant</span></span>
<span class="line"><span>WARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>ERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>Precompiling Enzyme...</span></span>
<span class="line"><span>Info Given Enzyme was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0KWARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>\x1B[0KERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>  13442.8 ms  ? Enzyme</span></span>
<span class="line"><span>WARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>ERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>  13541.0 ms  ? Enzyme</span></span>
<span class="line"><span>    802.5 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    906.2 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    951.4 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    714.5 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    701.7 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>Info Given LuxEnzymeExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    720.1 ms  ? Lux → LuxEnzymeExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeLogExpFunctionsExt...</span></span>
<span class="line"><span>  13044.7 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeLogExpFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    727.8 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeChainRulesCoreExt...</span></span>
<span class="line"><span>  13333.6 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeChainRulesCoreExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    695.1 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeSpecialFunctionsExt...</span></span>
<span class="line"><span>  13101.3 ms  ? Enzyme</span></span>
<span class="line"><span>    694.0 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>Info Given EnzymeSpecialFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    831.8 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeGPUArraysCoreExt...</span></span>
<span class="line"><span>  13256.2 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeGPUArraysCoreExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    695.7 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeStaticArraysExt...</span></span>
<span class="line"><span>  13103.9 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeStaticArraysExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    938.6 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling OptimisersReactantExt...</span></span>
<span class="line"><span>  13573.6 ms  ? Enzyme</span></span>
<span class="line"><span>    710.2 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    751.4 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   2108.1 ms  ? Reactant</span></span>
<span class="line"><span>    692.1 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>Info Given OptimisersReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    769.1 ms  ? Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  13351.1 ms  ? Enzyme</span></span>
<span class="line"><span>    708.8 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   2053.9 ms  ? Reactant</span></span>
<span class="line"><span>Info Given LuxCoreReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    742.3 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  13515.8 ms  ? Enzyme</span></span>
<span class="line"><span>    732.2 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   2168.4 ms  ? Reactant</span></span>
<span class="line"><span>Info Given MLDataDevicesReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    773.7 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  13661.9 ms  ? Enzyme</span></span>
<span class="line"><span>    716.0 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    749.3 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    895.9 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   2015.9 ms  ? Reactant</span></span>
<span class="line"><span>    729.3 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>Info Given WeightInitializersReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    757.0 ms  ? WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>    888.5 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantKernelAbstractionsExt...</span></span>
<span class="line"><span>  13202.8 ms  ? Enzyme</span></span>
<span class="line"><span>    724.0 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    899.2 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   2112.8 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantKernelAbstractionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    671.8 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantArrayInterfaceExt...</span></span>
<span class="line"><span>  13424.9 ms  ? Enzyme</span></span>
<span class="line"><span>    743.2 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1995.9 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantArrayInterfaceExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    697.8 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantSpecialFunctionsExt...</span></span>
<span class="line"><span>  13235.2 ms  ? Enzyme</span></span>
<span class="line"><span>    735.2 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    750.5 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    871.9 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   1913.2 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantSpecialFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    860.6 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantStatisticsExt...</span></span>
<span class="line"><span>  13470.2 ms  ? Enzyme</span></span>
<span class="line"><span>    748.7 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   2007.3 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantStatisticsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    684.0 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxLibReactantExt...</span></span>
<span class="line"><span>  13345.1 ms  ? Enzyme</span></span>
<span class="line"><span>    771.6 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    900.8 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    947.7 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    796.2 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    710.6 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1962.6 ms  ? Reactant</span></span>
<span class="line"><span>    726.7 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    756.4 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>    881.1 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>    694.9 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>    759.8 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>    769.4 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>Info Given LuxLibReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    700.6 ms  ? LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  13526.2 ms  ? Enzyme</span></span>
<span class="line"><span>    715.1 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    751.2 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    919.7 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   2044.6 ms  ? Reactant</span></span>
<span class="line"><span>    711.1 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    727.8 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>Info Given ReactantNNlibExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>   1023.3 ms  ? Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  13301.1 ms  ? Enzyme</span></span>
<span class="line"><span>    777.3 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    899.4 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    957.4 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    753.7 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    707.0 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    714.2 ms  ? Lux → LuxEnzymeExt</span></span>
<span class="line"><span>   1919.7 ms  ? Reactant</span></span>
<span class="line"><span>    713.8 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>    729.5 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    888.3 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>    702.7 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>    735.4 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>    744.8 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>    762.4 ms  ? WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>    808.8 ms  ? Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>    704.3 ms  ? LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>Info Given LuxReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    694.1 ms  ? Lux → LuxReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantOneHotArraysExt...</span></span>
<span class="line"><span>  13254.8 ms  ? Enzyme</span></span>
<span class="line"><span>    707.3 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    760.9 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    935.3 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   1932.1 ms  ? Reactant</span></span>
<span class="line"><span>    707.3 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    727.2 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>Info Given ReactantOneHotArraysExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>   1032.6 ms  ? Reactant → ReactantOneHotArraysExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-73cf-37c3b5c284cc is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span></code></pre></div><h2 id="Loading-Cora-Dataset" tabindex="-1">Loading Cora Dataset <a class="header-anchor" href="#Loading-Cora-Dataset" aria-label="Permalink to &quot;Loading Cora Dataset {#Loading-Cora-Dataset}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Cora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">graphs[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gnngraph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GNNGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_index; ndata</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data, edata</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_data, gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">num_nodes</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">features,</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">targets, data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">metadata[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;classes&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # We use a dense matrix here to avoid incompatibility with Reactant</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Matrix{Int32}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">adjacency_matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gnngraph)),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # We use this since Reactant doesn&#39;t yet support gather adjoint</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">140</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">141</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">640</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1709</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2708</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Model-Definition" tabindex="-1">Model Definition <a class="header-anchor" href="#Model-Definition" aria-label="Permalink to &quot;Model Definition {#Model-Definition}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dense</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> adj</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, h_dim, out_dim; nb_layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dropout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    layer_sizes </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, [h_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nb_layers])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn_layers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (in_dim, out_dim) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> zip</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)], layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    last_layer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dropout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dropout)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; gcn_layers, dropout, last_layer) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj, mask)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> layer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gcn_layers</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> relu</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> last_layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj))[:, mask]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Helper-Functions" tabindex="-1">Helper Functions <a class="header-anchor" href="#Helper-Functions" aria-label="Permalink to &quot;Helper Functions {#Helper-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y, adj, mask))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj, mask), ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; agg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">mean, logits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))(y_pred, y[:, mask])</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st, (; y_pred)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    hidden_dim</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dropout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    nb_layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    use_bias</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lr</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.001</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    weight_decay</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    patience</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    epochs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">200</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">seed!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    features, targets, adj, (train_idx, val_idx, test_idx) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(features, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), hidden_dim, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); nb_layers, dropout, use_bias)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, gcn))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    opt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(weight_decay) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lr) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AdamW</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; eta</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lr, lambda</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">weight_decay)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gcn, ps, st, opt)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Total Trainable Parameters: %0.4f M</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">parameterlength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ps) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0e6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_loss_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gcn, ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st), (features, targets, adj, val_idx))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, train_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, val_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Inf</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">epochs</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss_function,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (features, targets, adj, train_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            return_gradients</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                train_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, train_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, train_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            val_loss_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                gcn,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                (features, targets, adj, val_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                val_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, val_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, val_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch %3d</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Train Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Train Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Val Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 Val Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss train_acc val_loss val_acc</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> best_loss_val</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loss</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> patience</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Early Stopping at Epoch %d</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                break</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        test_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                gcn,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                (features, targets, adj, test_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        test_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                    gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                        (features, adj, test_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                        train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                        Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, test_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Test Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Test Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> test_loss test_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling EnzymeBFloat16sExt...</span></span>
<span class="line"><span>  13349.3 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeBFloat16sExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    692.4 ms  ? Enzyme → EnzymeBFloat16sExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5644-40f260d4fd9b is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>AssertionError(&quot;Could not find registered platform with name: \\&quot;cuda\\&quot;. Available platform names are: &quot;)</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-7/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-7/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>Epoch   1	Train Loss: 16.900967	Train Acc: 20.7143%	Val Loss: 7.453496	Val Acc: 26.8000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 8.828428	Train Acc: 23.5714%	Val Loss: 4.012742	Val Acc: 31.2000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 4.781260	Train Acc: 38.5714%	Val Loss: 2.289675	Val Acc: 34.8000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 2.112979	Train Acc: 47.8571%	Val Loss: 2.407374	Val Acc: 34.8000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 2.381186	Train Acc: 50.7143%	Val Loss: 2.453577	Val Acc: 37.8000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.831160	Train Acc: 57.8571%	Val Loss: 2.267815	Val Acc: 43.2000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.600700	Train Acc: 68.5714%	Val Loss: 1.987743	Val Acc: 48.6000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 1.359828	Train Acc: 71.4286%	Val Loss: 1.756406	Val Acc: 56.2000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 1.329730	Train Acc: 75.7143%	Val Loss: 1.649770	Val Acc: 59.6000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 1.151535	Train Acc: 78.5714%	Val Loss: 1.605249	Val Acc: 61.6000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 1.448032	Train Acc: 77.1429%	Val Loss: 1.601787	Val Acc: 63.0000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 1.011211	Train Acc: 77.1429%	Val Loss: 1.602901	Val Acc: 63.0000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.960311	Train Acc: 77.8571%	Val Loss: 1.599380	Val Acc: 63.8000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 1.057467	Train Acc: 78.5714%	Val Loss: 1.591251	Val Acc: 64.0000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 1.806404	Train Acc: 80.0000%	Val Loss: 1.581501	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.860397	Train Acc: 79.2857%	Val Loss: 1.595711	Val Acc: 65.2000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.734396	Train Acc: 80.7143%	Val Loss: 1.618162	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.673426	Train Acc: 82.8571%	Val Loss: 1.651589	Val Acc: 65.4000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.730562	Train Acc: 82.8571%	Val Loss: 1.689287	Val Acc: 64.4000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.744401	Train Acc: 83.5714%	Val Loss: 1.711970	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.724164	Train Acc: 84.2857%	Val Loss: 1.712736	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.721066	Train Acc: 85.0000%	Val Loss: 1.695096	Val Acc: 65.0000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.651210	Train Acc: 85.7143%	Val Loss: 1.672541	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.717768	Train Acc: 85.7143%	Val Loss: 1.668745	Val Acc: 65.4000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.813148	Train Acc: 87.1429%	Val Loss: 1.648924	Val Acc: 65.4000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.546969	Train Acc: 87.8571%	Val Loss: 1.624019	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.516378	Train Acc: 89.2857%	Val Loss: 1.598994	Val Acc: 65.0000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.479124	Train Acc: 89.2857%	Val Loss: 1.581874	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  29	Train Loss: 0.469026	Train Acc: 90.0000%	Val Loss: 1.567386	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  30	Train Loss: 0.490631	Train Acc: 90.0000%	Val Loss: 1.559431	Val Acc: 66.0000%</span></span>
<span class="line"><span>Epoch  31	Train Loss: 0.480068	Train Acc: 90.0000%	Val Loss: 1.558629	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  32	Train Loss: 0.474387	Train Acc: 90.0000%	Val Loss: 1.559061	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  33	Train Loss: 0.473973	Train Acc: 91.4286%	Val Loss: 1.566357	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  34	Train Loss: 0.590703	Train Acc: 92.1429%	Val Loss: 1.578132	Val Acc: 66.0000%</span></span>
<span class="line"><span>Epoch  35	Train Loss: 0.494798	Train Acc: 92.8571%	Val Loss: 1.590551	Val Acc: 66.2000%</span></span>
<span class="line"><span>Epoch  36	Train Loss: 0.493164	Train Acc: 93.5714%	Val Loss: 1.595978	Val Acc: 66.0000%</span></span>
<span class="line"><span>Epoch  37	Train Loss: 0.415880	Train Acc: 93.5714%	Val Loss: 1.601474	Val Acc: 65.8000%</span></span>
<span class="line"><span>Epoch  38	Train Loss: 0.422072	Train Acc: 93.5714%	Val Loss: 1.607498	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  39	Train Loss: 0.434265	Train Acc: 92.8571%	Val Loss: 1.617870	Val Acc: 65.8000%</span></span>
<span class="line"><span>Epoch  40	Train Loss: 0.374637	Train Acc: 92.8571%	Val Loss: 1.631233	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  41	Train Loss: 0.402360	Train Acc: 92.8571%	Val Loss: 1.645928	Val Acc: 66.0000%</span></span>
<span class="line"><span>Epoch  42	Train Loss: 0.465095	Train Acc: 93.5714%	Val Loss: 1.671271	Val Acc: 67.0000%</span></span>
<span class="line"><span>Epoch  43	Train Loss: 0.423041	Train Acc: 93.5714%	Val Loss: 1.719895	Val Acc: 67.0000%</span></span>
<span class="line"><span>Epoch  44	Train Loss: 0.720114	Train Acc: 94.2857%	Val Loss: 1.827938	Val Acc: 67.0000%</span></span>
<span class="line"><span>Epoch  45	Train Loss: 0.347155	Train Acc: 92.1429%	Val Loss: 1.936710	Val Acc: 66.6000%</span></span>
<span class="line"><span>Epoch  46	Train Loss: 0.475836	Train Acc: 91.4286%	Val Loss: 2.019416	Val Acc: 65.8000%</span></span>
<span class="line"><span>Epoch  47	Train Loss: 0.463268	Train Acc: 91.4286%	Val Loss: 2.064098	Val Acc: 65.4000%</span></span>
<span class="line"><span>Epoch  48	Train Loss: 0.397085	Train Acc: 90.7143%	Val Loss: 2.081776	Val Acc: 65.2000%</span></span>
<span class="line"><span>Epoch  49	Train Loss: 0.447446	Train Acc: 91.4286%	Val Loss: 2.064368	Val Acc: 65.4000%</span></span>
<span class="line"><span>Epoch  50	Train Loss: 0.476110	Train Acc: 93.5714%	Val Loss: 2.029231	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  51	Train Loss: 0.459895	Train Acc: 95.0000%	Val Loss: 1.977381	Val Acc: 66.2000%</span></span>
<span class="line"><span>Early Stopping at Epoch 51</span></span>
<span class="line"><span>Test Loss: 1.904060	Test Acc: 66.1000%</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,18)])])}const m=a(l,[["render",t]]);export{d as __pageData,m as default};
