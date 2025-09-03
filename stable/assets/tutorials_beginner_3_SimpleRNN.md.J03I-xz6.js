import{_ as a,c as n,o as i,al as p}from"./chunks/framework.CDztWmMz.js";const E=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(t,s,h,k,c,d){return i(),n("div",null,[...s[0]||(s[0]=[p(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><p>Note: If you wish to use <code>AutoZygote()</code> for automatic differentiation, add Zygote to your project dependencies and include <code>using Zygote</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>  13173.7 ms  ? Enzyme</span></span>
<span class="line"><span>  13627.6 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>Info Given Reactant was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0KWARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>\x1B[0KERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>  14897.2 ms  ? Reactant</span></span>
<span class="line"><span>WARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>ERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>Precompiling Enzyme...</span></span>
<span class="line"><span>Info Given Enzyme was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0KWARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>\x1B[0KERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>  13239.6 ms  ? Enzyme</span></span>
<span class="line"><span>WARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>ERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>  13477.6 ms  ? Enzyme</span></span>
<span class="line"><span>    712.3 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    864.5 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    899.6 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    701.6 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>Info Given LuxEnzymeExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    729.4 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    710.8 ms  ? Lux → LuxEnzymeExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeLogExpFunctionsExt...</span></span>
<span class="line"><span>  13046.4 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeLogExpFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    704.4 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeChainRulesCoreExt...</span></span>
<span class="line"><span>  13284.7 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeChainRulesCoreExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    699.9 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeSpecialFunctionsExt...</span></span>
<span class="line"><span>  12960.4 ms  ? Enzyme</span></span>
<span class="line"><span>    732.1 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>Info Given EnzymeSpecialFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    861.9 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeGPUArraysCoreExt...</span></span>
<span class="line"><span>  13060.0 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeGPUArraysCoreExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    685.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeStaticArraysExt...</span></span>
<span class="line"><span>  13450.1 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeStaticArraysExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    892.5 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling OptimisersReactantExt...</span></span>
<span class="line"><span>  13399.3 ms  ? Enzyme</span></span>
<span class="line"><span>    703.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    721.8 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   1963.7 ms  ? Reactant</span></span>
<span class="line"><span>    687.7 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>Info Given OptimisersReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    726.1 ms  ? Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  13065.9 ms  ? Enzyme</span></span>
<span class="line"><span>    705.8 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1951.9 ms  ? Reactant</span></span>
<span class="line"><span>Info Given LuxCoreReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    766.4 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  13113.1 ms  ? Enzyme</span></span>
<span class="line"><span>    683.8 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1879.9 ms  ? Reactant</span></span>
<span class="line"><span>Info Given MLDataDevicesReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    727.2 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  13172.6 ms  ? Enzyme</span></span>
<span class="line"><span>    723.1 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    721.2 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    923.3 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   1982.8 ms  ? Reactant</span></span>
<span class="line"><span>Info Given WeightInitializersReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    759.0 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    754.3 ms  ? WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>    906.1 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantKernelAbstractionsExt...</span></span>
<span class="line"><span>  13196.5 ms  ? Enzyme</span></span>
<span class="line"><span>    712.9 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    889.3 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   1901.9 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantKernelAbstractionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    684.2 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantArrayInterfaceExt...</span></span>
<span class="line"><span>  13729.1 ms  ? Enzyme</span></span>
<span class="line"><span>    716.1 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   2038.0 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantArrayInterfaceExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    687.8 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantSpecialFunctionsExt...</span></span>
<span class="line"><span>  13103.2 ms  ? Enzyme</span></span>
<span class="line"><span>    706.7 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    729.9 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    865.4 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   2052.4 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantSpecialFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    891.3 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantStatisticsExt...</span></span>
<span class="line"><span>  13680.2 ms  ? Enzyme</span></span>
<span class="line"><span>    693.9 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   2044.9 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantStatisticsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    732.6 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxLibReactantExt...</span></span>
<span class="line"><span>  13491.7 ms  ? Enzyme</span></span>
<span class="line"><span>    726.1 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    836.2 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    914.6 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    714.4 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    685.8 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1891.8 ms  ? Reactant</span></span>
<span class="line"><span>    688.0 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    740.3 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>    887.3 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>    748.1 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>    801.9 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>    751.1 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>Info Given LuxLibReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    704.9 ms  ? LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  13380.0 ms  ? Enzyme</span></span>
<span class="line"><span>    732.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    772.3 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    925.9 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   2102.9 ms  ? Reactant</span></span>
<span class="line"><span>    705.6 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    732.6 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>Info Given ReactantNNlibExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>   1024.3 ms  ? Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-cf3b-a57751497ca7 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  13342.6 ms  ? Enzyme</span></span>
<span class="line"><span>    754.4 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    862.0 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    917.2 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    742.6 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    733.6 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    714.3 ms  ? Lux → LuxEnzymeExt</span></span>
<span class="line"><span>   2103.5 ms  ? Reactant</span></span>
<span class="line"><span>    697.1 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    717.3 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>    890.5 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>    735.8 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>    770.7 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>    749.2 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>    760.8 ms  ? Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>    720.0 ms  ? WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>    718.5 ms  ? LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>Info Given LuxReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    728.8 ms  ? Lux → LuxReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-2d10-7b61019616ef is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span></code></pre></div><h2 id="Dataset" tabindex="-1">Dataset <a class="header-anchor" href="#Dataset" aria-label="Permalink to &quot;Dataset {#Dataset}&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> create_dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_data, labels</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data, labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> create_dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size, sequence_length)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a LSTM block and a classifier head.</p><p>We pass the field names <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/stable/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L,C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/stable/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/stable/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T,3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/stable/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T,3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binary cross-entropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), model))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_loader)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ad </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                ad, lossfn, (x, y), train_state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, epoch, total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">            &quot;Validation:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()((train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-15/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>AssertionError(&quot;Could not find registered platform with name: \\&quot;cuda\\&quot;. Available platform names are: &quot;)</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56034</span></span>
<span class="line"><span>Validation:	Loss 0.49995	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47919</span></span>
<span class="line"><span>Validation:	Loss 0.43457	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.41732</span></span>
<span class="line"><span>Validation:	Loss 0.37449	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.36183</span></span>
<span class="line"><span>Validation:	Loss 0.31881	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.30706</span></span>
<span class="line"><span>Validation:	Loss 0.26932	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.26053</span></span>
<span class="line"><span>Validation:	Loss 0.22607	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.21532</span></span>
<span class="line"><span>Validation:	Loss 0.18761	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.18034</span></span>
<span class="line"><span>Validation:	Loss 0.15336	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.14510</span></span>
<span class="line"><span>Validation:	Loss 0.12397	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.11857</span></span>
<span class="line"><span>Validation:	Loss 0.09905	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.09325</span></span>
<span class="line"><span>Validation:	Loss 0.07704	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.07055</span></span>
<span class="line"><span>Validation:	Loss 0.05727	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.05224</span></span>
<span class="line"><span>Validation:	Loss 0.04244	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.03941</span></span>
<span class="line"><span>Validation:	Loss 0.03247	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.03041</span></span>
<span class="line"><span>Validation:	Loss 0.02610	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02476</span></span>
<span class="line"><span>Validation:	Loss 0.02157	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.02061</span></span>
<span class="line"><span>Validation:	Loss 0.01812	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01751</span></span>
<span class="line"><span>Validation:	Loss 0.01541	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01479</span></span>
<span class="line"><span>Validation:	Loss 0.01334	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01293</span></span>
<span class="line"><span>Validation:	Loss 0.01174	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01157</span></span>
<span class="line"><span>Validation:	Loss 0.01048	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01030</span></span>
<span class="line"><span>Validation:	Loss 0.00948	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00935</span></span>
<span class="line"><span>Validation:	Loss 0.00867	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00865</span></span>
<span class="line"><span>Validation:	Loss 0.00799	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00796</span></span>
<span class="line"><span>Validation:	Loss 0.00741	Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-15/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48760</span></span>
<span class="line"><span>Validation:	Loss 0.41910	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34760</span></span>
<span class="line"><span>Validation:	Loss 0.29167	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25437</span></span>
<span class="line"><span>Validation:	Loss 0.23289	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20502</span></span>
<span class="line"><span>Validation:	Loss 0.19176	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16703</span></span>
<span class="line"><span>Validation:	Loss 0.15554	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13056</span></span>
<span class="line"><span>Validation:	Loss 0.11228	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09093</span></span>
<span class="line"><span>Validation:	Loss 0.07573	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06429</span></span>
<span class="line"><span>Validation:	Loss 0.05762	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04965</span></span>
<span class="line"><span>Validation:	Loss 0.04575	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03996</span></span>
<span class="line"><span>Validation:	Loss 0.03729	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03271</span></span>
<span class="line"><span>Validation:	Loss 0.03073	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02712</span></span>
<span class="line"><span>Validation:	Loss 0.02544	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02261</span></span>
<span class="line"><span>Validation:	Loss 0.02104	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01875</span></span>
<span class="line"><span>Validation:	Loss 0.01744	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01578</span></span>
<span class="line"><span>Validation:	Loss 0.01472	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01352</span></span>
<span class="line"><span>Validation:	Loss 0.01279	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01193</span></span>
<span class="line"><span>Validation:	Loss 0.01139	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01072</span></span>
<span class="line"><span>Validation:	Loss 0.01030	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00978</span></span>
<span class="line"><span>Validation:	Loss 0.00939	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00893</span></span>
<span class="line"><span>Validation:	Loss 0.00860	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00820</span></span>
<span class="line"><span>Validation:	Loss 0.00789	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00753</span></span>
<span class="line"><span>Validation:	Loss 0.00725	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00693</span></span>
<span class="line"><span>Validation:	Loss 0.00666	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00637</span></span>
<span class="line"><span>Validation:	Loss 0.00612	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00587</span></span>
<span class="line"><span>Validation:	Loss 0.00563	Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,43)])])}const o=a(l,[["render",e]]);export{E as __pageData,o as default};
