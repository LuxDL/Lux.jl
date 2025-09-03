import{_ as a,c as n,o as i,al as p}from"./chunks/framework.CDztWmMz.js";const r=JSON.parse('{"title":"MNIST Classification with SimpleChains","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/4_SimpleChains.md","filePath":"tutorials/beginner/4_SimpleChains.md","lastUpdated":null}'),e={name:"tutorials/beginner/4_SimpleChains.md"};function t(l,s,h,m,k,u){return i(),n("div",null,[...s[0]||(s[0]=[p(`<h1 id="MNIST-Classification-with-SimpleChains" tabindex="-1">MNIST Classification with SimpleChains <a class="header-anchor" href="#MNIST-Classification-with-SimpleChains" aria-label="Permalink to &quot;MNIST Classification with SimpleChains {#MNIST-Classification-with-SimpleChains}&quot;">​</a></h1><p>SimpleChains.jl is an excellent framework for training small neural networks. In this tutorial we will demonstrate how to use the same API as Lux.jl to train a model using SimpleChains.jl. We will use the tutorial from <a href="https://pumasai.github.io/SimpleChains.jl/dev/examples/mnist/" target="_blank" rel="noreferrer">SimpleChains.jl</a> as a reference.</p><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Statistics, Printf, Reactant</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDatasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MNIST</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SimpleChains</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SimpleChains</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_default_backend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;cpu&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Reactant...</span></span>
<span class="line"><span>  14084.7 ms  ? Enzyme</span></span>
<span class="line"><span>  13657.2 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>Info Given Reactant was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0KWARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>\x1B[0KERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>  14900.0 ms  ? Reactant</span></span>
<span class="line"><span>WARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>ERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>Precompiling Enzyme...</span></span>
<span class="line"><span>Info Given Enzyme was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0KWARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>\x1B[0KERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>  13308.6 ms  ? Enzyme</span></span>
<span class="line"><span>WARNING: Method definition within_autodiff() in module EnzymeCore at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/EnzymeCore/0ptb3/src/EnzymeCore.jl:619 overwritten in module Enzyme at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Enzyme/nqe7m/src/Enzyme.jl:1561.</span></span>
<span class="line"><span>ERROR: Method overwriting is not permitted during Module precompilation. Use \`__precompile__(false)\` to opt-out of precompilation.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>  13255.4 ms  ? Enzyme</span></span>
<span class="line"><span>    772.9 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    914.9 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    915.2 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    730.0 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>Info Given LuxEnzymeExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    716.2 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    712.9 ms  ? Lux → LuxEnzymeExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeLogExpFunctionsExt...</span></span>
<span class="line"><span>  13479.6 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeLogExpFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    728.0 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeChainRulesCoreExt...</span></span>
<span class="line"><span>  13252.1 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeChainRulesCoreExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    738.1 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeSpecialFunctionsExt...</span></span>
<span class="line"><span>  13099.2 ms  ? Enzyme</span></span>
<span class="line"><span>    766.2 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>Info Given EnzymeSpecialFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    893.9 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeGPUArraysCoreExt...</span></span>
<span class="line"><span>  13129.6 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeGPUArraysCoreExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    703.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling EnzymeStaticArraysExt...</span></span>
<span class="line"><span>  13189.8 ms  ? Enzyme</span></span>
<span class="line"><span>Info Given EnzymeStaticArraysExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    907.2 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling OptimisersReactantExt...</span></span>
<span class="line"><span>  13450.9 ms  ? Enzyme</span></span>
<span class="line"><span>    727.2 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    740.2 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>   2003.3 ms  ? Reactant</span></span>
<span class="line"><span>    715.9 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>Info Given OptimisersReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    826.6 ms  ? Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxCoreReactantExt...</span></span>
<span class="line"><span>  13240.2 ms  ? Enzyme</span></span>
<span class="line"><span>    711.7 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   2029.1 ms  ? Reactant</span></span>
<span class="line"><span>Info Given LuxCoreReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    764.6 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling MLDataDevicesReactantExt...</span></span>
<span class="line"><span>  13668.2 ms  ? Enzyme</span></span>
<span class="line"><span>    708.6 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   2113.6 ms  ? Reactant</span></span>
<span class="line"><span>Info Given MLDataDevicesReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    738.3 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling WeightInitializersReactantExt...</span></span>
<span class="line"><span>  13612.9 ms  ? Enzyme</span></span>
<span class="line"><span>    729.1 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    735.6 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    858.7 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   1859.7 ms  ? Reactant</span></span>
<span class="line"><span>    712.8 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>Info Given WeightInitializersReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    727.3 ms  ? WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>    837.5 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantAbstractFFTsExt...</span></span>
<span class="line"><span>  13548.2 ms  ? Enzyme</span></span>
<span class="line"><span>    711.9 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1920.6 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantAbstractFFTsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    710.8 ms  ? Reactant → ReactantAbstractFFTsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantKernelAbstractionsExt...</span></span>
<span class="line"><span>  13263.5 ms  ? Enzyme</span></span>
<span class="line"><span>    730.9 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    904.6 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   1896.4 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantKernelAbstractionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    725.4 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantArrayInterfaceExt...</span></span>
<span class="line"><span>  13292.9 ms  ? Enzyme</span></span>
<span class="line"><span>    686.8 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1969.8 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantArrayInterfaceExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    687.8 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantSpecialFunctionsExt...</span></span>
<span class="line"><span>  13204.8 ms  ? Enzyme</span></span>
<span class="line"><span>    737.1 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    752.7 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    850.4 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>   1980.4 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantSpecialFunctionsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    843.5 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantFillArraysExt...</span></span>
<span class="line"><span>  13221.0 ms  ? Enzyme</span></span>
<span class="line"><span>    687.2 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   1889.1 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantFillArraysExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    744.1 ms  ? Reactant → ReactantFillArraysExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantStatisticsExt...</span></span>
<span class="line"><span>  13291.3 ms  ? Enzyme</span></span>
<span class="line"><span>    690.9 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   2004.8 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantStatisticsExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    714.4 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantOneHotArraysExt...</span></span>
<span class="line"><span>  13112.3 ms  ? Enzyme</span></span>
<span class="line"><span>    715.8 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    745.5 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    929.1 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   1877.3 ms  ? Reactant</span></span>
<span class="line"><span>    677.6 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    729.6 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>Info Given ReactantOneHotArraysExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>   1065.5 ms  ? Reactant → ReactantOneHotArraysExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxLibReactantExt...</span></span>
<span class="line"><span>  13303.1 ms  ? Enzyme</span></span>
<span class="line"><span>    755.8 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    843.3 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    945.8 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    692.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    766.7 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>   1907.3 ms  ? Reactant</span></span>
<span class="line"><span>    670.0 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    702.5 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>    854.1 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>    705.3 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>    816.4 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>    758.3 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>Info Given LuxLibReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    708.3 ms  ? LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantNNlibExt...</span></span>
<span class="line"><span>  13434.5 ms  ? Enzyme</span></span>
<span class="line"><span>    713.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    740.1 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    921.0 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>   1959.8 ms  ? Reactant</span></span>
<span class="line"><span>    709.9 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>    716.9 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>Info Given ReactantNNlibExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>   1043.8 ms  ? Reactant → ReactantNNlibExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  13513.6 ms  ? Enzyme</span></span>
<span class="line"><span>    720.3 ms  ? Enzyme → EnzymeChainRulesCoreExt</span></span>
<span class="line"><span>    864.1 ms  ? Enzyme → EnzymeSpecialFunctionsExt</span></span>
<span class="line"><span>    904.2 ms  ? Enzyme → EnzymeStaticArraysExt</span></span>
<span class="line"><span>    731.6 ms  ? Enzyme → EnzymeLogExpFunctionsExt</span></span>
<span class="line"><span>    732.6 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>    731.1 ms  ? Lux → LuxEnzymeExt</span></span>
<span class="line"><span>   1900.9 ms  ? Reactant</span></span>
<span class="line"><span>    713.1 ms  ? Reactant → ReactantStatisticsExt</span></span>
<span class="line"><span>    736.3 ms  ? Reactant → ReactantKernelAbstractionsExt</span></span>
<span class="line"><span>    905.0 ms  ? Reactant → ReactantSpecialFunctionsExt</span></span>
<span class="line"><span>    734.2 ms  ? Reactant → ReactantArrayInterfaceExt</span></span>
<span class="line"><span>    816.8 ms  ? MLDataDevices → MLDataDevicesReactantExt</span></span>
<span class="line"><span>    796.2 ms  ? LuxCore → LuxCoreReactantExt</span></span>
<span class="line"><span>    762.2 ms  ? Optimisers → OptimisersReactantExt</span></span>
<span class="line"><span>    740.4 ms  ? WeightInitializers → WeightInitializersReactantExt</span></span>
<span class="line"><span>    713.8 ms  ? LuxLib → LuxLibReactantExt</span></span>
<span class="line"><span>Info Given LuxReactantExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    716.4 ms  ? Lux → LuxReactantExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Enzyme with build ID ffffffff-ffff-ffff-5810-c42874ba469c is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Enzyme [7da242da-08ed-463a-9acd-ee780be4f1d9] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>Precompiling ReactantOffsetArraysExt...</span></span>
<span class="line"><span>  13598.8 ms  ? Enzyme</span></span>
<span class="line"><span>    712.5 ms  ? Enzyme → EnzymeGPUArraysCoreExt</span></span>
<span class="line"><span>   2038.5 ms  ? Reactant</span></span>
<span class="line"><span>Info Given ReactantOffsetArraysExt was explicitly requested, output will be shown live \x1B[0K</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[0K\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>    740.7 ms  ? Reactant → ReactantOffsetArraysExt</span></span>
<span class="line"><span>\x1B[33m\x1B[1m┌ \x1B[22m\x1B[39m\x1B[33m\x1B[1mWarning: \x1B[22m\x1B[39mModule Reactant with build ID ffffffff-ffff-ffff-db1e-5e370382e699 is missing from the cache.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m│ \x1B[22m\x1B[39mThis may mean Reactant [3c362404-f566-11ee-1572-e11a4b42c853] does not support precompilation but is imported by a module that does.</span></span>
<span class="line"><span>\x1B[33m\x1B[1m└ \x1B[22m\x1B[39m\x1B[90m@ Base loading.jl:2541\x1B[39m</span></span>
<span class="line"><span>AssertionError(&quot;Could not find registered platform with name: \\&quot;cuda\\&quot;. Available platform names are: &quot;)</span></span></code></pre></div><h2 id="Loading-MNIST" tabindex="-1">Loading MNIST <a class="header-anchor" href="#Loading-MNIST" aria-label="Permalink to &quot;Loading MNIST {#Loading-MNIST}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadmnist</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(batchsize, train_split)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Load MNIST</span></span>
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
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Process images into (H, W, C, BS) batches</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imgs, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(labels_raw, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_test, y_test) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, y_data); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">train_split)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the test data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_test, y_test)); batchsize, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Define-the-Model" tabindex="-1">Define the Model <a class="header-anchor" href="#Define-the-Model" aria-label="Permalink to &quot;Define the Model {#Define-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lux_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    MaxPool</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">6</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    MaxPool</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    FlattenLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 84</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">84</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Chain(</span></span>
<span class="line"><span>    layer_1 = Conv((5, 5), 1 =&gt; 6, relu),         # 156 parameters</span></span>
<span class="line"><span>    layer_2 = MaxPool((2, 2)),</span></span>
<span class="line"><span>    layer_3 = Conv((5, 5), 6 =&gt; 16, relu),        # 2_416 parameters</span></span>
<span class="line"><span>    layer_4 = MaxPool((2, 2)),</span></span>
<span class="line"><span>    layer_5 = Lux.FlattenLayer{Static.StaticInt{3}}(static(3)),</span></span>
<span class="line"><span>    layer_6 = Chain(</span></span>
<span class="line"><span>        layer_1 = Dense(256 =&gt; 128, relu),        # 32_896 parameters</span></span>
<span class="line"><span>        layer_2 = Dense(128 =&gt; 84, relu),         # 10_836 parameters</span></span>
<span class="line"><span>        layer_3 = Dense(84 =&gt; 10),                # 850 parameters</span></span>
<span class="line"><span>    ),</span></span>
<span class="line"><span>)         # Total: 47_154 parameters,</span></span>
<span class="line"><span>          #        plus 0 states.</span></span></code></pre></div><p>We now need to convert the lux_model to SimpleChains.jl. We need to do this by defining the <a href="/stable/api/Lux/interop#Lux.ToSimpleChainsAdaptor"><code>ToSimpleChainsAdaptor</code></a> and providing the input dimensions.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">adaptor </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ToSimpleChainsAdaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">simple_chains_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> adaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lux_model)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SimpleChainsLayer(</span></span>
<span class="line"><span>    Chain(</span></span>
<span class="line"><span>        layer_1 = Conv((5, 5), 1 =&gt; 6, relu),     # 156 parameters</span></span>
<span class="line"><span>        layer_2 = MaxPool((2, 2)),</span></span>
<span class="line"><span>        layer_3 = Conv((5, 5), 6 =&gt; 16, relu),    # 2_416 parameters</span></span>
<span class="line"><span>        layer_4 = MaxPool((2, 2)),</span></span>
<span class="line"><span>        layer_5 = Lux.FlattenLayer{Static.StaticInt{3}}(static(3)),</span></span>
<span class="line"><span>        layer_6 = Chain(</span></span>
<span class="line"><span>            layer_1 = Dense(256 =&gt; 128, relu),    # 32_896 parameters</span></span>
<span class="line"><span>            layer_2 = Dense(128 =&gt; 84, relu),     # 10_836 parameters</span></span>
<span class="line"><span>            layer_3 = Dense(84 =&gt; 10),            # 850 parameters</span></span>
<span class="line"><span>        ),</span></span>
<span class="line"><span>    ),</span></span>
<span class="line"><span>)         # Total: 47_154 parameters,</span></span>
<span class="line"><span>          #        plus 0 states.</span></span></code></pre></div><h2 id="Helper-Functions" tabindex="-1">Helper Functions <a class="header-anchor" href="#Helper-Functions" aria-label="Permalink to &quot;Helper Functions {#Helper-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; logits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, dataloader)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    total_correct, total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        predicted_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st))))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(target_class </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> predicted_class)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(target_class)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Define-the-Training-Loop" tabindex="-1">Define the Training Loop <a class="header-anchor" href="#Define-the-Training-Loop" aria-label="Permalink to &quot;Define the Training Loop {#Define-the-Training-Loop}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, dev</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(); rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_dataloader, test_dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loadmnist</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    vjp </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3.0f-4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_ra </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(test_dataloader)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_ra, ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        tr_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                model_compiled, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, train_dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                model_compiled, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states, test_dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;[%2d/%2d] </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Time %.2fs </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Training Accuracy: %.2f%% </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Test Accuracy: \\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 %.2f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch nepochs ttime tr_acc te_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> tr_acc, te_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Finally-Training-the-Model" tabindex="-1">Finally Training the Model <a class="header-anchor" href="#Finally-Training-the-Model" aria-label="Permalink to &quot;Finally Training the Model {#Finally-Training-the-Model}&quot;">​</a></h2><p>First we will train the Lux model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tr_acc, te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lux_model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ 1/10] 	 Time 458.19s 	 Training Accuracy: 12.34% 	 Test Accuracy: 11.72%</span></span>
<span class="line"><span>[ 2/10] 	 Time 0.10s 	 Training Accuracy: 23.52% 	 Test Accuracy: 25.78%</span></span>
<span class="line"><span>[ 3/10] 	 Time 0.09s 	 Training Accuracy: 29.45% 	 Test Accuracy: 28.12%</span></span>
<span class="line"><span>[ 4/10] 	 Time 0.09s 	 Training Accuracy: 38.44% 	 Test Accuracy: 39.84%</span></span>
<span class="line"><span>[ 5/10] 	 Time 0.09s 	 Training Accuracy: 48.91% 	 Test Accuracy: 48.44%</span></span>
<span class="line"><span>[ 6/10] 	 Time 0.09s 	 Training Accuracy: 58.20% 	 Test Accuracy: 53.91%</span></span>
<span class="line"><span>[ 7/10] 	 Time 0.09s 	 Training Accuracy: 64.30% 	 Test Accuracy: 60.94%</span></span>
<span class="line"><span>[ 8/10] 	 Time 0.09s 	 Training Accuracy: 69.84% 	 Test Accuracy: 64.06%</span></span>
<span class="line"><span>[ 9/10] 	 Time 0.09s 	 Training Accuracy: 74.53% 	 Test Accuracy: 71.09%</span></span>
<span class="line"><span>[10/10] 	 Time 0.08s 	 Training Accuracy: 78.59% 	 Test Accuracy: 72.66%</span></span></code></pre></div><p>Now we will train the SimpleChains model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tr_acc, te_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(simple_chains_model)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ 1/10] 	 Time 909.02s 	 Training Accuracy: 26.17% 	 Test Accuracy: 28.91%</span></span>
<span class="line"><span>[ 2/10] 	 Time 12.40s 	 Training Accuracy: 45.47% 	 Test Accuracy: 44.53%</span></span>
<span class="line"><span>[ 3/10] 	 Time 12.38s 	 Training Accuracy: 61.56% 	 Test Accuracy: 64.06%</span></span>
<span class="line"><span>[ 4/10] 	 Time 12.37s 	 Training Accuracy: 69.38% 	 Test Accuracy: 67.97%</span></span>
<span class="line"><span>[ 5/10] 	 Time 12.40s 	 Training Accuracy: 71.95% 	 Test Accuracy: 67.19%</span></span>
<span class="line"><span>[ 6/10] 	 Time 12.39s 	 Training Accuracy: 77.27% 	 Test Accuracy: 71.09%</span></span>
<span class="line"><span>[ 7/10] 	 Time 12.42s 	 Training Accuracy: 79.53% 	 Test Accuracy: 75.00%</span></span>
<span class="line"><span>[ 8/10] 	 Time 12.57s 	 Training Accuracy: 82.50% 	 Test Accuracy: 74.22%</span></span>
<span class="line"><span>[ 9/10] 	 Time 12.38s 	 Training Accuracy: 84.06% 	 Test Accuracy: 75.78%</span></span>
<span class="line"><span>[10/10] 	 Time 12.39s 	 Training Accuracy: 86.02% 	 Test Accuracy: 78.91%</span></span></code></pre></div><p>On my local machine we see a 3-4x speedup when using SimpleChains.jl. The conditions of the server this documentation is being built on is not ideal for CPU benchmarking hence, the speedup may not be as significant and even there might be regressions.</p><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,31)])])}const d=a(e,[["render",t]]);export{r as __pageData,d as default};
