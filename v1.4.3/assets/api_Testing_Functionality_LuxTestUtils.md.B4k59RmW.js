import{_ as n,c as d,a2 as e,j as t,a as i,G as l,B as h,o as p}from"./chunks/framework.DtydgIfp.js";const C=JSON.parse('{"title":"LuxTestUtils","description":"","frontmatter":{},"headers":[],"relativePath":"api/Testing_Functionality/LuxTestUtils.md","filePath":"api/Testing_Functionality/LuxTestUtils.md","lastUpdated":null}'),k={name:"api/Testing_Functionality/LuxTestUtils.md"},r={class:"jldocstring custom-block"},o={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},c={class:"jldocstring custom-block"},E={class:"jldocstring custom-block"};function u(y,s,f,b,x,F){const a=h("Badge");return p(),d("div",null,[s[15]||(s[15]=e('<h1 id="luxtestutils" tabindex="-1">LuxTestUtils <a class="header-anchor" href="#luxtestutils" aria-label="Permalink to &quot;LuxTestUtils&quot;">​</a></h1><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>This is a testing package. Hence, we don&#39;t use features like weak dependencies to reduce load times. It is recommended that you exclusively use this package for testing and not add a dependency to it in your main package Project.toml.</p></div><p>Implements utilities for testing <strong>gradient correctness</strong> and <strong>dynamic dispatch</strong> of Lux.jl models.</p><h2 id="Testing-using-JET.jl" tabindex="-1">Testing using JET.jl <a class="header-anchor" href="#Testing-using-JET.jl" aria-label="Permalink to &quot;Testing using JET.jl {#Testing-using-JET.jl}&quot;">​</a></h2>',4)),t("details",r,[t("summary",null,[s[0]||(s[0]=t("a",{id:"LuxTestUtils.@jet",href:"#LuxTestUtils.@jet"},[t("span",{class:"jlbinding"},"LuxTestUtils.@jet")],-1)),s[1]||(s[1]=i()),l(a,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),s[2]||(s[2]=e(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@jet</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> f</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) call_broken</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> opt_broken</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span></span></code></pre></div><p>Run JET tests on the function <code>f</code> with the arguments <code>args...</code>. If <code>JET.jl</code> fails to compile, then the macro will be a no-op.</p><p><strong>Keyword Arguments</strong></p><ul><li><p><code>call_broken</code>: Marks the test_call as broken.</p></li><li><p><code>opt_broken</code>: Marks the test_opt as broken.</p></li></ul><p>All additional arguments will be forwarded to <code>JET.@test_call</code> and <code>JET.@test_opt</code>.</p><div class="tip custom-block"><p class="custom-block-title">Tip</p><p>Instead of specifying <code>target_modules</code> with every call, you can set global target modules using <a href="/v1.4.3/api/Testing_Functionality/LuxTestUtils#LuxTestUtils.jet_target_modules!"><code>jet_target_modules!</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxTestUtils</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">jet_target_modules!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Lux&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;LuxLib&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Expects Lux and LuxLib to be present in the module calling \`@jet\`</span></span></code></pre></div></div><p><strong>Example</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @jet</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]) target_modules</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Base, Core)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Test Passed</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @jet</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) target_modules</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Base, Core) opt_broken</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> call_broken</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Test Broken</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">  Expression</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> #= REPL[21]:1 =#</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> JET</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@test_opt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> target_modules </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (Base, Core) </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p><a href="https://github.com/LuxDL/Lux.jl/blob/3c3a43266f688f5d7eed6fe1ca61e2ad2073dbe4/lib/LuxTestUtils/src/jet.jl#L20-L54" target="_blank" rel="noreferrer">source</a></p>`,9))]),t("details",o,[t("summary",null,[s[3]||(s[3]=t("a",{id:"LuxTestUtils.jet_target_modules!",href:"#LuxTestUtils.jet_target_modules!"},[t("span",{class:"jlbinding"},"LuxTestUtils.jet_target_modules!")],-1)),s[4]||(s[4]=i()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[5]||(s[5]=e('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">jet_target_modules!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(list</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Vector{String}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; force</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>This sets <code>target_modules</code> for all JET tests when using <a href="/v1.4.3/api/Testing_Functionality/LuxTestUtils#LuxTestUtils.@jet"><code>@jet</code></a>.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/3c3a43266f688f5d7eed6fe1ca61e2ad2073dbe4/lib/LuxTestUtils/src/jet.jl#L4-L8" target="_blank" rel="noreferrer">source</a></p>',3))]),s[16]||(s[16]=t("h2",{id:"Gradient-Correctness",tabindex:"-1"},[i("Gradient Correctness "),t("a",{class:"header-anchor",href:"#Gradient-Correctness","aria-label":'Permalink to "Gradient Correctness {#Gradient-Correctness}"'},"​")],-1)),t("details",g,[t("summary",null,[s[6]||(s[6]=t("a",{id:"LuxTestUtils.test_gradients",href:"#LuxTestUtils.test_gradients"},[t("span",{class:"jlbinding"},"LuxTestUtils.test_gradients")],-1)),s[7]||(s[7]=i()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[8]||(s[8]=e(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">test_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f, args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; skip_backends</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[], broken_backends</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[], kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Test the gradients of <code>f</code> with respect to <code>args</code> using the specified backends.</p><table tabindex="0"><thead><tr><th style="text-align:left;">Backend</th><th style="text-align:left;">ADType</th><th style="text-align:left;">CPU</th><th style="text-align:left;">GPU</th><th style="text-align:left;">Notes</th></tr></thead><tbody><tr><td style="text-align:left;">Zygote.jl</td><td style="text-align:left;"><code>AutoZygote()</code></td><td style="text-align:left;">✔</td><td style="text-align:left;">✔</td><td style="text-align:left;"></td></tr><tr><td style="text-align:left;">Tracker.jl</td><td style="text-align:left;"><code>AutoTracker()</code></td><td style="text-align:left;">✔</td><td style="text-align:left;">✔</td><td style="text-align:left;"></td></tr><tr><td style="text-align:left;">ReverseDiff.jl</td><td style="text-align:left;"><code>AutoReverseDiff()</code></td><td style="text-align:left;">✔</td><td style="text-align:left;">✖</td><td style="text-align:left;"></td></tr><tr><td style="text-align:left;">ForwardDiff.jl</td><td style="text-align:left;"><code>AutoForwardDiff()</code></td><td style="text-align:left;">✔</td><td style="text-align:left;">✖</td><td style="text-align:left;"><code>len ≤ 100</code></td></tr><tr><td style="text-align:left;">FiniteDiff.jl</td><td style="text-align:left;"><code>AutoFiniteDiff()</code></td><td style="text-align:left;">✔</td><td style="text-align:left;">✖</td><td style="text-align:left;"><code>len ≤ 100</code></td></tr><tr><td style="text-align:left;">Enzyme.jl</td><td style="text-align:left;"><code>AutoEnzyme()</code></td><td style="text-align:left;">✔</td><td style="text-align:left;">✖</td><td style="text-align:left;">Only Reverse Mode</td></tr></tbody></table><p><strong>Arguments</strong></p><ul><li><p><code>f</code>: The function to test the gradients of.</p></li><li><p><code>args</code>: The arguments to test the gradients of. Only <code>AbstractArray</code>s are considered for gradient computation. Gradients wrt all other arguments are assumed to be <code>NoTangent()</code>.</p></li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>skip_backends</code>: A list of backends to skip.</p></li><li><p><code>broken_backends</code>: A list of backends to treat as broken.</p></li><li><p><code>soft_fail</code>: If <code>true</code>, then the test will be recorded as a <code>soft_fail</code> test. This overrides any <code>broken</code> kwargs. Alternatively, a list of backends can be passed to <code>soft_fail</code> to allow soft_fail tests for only those backends.</p></li><li><p><code>enzyme_set_runtime_activity</code>: If <code>true</code>, then activate runtime activity for Enzyme.</p></li><li><p><code>enable_enzyme_reverse_mode</code>: If <code>true</code>, then enable reverse mode for Enzyme.</p></li><li><p><code>kwargs</code>: Additional keyword arguments to pass to <code>check_approx</code>.</p></li></ul><p><strong>Example</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> f</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y, z) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(abs2, y</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">t) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">z)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (; t</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(z</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> test_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p><a href="https://github.com/LuxDL/Lux.jl/blob/3c3a43266f688f5d7eed6fe1ca61e2ad2073dbe4/lib/LuxTestUtils/src/autodiff.jl#L74-L116" target="_blank" rel="noreferrer">source</a></p>`,10))]),t("details",c,[t("summary",null,[s[9]||(s[9]=t("a",{id:"LuxTestUtils.@test_gradients",href:"#LuxTestUtils.@test_gradients"},[t("span",{class:"jlbinding"},"LuxTestUtils.@test_gradients")],-1)),s[10]||(s[10]=i()),l(a,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),s[11]||(s[11]=e('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@test_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f, args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>See the documentation of <a href="/v1.4.3/api/Testing_Functionality/LuxTestUtils#LuxTestUtils.test_gradients"><code>test_gradients</code></a> for more details. This macro provides correct line information for the failing tests.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/3c3a43266f688f5d7eed6fe1ca61e2ad2073dbe4/lib/LuxTestUtils/src/autodiff.jl#L216-L221" target="_blank" rel="noreferrer">source</a></p>',3))]),s[17]||(s[17]=t("h2",{id:"Extensions-to-@test",tabindex:"-1"},[i("Extensions to "),t("code",null,"@test"),i(),t("a",{class:"header-anchor",href:"#Extensions-to-@test","aria-label":'Permalink to "Extensions to `@test` {#Extensions-to-@test}"'},"​")],-1)),t("details",E,[t("summary",null,[s[12]||(s[12]=t("a",{id:"LuxTestUtils.@test_softfail",href:"#LuxTestUtils.@test_softfail"},[t("span",{class:"jlbinding"},"LuxTestUtils.@test_softfail")],-1)),s[13]||(s[13]=i()),l(a,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),s[14]||(s[14]=e('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@test_softfail</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> expr</span></span></code></pre></div><p>Evaluate <code>expr</code> and record a test result. If <code>expr</code> throws an exception, the test result will be recorded as an error. If <code>expr</code> returns a value, and it is not a boolean, the test result will be recorded as an error.</p><p>If the test result is false then the test will be recorded as a broken test, else it will be recorded as a pass.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/3c3a43266f688f5d7eed6fe1ca61e2ad2073dbe4/lib/LuxTestUtils/src/test_softfail.jl#L2-L11" target="_blank" rel="noreferrer">source</a></p>',4))])])}const j=n(k,[["render",u]]);export{C as __pageData,j as default};