import { _ as _export_sfc, c as createElementBlock, o as openBlock, a4 as createStaticVNode } from "./chunks/framework.Cu_l04oX.js";
const __pageData = JSON.parse('{"title":"LuxTestUtils","description":"","frontmatter":{},"headers":[],"relativePath":"api/Testing_Functionality/LuxTestUtils.md","filePath":"api/Testing_Functionality/LuxTestUtils.md","lastUpdated":null}');
const _sfc_main = { name: "api/Testing_Functionality/LuxTestUtils.md" };
const _hoisted_1 = /* @__PURE__ */ createStaticVNode('<h1 id="LuxTestUtils" tabindex="-1">LuxTestUtils <a class="header-anchor" href="#LuxTestUtils" aria-label="Permalink to &quot;LuxTestUtils {#LuxTestUtils}&quot;">​</a></h1><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>This is a testing package. Hence, we don&#39;t use features like weak dependencies to reduce load times. It is recommended that you exclusively use this package for testing and not add a dependency to it in your main package Project.toml.</p></div><p>Implements utilities for testing <strong>gradient correctness</strong> and <strong>dynamic dispatch</strong> of Lux.jl models.</p><h2 id="Index" tabindex="-1">Index <a class="header-anchor" href="#Index" aria-label="Permalink to &quot;Index {#Index}&quot;">​</a></h2><ul><li><a href="#LuxTestUtils.@jet"><code>LuxTestUtils.@jet</code></a></li><li><a href="#LuxTestUtils.@test_gradients"><code>LuxTestUtils.@test_gradients</code></a></li></ul><h2 id="Testing-using-JET.jl" tabindex="-1">Testing using JET.jl <a class="header-anchor" href="#Testing-using-JET.jl" aria-label="Permalink to &quot;Testing using JET.jl {#Testing-using-JET.jl}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxTestUtils.@jet" href="#LuxTestUtils.@jet">#</a> <b><u>LuxTestUtils.@jet</u></b> — <i>Macro</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@jet</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> f</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) call_broken</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> opt_broken</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span></span></code></pre></div><p>Run JET tests on the function <code>f</code> with the arguments <code>args...</code>. If <code>JET</code> fails to compile or julia version is &lt; 1.7, then the macro will be a no-op.</p><p><strong>Keyword Arguments</strong></p><ul><li><p><code>call_broken</code>: Marks the test_call as broken.</p></li><li><p><code>opt_broken</code>: Marks the test_opt as broken.</p></li></ul><p>All additional arguments will be forwarded to <code>@JET.test_call</code> and <code>@JET.test_opt</code>.</p><div class="tip custom-block"><p class="custom-block-title">TIP</p><p>Instead of specifying <code>target_modules</code> with every call, you can set preferences for <code>target_modules</code> using <code>Preferences.jl</code>. For example, to set <code>target_modules</code> to <code>(Lux, LuxLib)</code> we can run:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Preferences</span></span>\n<span class="line"></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_preferences!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Base</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">UUID</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;ac9de150-d08f-4546-94fb-7472b5760531&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>\n<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;target_modules&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Lux&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;LuxLib&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span></code></pre></div></div><p><strong>Example</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxTestUtils</span></span>\n<span class="line"></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@testset</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Showcase JET Testing&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> begin</span></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @jet</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]) target_modules</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Base, Core)</span></span>\n<span class="line"></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @jet</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) target_modules</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Base, Core) opt_broken</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span></span>\n<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p><a href="https://github.com/LuxDL/LuxTestUtils.jl/blob/v0.1.15/src/LuxTestUtils.jl#L22-L61" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Gradient-Correctness" tabindex="-1">Gradient Correctness <a class="header-anchor" href="#Gradient-Correctness" aria-label="Permalink to &quot;Gradient Correctness {#Gradient-Correctness}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxTestUtils.@test_gradients" href="#LuxTestUtils.@test_gradients">#</a> <b><u>LuxTestUtils.@test_gradients</u></b> — <i>Macro</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@test_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> f args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span></code></pre></div><p>Compare the gradients computed by Zygote.jl (Reverse Mode AD) against:</p><ul><li><p>Tracker.jl (Reverse Mode AD)</p></li><li><p>ReverseDiff.jl (Reverse Mode AD)</p></li><li><p>ForwardDiff.jl (Forward Mode AD)</p></li><li><p>FiniteDifferences.jl (Finite Differences)</p></li></ul><div class="tip custom-block"><p class="custom-block-title">TIP</p><p>This function is completely compatible with Test.jl</p></div><p><strong>Arguments</strong></p><ul><li><p><code>f</code>: The function to test.</p></li><li><p><code>args...</code>: Inputs to <code>f</code> wrt which the gradients are computed.</p></li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>gpu_testing</code>: Disables ForwardDiff, ReverseDiff and FiniteDifferences tests. (Default: <code>false</code>)</p></li><li><p><code>soft_fail</code>: If <code>true</code>, the test will not fail if any of the gradients are incorrect, instead it will show up as broken. (Default: <code>false</code>)</p></li><li><p><code>skip_(tracker|reverse_diff|forward_diff|finite_differences)</code>: Skip the corresponding gradient computation and check. (Default: <code>false</code>)</p></li><li><p><code>large_arrays_skip_(forward_diff|finite_differences)</code>: Skip the corresponding gradient computation and check for large arrays. (Forward Mode and Finite Differences are not efficient for large arrays.) (Default: <code>true</code>)</p></li><li><p><code>large_array_length</code>: The length of the array above which the gradient computation is considered large. (Default: 25)</p></li><li><p><code>max_total_array_size</code>: Treat as large array if the total size of all arrays is greater than this value. (Default: 100)</p></li><li><p><code>(tracker|reverse_diff|forward_diff|finite_differences)_broken</code>: Mark the corresponding gradient test as broken. (Default: <code>false</code>)</p></li></ul><p><strong>Keyword Arguments for <code>check_approx</code></strong></p><ul><li><p><code>atol</code>: Absolute tolerance for gradient comparisons. (Default: <code>0.0</code>)</p></li><li><p><code>rtol</code>: Relative tolerance for gradient comparisons. (Default: <code>atol &gt; 0 ? 0.0 : √eps(typeof(atol))</code>)</p></li><li><p><code>nans</code>: Whether or not NaNs are considered equal. (Default: <code>false</code>)</p></li></ul><p><strong>Example</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxTestUtils</span></span>\n<span class="line"></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>\n<span class="line"></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@testset</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Showcase Gradient Testing&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> begin</span></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @test_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> sum abs2 x</span></span>\n<span class="line"></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @test_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> prod x</span></span>\n<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p><a href="https://github.com/LuxDL/LuxTestUtils.jl/blob/v0.1.15/src/LuxTestUtils.jl#L156-L215" target="_blank" rel="noreferrer">source</a></p></div><br>', 11);
const _hoisted_12 = [
  _hoisted_1
];
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, _hoisted_12);
}
const LuxTestUtils = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  LuxTestUtils as default
};
