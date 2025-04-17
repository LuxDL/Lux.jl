import{_ as s,c as a,a2 as e,o as t}from"./chunks/framework.BgaJsmT1.js";const c=JSON.parse('{"title":"Performance Pitfalls &amp; How to Catch Them","description":"","frontmatter":{},"headers":[],"relativePath":"manual/performance_pitfalls.md","filePath":"manual/performance_pitfalls.md","lastUpdated":null}'),n={name:"manual/performance_pitfalls.md"};function l(p,i,h,o,r,d){return t(),a("div",null,i[0]||(i[0]=[e(`<h1 id="Performance-Pitfalls-and-How-to-Catch-Them" tabindex="-1">Performance Pitfalls &amp; How to Catch Them <a class="header-anchor" href="#Performance-Pitfalls-and-How-to-Catch-Them" aria-label="Permalink to &quot;Performance Pitfalls &amp;amp; How to Catch Them {#Performance-Pitfalls-and-How-to-Catch-Them}&quot;">​</a></h1><p>Go through the following documentations for general performance tips:</p><ol><li><p><a href="https://docs.julialang.org/en/v1/manual/performance-tips/" target="_blank" rel="noreferrer">Official Julia Performance Tips</a>.</p></li><li><p><a href="/v1.2.0/manual/autodiff#autodiff-recommendations">Recommendations for selecting AD packages</a>.</p></li></ol><h2 id="Spurious-Type-Promotion" tabindex="-1">Spurious Type-Promotion <a class="header-anchor" href="#Spurious-Type-Promotion" aria-label="Permalink to &quot;Spurious Type-Promotion {#Spurious-Type-Promotion}&quot;">​</a></h2><p>Lux by-default uses Julia semantics for type-promotions, while this means that we do the &quot;correct&quot; numerical thing, this can often come as a surprise to users coming from a more deep learning background. For example, consider the following code:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, Random</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, gelu)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">recursive_eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((ps, st))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Float32</span></span></code></pre></div><p>As we can see that <code>ps</code> and <code>st</code> are structures with the highest precision being <code>Float32</code>. Now let&#39;s run the model using some random data:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Float64</span></span></code></pre></div><p>Oops our output became <code>Float64</code>. This will be bad on CPUs but an absolute performance disaster on GPUs. The reason this happened is that our input <code>x</code> was <code>Float64</code>. Instead, we should have used <code>Float32</code> input:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Float32</span></span></code></pre></div><p>This was easy to fix for a small model. But certain layers might incorrectly promote objects to a higher precision. This will cause a regression in performance. There are 2 recommendations to fix this or track them down:</p><ol><li><p>Use <a href="/v1.2.0/manual/debugging#debug-lux-layers"><code>Lux.Experimental.@debug_mode</code></a> to see which layer is causing the type-promotion.</p></li><li><p>Alternatively to control the global behavior of eltypes in Lux and allow it to auto-correct the precision use <a href="/v1.2.0/api/Lux/utilities#Lux.match_eltype"><code>match_eltype</code></a> and the <a href="/v1.2.0/manual/preferences#automatic-eltypes-preference"><code>eltype_mismatch_handling</code></a> preference.</p></li></ol><h2 id="Scalar-Indexing-on-GPU-Arrays" tabindex="-1">Scalar Indexing on GPU Arrays <a class="header-anchor" href="#Scalar-Indexing-on-GPU-Arrays" aria-label="Permalink to &quot;Scalar Indexing on GPU Arrays {#Scalar-Indexing-on-GPU-Arrays}&quot;">​</a></h2><p>When running code on GPUs, it is recommended to <a href="https://cuda.juliagpu.org/stable/usage/workflow/#UsageWorkflowScalar" target="_blank" rel="noreferrer">disallow scalar indexing</a>. Note that this is disabled by default except in REPL. You can disable it even in REPL mode using:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> GPUArraysCore</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">GPUArraysCore</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">allowscalar</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><h2 id="Type-Instabilities" tabindex="-1">Type Instabilities <a class="header-anchor" href="#Type-Instabilities" aria-label="Permalink to &quot;Type Instabilities {#Type-Instabilities}&quot;">​</a></h2><p><code>Lux.jl</code> is integrated with <code>DispatchDoctor.jl</code> to catch type instabilities. You can easily enable it by setting the <code>instability_check</code> preference. This will help you catch type instabilities in your code. For more information on how to set preferences, check out <a href="/v1.2.0/api/Lux/utilities#Lux.set_dispatch_doctor_preferences!"><code>Lux.set_dispatch_doctor_preferences!</code></a>.</p><h2 id="Faster-Primitives" tabindex="-1">Faster Primitives <a class="header-anchor" href="#Faster-Primitives" aria-label="Permalink to &quot;Faster Primitives {#Faster-Primitives}&quot;">​</a></h2><p>Prefer to use deep learning primitives and their fused variants from <code>LuxLib.jl</code> instead of <code>NNlib.jl</code>. Some of the alternatives are:</p><ol><li><p>Replace <code>NNlib.batched_mul</code> with <a href="/v1.2.0/api/Building_Blocks/LuxLib#LuxLib.API.batched_matmul"><code>LuxLib.batched_matmul</code></a>.</p></li><li><p>Replace <code>NNlib.conv</code> with bias and activation with <a href="/v1.2.0/api/Building_Blocks/LuxLib#LuxLib.API.fused_conv_bias_activation"><code>LuxLib.fused_conv_bias_activation</code></a>.</p></li><li><p>Replace <code>σ.(w * x .+ b)</code> with <a href="/v1.2.0/api/Building_Blocks/LuxLib#LuxLib.API.fused_dense_bias_activation"><code>LuxLib.fused_dense_bias_activation</code></a>.</p></li><li><p>Replace uses of <code>σ.(x)</code> with <a href="/v1.2.0/api/Building_Blocks/LuxLib#LuxLib.API.fast_activation"><code>LuxLib.fast_activation</code></a> or <a href="/v1.2.0/api/Building_Blocks/LuxLib#LuxLib.API.fast_activation!!"><code>LuxLib.fast_activation!!</code></a> (the latter one is often faster).</p></li><li><p>Replace uses of <code>σ.(x .+ b)</code> with <a href="/v1.2.0/api/Building_Blocks/LuxLib#LuxLib.API.bias_activation"><code>LuxLib.bias_activation</code></a> or <a href="/v1.2.0/api/Building_Blocks/LuxLib#LuxLib.API.bias_activation!!"><code>LuxLib.bias_activation!!</code></a> (the latter one is often faster).</p></li></ol><h2 id="Optional-Dependencies-for-Performance" tabindex="-1">Optional Dependencies for Performance <a class="header-anchor" href="#Optional-Dependencies-for-Performance" aria-label="Permalink to &quot;Optional Dependencies for Performance {#Optional-Dependencies-for-Performance}&quot;">​</a></h2><p>For faster performance on CPUs load the following packages:</p><ol><li><p><code>LoopVectorization.jl</code></p></li><li><p><code>Octavian.jl</code></p></li></ol><p>If these are available, we automatically use optimized versions of the layers. Though there are cases where this might be an issue (see <a href="https://github.com/LuxDL/Lux.jl/issues/980" target="_blank" rel="noreferrer">#980</a> and <a href="/v1.2.0/manual/preferences#disable_loop_vectorization">disabling loop vectorization</a>).</p><h2 id="Data-Loading-and-Device-Transfer" tabindex="-1">Data Loading and Device Transfer <a class="header-anchor" href="#Data-Loading-and-Device-Transfer" aria-label="Permalink to &quot;Data Loading and Device Transfer {#Data-Loading-and-Device-Transfer}&quot;">​</a></h2><p>A common pattern for loading data and transferring data to GPUs looks like this:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset; parallel</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">12</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)  </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># from MLUtils.jl</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (X, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gdev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gdev</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # ...</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # do some computation</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # ...</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>This is typically fast enough, but the data transfer to the device is happening in main process, not exploiting the parallelism in the dataloader. Instead, we can do this:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset; parallel</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">12</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)  </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># from MLUtils.jl</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (X, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataloader)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # ...</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # do some computation</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # ...</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>Here, <code>X</code> and <code>y</code> are on the gpu device <code>gdev</code> and the data transfer happens in the worker processes. Additionally, it behaves similar to <code>CuIterator</code> from CUDA.jl and eagerly frees the data after every iteration (this is device agnostic and works on all supported GPU backends).</p>`,33)]))}const g=s(n,[["render",l]]);export{c as __pageData,g as default};
