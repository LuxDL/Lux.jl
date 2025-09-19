import { _ as _export_sfc, c as createElementBlock, l as createBaseVNode, a as createTextVNode, a4 as createStaticVNode, o as openBlock } from "./chunks/framework.nJ_0eXy8.js";
const __pageData = JSON.parse('{"title":"LuxLib","description":"","frontmatter":{},"headers":[],"relativePath":"api/Building_Blocks/LuxLib.md","filePath":"api/Building_Blocks/LuxLib.md","lastUpdated":null}');
const _sfc_main = { name: "api/Building_Blocks/LuxLib.md" };
const _hoisted_1 = /* @__PURE__ */ createStaticVNode('<h1 id="LuxLib" tabindex="-1">LuxLib <a class="header-anchor" href="#LuxLib" aria-label="Permalink to &quot;LuxLib {#LuxLib}&quot;">​</a></h1><p>Backend for Lux.jl</p><h2 id="Index" tabindex="-1">Index <a class="header-anchor" href="#Index" aria-label="Permalink to &quot;Index {#Index}&quot;">​</a></h2><ul><li><a href="#LuxLib.alpha_dropout"><code>LuxLib.alpha_dropout</code></a></li><li><a href="#LuxLib.batchnorm"><code>LuxLib.batchnorm</code></a></li><li><a href="#LuxLib.dropout"><code>LuxLib.dropout</code></a></li><li><a href="#LuxLib.fast_activation!!"><code>LuxLib.fast_activation!!</code></a></li><li><a href="#LuxLib.fused_conv_bias_activation"><code>LuxLib.fused_conv_bias_activation</code></a></li><li><a href="#LuxLib.fused_dense_bias_activation"><code>LuxLib.fused_dense_bias_activation</code></a></li><li><a href="#LuxLib.groupnorm"><code>LuxLib.groupnorm</code></a></li><li><a href="#LuxLib.instancenorm"><code>LuxLib.instancenorm</code></a></li><li><a href="#LuxLib.layernorm"><code>LuxLib.layernorm</code></a></li></ul><h2 id="Fully-Connected-Layers" tabindex="-1">Fully Connected Layers <a class="header-anchor" href="#Fully-Connected-Layers" aria-label="Permalink to &quot;Fully Connected Layers {#Fully-Connected-Layers}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxLib.fused_dense_bias_activation" href="#LuxLib.fused_dense_bias_activation">#</a> <b><u>LuxLib.fused_dense_bias_activation</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">fused_dense_bias_activation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(σ</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">F</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, weight</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractMatrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractMatrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    b</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Nothing, AbstractVector}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {F}</span></span></code></pre></div><p>Compute <code>σ.(weight * x .+ b)</code> with the best possible implementation available. Currently this implementation attempts to minimize reallocations by reusing the output buffer for multiple operations.</p><p><strong>Arguments</strong></p><ul><li><p><code>σ</code>: Activation function</p></li><li><p><code>weight</code>: Weight matrix</p></li><li><p><code>x</code>: Input matrix</p></li><li><p><code>b</code>: Bias vector (can be <code>nothing</code>)</p></li></ul><p><strong>Notes on implementation</strong></p><ul><li><p>Despite the naming, currently only the activation (σ) is fused with the bias addition. We are working towards using faster hardware specific fused kernels for this operation. Currently this is equivalent to using matrix multiply followed by <code>NNlib.bias_act!</code>, though this function doesn&#39;t call those operations.</p></li><li><p>If any of the inputs, don&#39;t support setindexing (aka immutable arrays) we fallback to the generic non-mutating implementation.</p></li><li><p>For mixed precision inputs, we use the fallback allocating implementation.</p></li><li><p>Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD backends or backends that support mutation. Backends like <code>Tracker</code> and <code>ReverseDiff</code> fallback to the generic implementation.</p></li></ul><p><a href="https://github.com/LuxDL/LuxLib.jl/blob/v0.3.17/src/api/dense.jl#L2-L29" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Convolutional-Layers" tabindex="-1">Convolutional Layers <a class="header-anchor" href="#Convolutional-Layers" aria-label="Permalink to &quot;Convolutional Layers {#Convolutional-Layers}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxLib.fused_conv_bias_activation" href="#LuxLib.fused_conv_bias_activation">#</a> <b><u>LuxLib.fused_conv_bias_activation</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">fused_conv_bias_activation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(σ</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">F</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, weight</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    b</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Nothing, AbstractArray}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, cdims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ConvDims</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {F}</span></span></code></pre></div><p>Computes <code>σ.(conv(x, weight, cdims) .+ b)</code> with the best possible implementation available. This operation fuses operations into a single kernel if possible, and minimizes reallocations by reusing the output buffer for multiple operations.</p><p><strong>Arguments</strong></p><ul><li><p><code>σ</code>: Activation function</p></li><li><p><code>weight</code>: Weight tensor</p></li><li><p><code>x</code>: Input tensor</p></li><li><p><code>b</code>: Bias tensor (can be <code>nothing</code>)</p></li><li><p><code>cdims</code>: <code>ConvDims</code> object</p></li></ul><p><strong>Notes on implementation</strong></p><ul><li><p>For CUDA Arrays, this uses fused CUDNN kernels when the activation is <code>identity</code> or <code>relu</code>. For other activations, it tries to fuse the operations on the Julia side.</p></li><li><p>If any of the inputs, don&#39;t support setindexing (aka immutable arrays) we fallback to the generic non-mutating implementation.</p></li><li><p>For mixed precision inputs, we use the fallback allocating implementation.</p></li><li><p>Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD backends or backends that support mutation. Backends like <code>Tracker</code> and <code>ReverseDiff</code> fallback to the generic implementation.</p></li><li><p>For Mixed-Precision Inputs on GPU, we type promote the inputs to the highest precision, with a warning.</p></li></ul><p><a href="https://github.com/LuxDL/LuxLib.jl/blob/v0.3.17/src/api/conv.jl#L2-L30" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Dropout" tabindex="-1">Dropout <a class="header-anchor" href="#Dropout" aria-label="Permalink to &quot;Dropout {#Dropout}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxLib.alpha_dropout" href="#LuxLib.alpha_dropout">#</a> <b><u>LuxLib.alpha_dropout</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">alpha_dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractRNG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x, p, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val{training}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">alpha_dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractRNG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x, p, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val{training}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, α, A, B)</span></span></code></pre></div><p>Alpha Dropout: Dropout ensuring that the mean and variance of the output remains same as the input. For details see [1]. Use the second call signature to avoid recomputing the constants for a fixed dropout probability.</p><p><strong>Arguments</strong></p><ul><li><p><code>rng</code>: Random number generator</p></li><li><p><code>x</code>: Input Array</p></li><li><p><code>p</code>: Probability of an element to be dropped out</p></li><li><p><code>Val(training)</code>: If <code>true</code> then dropout is applied on <code>x</code> with probability <code>p</code>. Else, <code>x</code> is returned</p></li><li><p><code>α</code>: <code>-1.7580993408473766</code>. Computed at limit x tends to infinity, <code>selu(x) = -λβ = α</code></p></li><li><p><code>A</code>: Scaling factor for the mean</p></li><li><p><code>B</code>: Scaling factor for the variance</p></li></ul><p><strong>Returns</strong></p><ul><li><p>Output Array after applying alpha dropout</p></li><li><p>Updated state for the random number generator</p></li></ul><p><strong>References</strong></p><p>[1] Klambauer, Günter, et al. &quot;Self-normalizing neural networks.&quot; Advances in neural information processing systems 30 (2017).</p><p><a href="https://github.com/LuxDL/LuxLib.jl/blob/v0.3.17/src/api/dropout.jl#L74-L102" target="_blank" rel="noreferrer">source</a></p></div><br>', 13);
const _hoisted_14 = { style: { "border-width": "1px", "border-style": "solid", "border-color": "black", "padding": "1em", "border-radius": "25px" } };
const _hoisted_15 = /* @__PURE__ */ createBaseVNode("a", {
  id: "LuxLib.dropout",
  href: "#LuxLib.dropout"
}, "#", -1);
const _hoisted_16 = /* @__PURE__ */ createBaseVNode("b", null, [
  /* @__PURE__ */ createBaseVNode("u", null, "LuxLib.dropout")
], -1);
const _hoisted_17 = /* @__PURE__ */ createBaseVNode("i", null, "Function", -1);
const _hoisted_18 = /* @__PURE__ */ createStaticVNode('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractRNG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x, p, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val{training}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, invp; dims)</span></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractRNG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x, mask, p, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val{training}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val{update_mask}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, invp;</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        dims)</span></span></code></pre></div><p>Dropout: Simple Way to prevent Neural Networks for Overfitting. For details see [1].</p><p><strong>Arguments</strong></p><ul><li><p><code>rng</code>: Random number generator</p></li><li><p><code>x</code>: Input Array</p></li><li><p><code>mask</code>: Dropout Mask. If not used then it is constructed automatically</p></li><li><p><code>p</code>: Probability of an element to be dropped out</p></li><li><p><code>Val(training)</code>: If <code>true</code> then dropout is applied on <code>x</code> with probability <code>p</code> along <code>dims</code>. Else, <code>x</code> is returned</p></li><li><p><code>Val(update_mask)</code>: If <code>true</code> then the mask is generated and used. Else, the <code>mask</code> provided is directly used</p></li><li><p><code>invp</code>: Inverse of the probability</p></li></ul><p><strong>Keyword Arguments</strong></p>', 5);
const _hoisted_23 = /* @__PURE__ */ createBaseVNode("li", null, [
  /* @__PURE__ */ createBaseVNode("p", null, [
    /* @__PURE__ */ createBaseVNode("code", null, "dims"),
    /* @__PURE__ */ createTextVNode(": Dimensions along which dropout is applied")
  ])
], -1);
const _hoisted_24 = /* @__PURE__ */ createBaseVNode("code", null, "invp", -1);
const _hoisted_25 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_26 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-1.091ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.8ex",
  height: "3.048ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -864.9 795.7 1347.1",
  "aria-hidden": "true"
};
const _hoisted_27 = /* @__PURE__ */ createStaticVNode('<g stroke="currentColor" fill="currentColor" stroke-width="0" transform="scale(1,-1)"><g data-mml-node="math"><g data-mml-node="mfrac"><g data-mml-node="mn" transform="translate(221.1,394) scale(0.707)"><path data-c="31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(220,-345) scale(0.707)"><path data-c="1D45D" d="M23 287Q24 290 25 295T30 317T40 348T55 381T75 411T101 433T134 442Q209 442 230 378L240 387Q302 442 358 442Q423 442 460 395T497 281Q497 173 421 82T249 -10Q227 -10 210 -4Q199 1 187 11T168 28L161 36Q160 35 139 -51T118 -138Q118 -144 126 -145T163 -148H188Q194 -155 194 -157T191 -175Q188 -187 185 -190T172 -194Q170 -194 161 -194T127 -193T65 -192Q-5 -192 -24 -194H-32Q-39 -187 -39 -183Q-37 -156 -26 -148H-6Q28 -147 33 -136Q36 -130 94 103T155 350Q156 355 156 364Q156 405 131 405Q109 405 94 377T71 316T59 280Q57 278 43 278H29Q23 284 23 287ZM178 102Q200 26 252 26Q282 26 310 49T356 107Q374 141 392 215T411 325V331Q411 405 350 405Q339 405 328 402T306 393T286 380T269 365T254 350T243 336T235 326L232 322Q232 321 229 308T218 264T204 212Q178 106 178 102Z" style="stroke-width:3;"></path></g><rect width="555.7" height="60" x="120" y="220"></rect></g></g></g>', 1);
const _hoisted_28 = [
  _hoisted_27
];
const _hoisted_29 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mfrac", null, [
      /* @__PURE__ */ createBaseVNode("mn", null, "1"),
      /* @__PURE__ */ createBaseVNode("mi", null, "p")
    ])
  ])
], -1);
const _hoisted_30 = /* @__PURE__ */ createStaticVNode('<p><strong>Returns</strong></p><ul><li><p>Output Array after applying dropout</p></li><li><p>Dropout Mask (if <code>training == false</code>, the returned value is meaningless)</p></li><li><p>Updated state for the random number generator</p></li></ul><p><strong>References</strong></p><p>[1] Srivastava, Nitish, et al. &quot;Dropout: a simple way to prevent neural networks from overfitting.&quot; The journal of machine learning research 15.1 (2014): 1929-1958.</p><p><a href="https://github.com/LuxDL/LuxLib.jl/blob/v0.3.17/src/api/dropout.jl#L1" target="_blank" rel="noreferrer">source</a></p>', 5);
const _hoisted_35 = /* @__PURE__ */ createBaseVNode("br", null, null, -1);
const _hoisted_36 = /* @__PURE__ */ createBaseVNode("h2", {
  id: "Normalization",
  tabindex: "-1"
}, [
  /* @__PURE__ */ createTextVNode("Normalization "),
  /* @__PURE__ */ createBaseVNode("a", {
    class: "header-anchor",
    href: "#Normalization",
    "aria-label": 'Permalink to "Normalization {#Normalization}"'
  }, "​")
], -1);
const _hoisted_37 = { style: { "border-width": "1px", "border-style": "solid", "border-color": "black", "padding": "1em", "border-radius": "25px" } };
const _hoisted_38 = /* @__PURE__ */ createBaseVNode("a", {
  id: "LuxLib.batchnorm",
  href: "#LuxLib.batchnorm"
}, "#", -1);
const _hoisted_39 = /* @__PURE__ */ createBaseVNode("b", null, [
  /* @__PURE__ */ createBaseVNode("u", null, "LuxLib.batchnorm")
], -1);
const _hoisted_40 = /* @__PURE__ */ createBaseVNode("i", null, "Function", -1);
const _hoisted_41 = /* @__PURE__ */ createStaticVNode('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">batchnorm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, scale, bias, running_mean, running_var, σ</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">identity; momentum, epsilon,</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    training)</span></span></code></pre></div><p>Batch Normalization. For details see [1].</p>', 2);
const _hoisted_43 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_44 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.471ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "25.07ex",
  height: "2.016ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -683 11080.9 891",
  "aria-hidden": "true"
};
const _hoisted_45 = /* @__PURE__ */ createStaticVNode('<g stroke="currentColor" fill="currentColor" stroke-width="0" transform="scale(1,-1)"><g data-mml-node="math"><g data-mml-node="msub"><g data-mml-node="mi"><path data-c="1D437" d="M287 628Q287 635 230 637Q207 637 200 638T193 647Q193 655 197 667T204 682Q206 683 403 683Q570 682 590 682T630 676Q702 659 752 597T803 431Q803 275 696 151T444 3L430 1L236 0H125H72Q48 0 41 2T33 11Q33 13 36 25Q40 41 44 43T67 46Q94 46 127 49Q141 52 146 61Q149 65 218 339T287 628ZM703 469Q703 507 692 537T666 584T629 613T590 629T555 636Q553 636 541 636T512 636T479 637H436Q392 637 386 627Q384 623 313 339T242 52Q242 48 253 48T330 47Q335 47 349 47T373 46Q499 46 581 128Q617 164 640 212T683 339T703 469Z" style="stroke-width:3;"></path></g><g data-mml-node="mn" transform="translate(861,-150) scale(0.707)"><path data-c="31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z" style="stroke-width:3;"></path></g></g><g data-mml-node="mo" transform="translate(1264.6,0)"><path data-c="D7" d="M630 29Q630 9 609 9Q604 9 587 25T493 118L389 222L284 117Q178 13 175 11Q171 9 168 9Q160 9 154 15T147 29Q147 36 161 51T255 146L359 250L255 354Q174 435 161 449T147 471Q147 480 153 485T168 490Q173 490 175 489Q178 487 284 383L389 278L493 382Q570 459 587 475T609 491Q630 491 630 471Q630 464 620 453T522 355L418 250L522 145Q606 61 618 48T630 29Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(2042.6,0)"><path data-c="2E" d="M78 60Q78 84 95 102T138 120Q162 120 180 104T199 61Q199 36 182 18T139 0T96 17T78 60Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(2487.2,0)"><path data-c="2E" d="M78 60Q78 84 95 102T138 120Q162 120 180 104T199 61Q199 36 182 18T139 0T96 17T78 60Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(2931.9,0)"><path data-c="2E" d="M78 60Q78 84 95 102T138 120Q162 120 180 104T199 61Q199 36 182 18T139 0T96 17T78 60Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(3376.6,0)"><path data-c="D7" d="M630 29Q630 9 609 9Q604 9 587 25T493 118L389 222L284 117Q178 13 175 11Q171 9 168 9Q160 9 154 15T147 29Q147 36 161 51T255 146L359 250L255 354Q174 435 161 449T147 471Q147 480 153 485T168 490Q173 490 175 489Q178 487 284 383L389 278L493 382Q570 459 587 475T609 491Q630 491 630 471Q630 464 620 453T522 355L418 250L522 145Q606 61 618 48T630 29Z" style="stroke-width:3;"></path></g><g data-mml-node="msub" transform="translate(4154.6,0)"><g data-mml-node="mi"><path data-c="1D437" d="M287 628Q287 635 230 637Q207 637 200 638T193 647Q193 655 197 667T204 682Q206 683 403 683Q570 682 590 682T630 676Q702 659 752 597T803 431Q803 275 696 151T444 3L430 1L236 0H125H72Q48 0 41 2T33 11Q33 13 36 25Q40 41 44 43T67 46Q94 46 127 49Q141 52 146 61Q149 65 218 339T287 628ZM703 469Q703 507 692 537T666 584T629 613T590 629T555 636Q553 636 541 636T512 636T479 637H436Q392 637 386 627Q384 623 313 339T242 52Q242 48 253 48T330 47Q335 47 349 47T373 46Q499 46 581 128Q617 164 640 212T683 339T703 469Z" style="stroke-width:3;"></path></g><g data-mml-node="TeXAtom" transform="translate(861,-150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><path data-c="1D441" d="M234 637Q231 637 226 637Q201 637 196 638T191 649Q191 676 202 682Q204 683 299 683Q376 683 387 683T401 677Q612 181 616 168L670 381Q723 592 723 606Q723 633 659 637Q635 637 635 648Q635 650 637 660Q641 676 643 679T653 683Q656 683 684 682T767 680Q817 680 843 681T873 682Q888 682 888 672Q888 650 880 642Q878 637 858 637Q787 633 769 597L620 7Q618 0 599 0Q585 0 582 2Q579 5 453 305L326 604L261 344Q196 88 196 79Q201 46 268 46H278Q284 41 284 38T282 19Q278 6 272 0H259Q228 2 151 2Q123 2 100 2T63 2T46 1Q31 1 31 10Q31 14 34 26T39 40Q41 46 62 46Q130 49 150 85Q154 91 221 362L289 634Q287 635 234 637Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(888,0)"><path data-c="2212" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z" style="stroke-width:3;"></path></g><g data-mml-node="mn" transform="translate(1666,0)"><path data-c="32" d="M109 429Q82 429 66 447T50 491Q50 562 103 614T235 666Q326 666 387 610T449 465Q449 422 429 383T381 315T301 241Q265 210 201 149L142 93L218 92Q375 92 385 97Q392 99 409 186V189H449V186Q448 183 436 95T421 3V0H50V19V31Q50 38 56 46T86 81Q115 113 136 137Q145 147 170 174T204 211T233 244T261 278T284 308T305 340T320 369T333 401T340 431T343 464Q343 527 309 573T212 619Q179 619 154 602T119 569T109 550Q109 549 114 549Q132 549 151 535T170 489Q170 464 154 447T109 429Z" style="stroke-width:3;"></path></g></g></g><g data-mml-node="mo" transform="translate(6819.4,0)"><path data-c="D7" d="M630 29Q630 9 609 9Q604 9 587 25T493 118L389 222L284 117Q178 13 175 11Q171 9 168 9Q160 9 154 15T147 29Q147 36 161 51T255 146L359 250L255 354Q174 435 161 449T147 471Q147 480 153 485T168 490Q173 490 175 489Q178 487 284 383L389 278L493 382Q570 459 587 475T609 491Q630 491 630 471Q630 464 620 453T522 355L418 250L522 145Q606 61 618 48T630 29Z" style="stroke-width:3;"></path></g><g data-mml-node="mn" transform="translate(7819.6,0)"><path data-c="31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(8541.8,0)"><path data-c="D7" d="M630 29Q630 9 609 9Q604 9 587 25T493 118L389 222L284 117Q178 13 175 11Q171 9 168 9Q160 9 154 15T147 29Q147 36 161 51T255 146L359 250L255 354Q174 435 161 449T147 471Q147 480 153 485T168 490Q173 490 175 489Q178 487 284 383L389 278L493 382Q570 459 587 475T609 491Q630 491 630 471Q630 464 620 453T522 355L418 250L522 145Q606 61 618 48T630 29Z" style="stroke-width:3;"></path></g><g data-mml-node="msub" transform="translate(9542,0)"><g data-mml-node="mi"><path data-c="1D437" d="M287 628Q287 635 230 637Q207 637 200 638T193 647Q193 655 197 667T204 682Q206 683 403 683Q570 682 590 682T630 676Q702 659 752 597T803 431Q803 275 696 151T444 3L430 1L236 0H125H72Q48 0 41 2T33 11Q33 13 36 25Q40 41 44 43T67 46Q94 46 127 49Q141 52 146 61Q149 65 218 339T287 628ZM703 469Q703 507 692 537T666 584T629 613T590 629T555 636Q553 636 541 636T512 636T479 637H436Q392 637 386 627Q384 623 313 339T242 52Q242 48 253 48T330 47Q335 47 349 47T373 46Q499 46 581 128Q617 164 640 212T683 339T703 469Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(861,-150) scale(0.707)"><path data-c="1D441" d="M234 637Q231 637 226 637Q201 637 196 638T191 649Q191 676 202 682Q204 683 299 683Q376 683 387 683T401 677Q612 181 616 168L670 381Q723 592 723 606Q723 633 659 637Q635 637 635 648Q635 650 637 660Q641 676 643 679T653 683Q656 683 684 682T767 680Q817 680 843 681T873 682Q888 682 888 672Q888 650 880 642Q878 637 858 637Q787 633 769 597L620 7Q618 0 599 0Q585 0 582 2Q579 5 453 305L326 604L261 344Q196 88 196 79Q201 46 268 46H278Q284 41 284 38T282 19Q278 6 272 0H259Q228 2 151 2Q123 2 100 2T63 2T46 1Q31 1 31 10Q31 14 34 26T39 40Q41 46 62 46Q130 49 150 85Q154 91 221 362L289 634Q287 635 234 637Z" style="stroke-width:3;"></path></g></g></g></g>', 1);
const _hoisted_46 = [
  _hoisted_45
];
const _hoisted_47 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "D"),
      /* @__PURE__ */ createBaseVNode("mn", null, "1")
    ]),
    /* @__PURE__ */ createBaseVNode("mo", null, "×"),
    /* @__PURE__ */ createBaseVNode("mo", null, "."),
    /* @__PURE__ */ createBaseVNode("mo", null, "."),
    /* @__PURE__ */ createBaseVNode("mo", null, "."),
    /* @__PURE__ */ createBaseVNode("mo", null, "×"),
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "D"),
      /* @__PURE__ */ createBaseVNode("mrow", { "data-mjx-texclass": "ORD" }, [
        /* @__PURE__ */ createBaseVNode("mi", null, "N"),
        /* @__PURE__ */ createBaseVNode("mo", null, "−"),
        /* @__PURE__ */ createBaseVNode("mn", null, "2")
      ])
    ]),
    /* @__PURE__ */ createBaseVNode("mo", null, "×"),
    /* @__PURE__ */ createBaseVNode("mn", null, "1"),
    /* @__PURE__ */ createBaseVNode("mo", null, "×"),
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "D"),
      /* @__PURE__ */ createBaseVNode("mi", null, "N")
    ])
  ])
], -1);
const _hoisted_48 = /* @__PURE__ */ createBaseVNode("p", null, [
  /* @__PURE__ */ createBaseVNode("strong", null, "Arguments")
], -1);
const _hoisted_49 = /* @__PURE__ */ createBaseVNode("li", null, [
  /* @__PURE__ */ createBaseVNode("p", null, [
    /* @__PURE__ */ createBaseVNode("code", null, "x"),
    /* @__PURE__ */ createTextVNode(": Input to be Normalized")
  ])
], -1);
const _hoisted_50 = /* @__PURE__ */ createBaseVNode("code", null, "scale", -1);
const _hoisted_51 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_52 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.489ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.229ex",
  height: "1.486ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -441 543 657",
  "aria-hidden": "true"
};
const _hoisted_53 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D6FE",
        d: "M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_54 = [
  _hoisted_53
];
const _hoisted_55 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "γ")
  ])
], -1);
const _hoisted_56 = /* @__PURE__ */ createBaseVNode("code", null, "nothing", -1);
const _hoisted_57 = /* @__PURE__ */ createBaseVNode("code", null, "bias", -1);
const _hoisted_58 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_59 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.439ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.281ex",
  height: "2.034ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -705 566 899",
  "aria-hidden": "true"
};
const _hoisted_60 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D6FD",
        d: "M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_61 = [
  _hoisted_60
];
const _hoisted_62 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "β")
  ])
], -1);
const _hoisted_63 = /* @__PURE__ */ createBaseVNode("code", null, "nothing", -1);
const _hoisted_64 = /* @__PURE__ */ createStaticVNode("<li><p><code>running_mean</code>: Running mean (can be <code>nothing</code>)</p></li><li><p><code>running_var</code>: Running variance (can be <code>nothing</code>)</p></li><li><p><code>σ</code>: Activation function (default: <code>identity</code>)</p></li>", 3);
const _hoisted_67 = /* @__PURE__ */ createStaticVNode('<p><strong>Keyword Arguments</strong></p><ul><li><p><code>momentum</code>: Momentum for updating running mean and variance</p></li><li><p><code>epsilon</code>: Value added to the denominator for numerical stability</p></li><li><p><code>training</code>: Set to <code>Val(true)</code> if running in training mode</p></li></ul><p><strong>Returns</strong></p><p>Normalized Array of same size as <code>x</code>. And a Named Tuple containing the updated running mean and variance.</p><p><strong>Performance Considerations</strong></p><p>If the input array is <code>2D</code>, <code>4D</code>, or <code>5D</code> <code>CuArray</code> with element types <code>Float16</code>, <code>Float32</code> and <code>Float64</code>, then the CUDNN code path will be used. In all other cases, a broadcasting fallback is used which is not highly optimized.</p><p><strong>References</strong></p><p>[1] Ioffe, Sergey, and Christian Szegedy. &quot;Batch normalization: Accelerating deep network training by reducing internal covariate shift.&quot; International conference on machine learning. PMLR, 2015.</p><p><a href="https://github.com/LuxDL/LuxLib.jl/blob/v0.3.17/src/api/batchnorm.jl#L1" target="_blank" rel="noreferrer">source</a></p>', 9);
const _hoisted_76 = /* @__PURE__ */ createBaseVNode("br", null, null, -1);
const _hoisted_77 = { style: { "border-width": "1px", "border-style": "solid", "border-color": "black", "padding": "1em", "border-radius": "25px" } };
const _hoisted_78 = /* @__PURE__ */ createBaseVNode("a", {
  id: "LuxLib.groupnorm",
  href: "#LuxLib.groupnorm"
}, "#", -1);
const _hoisted_79 = /* @__PURE__ */ createBaseVNode("b", null, [
  /* @__PURE__ */ createBaseVNode("u", null, "LuxLib.groupnorm")
], -1);
const _hoisted_80 = /* @__PURE__ */ createBaseVNode("i", null, "Function", -1);
const _hoisted_81 = /* @__PURE__ */ createStaticVNode('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">groupnorm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, scale, bias; groups, epsilon)</span></span></code></pre></div><p>Group Normalization. For details see [1].</p><p>This op is similar to batch normalization, but statistics are shared across equally-sized groups of channels and not shared across batch dimension. Thus, group normalization does not depend on the batch composition and does not require maintaining internal state for storing statistics.</p><p><strong>Arguments</strong></p>', 4);
const _hoisted_85 = /* @__PURE__ */ createBaseVNode("li", null, [
  /* @__PURE__ */ createBaseVNode("p", null, [
    /* @__PURE__ */ createBaseVNode("code", null, "x"),
    /* @__PURE__ */ createTextVNode(": Input to be Normalized")
  ])
], -1);
const _hoisted_86 = /* @__PURE__ */ createBaseVNode("code", null, "scale", -1);
const _hoisted_87 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_88 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.489ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.229ex",
  height: "1.486ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -441 543 657",
  "aria-hidden": "true"
};
const _hoisted_89 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D6FE",
        d: "M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_90 = [
  _hoisted_89
];
const _hoisted_91 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "γ")
  ])
], -1);
const _hoisted_92 = /* @__PURE__ */ createBaseVNode("code", null, "nothing", -1);
const _hoisted_93 = /* @__PURE__ */ createBaseVNode("code", null, "bias", -1);
const _hoisted_94 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_95 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.439ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.281ex",
  height: "2.034ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -705 566 899",
  "aria-hidden": "true"
};
const _hoisted_96 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D6FD",
        d: "M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_97 = [
  _hoisted_96
];
const _hoisted_98 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "β")
  ])
], -1);
const _hoisted_99 = /* @__PURE__ */ createBaseVNode("code", null, "nothing", -1);
const _hoisted_100 = /* @__PURE__ */ createStaticVNode('<p><strong>Keyword Arguments</strong></p><ul><li><p><code>groups</code>: Number of groups</p></li><li><p><code>epsilon</code>: Value added to the denominator for numerical stability</p></li></ul><p><strong>Returns</strong></p><p>The normalized array is returned.</p><p><strong>Performance Considerations</strong></p><p>The most common case of this Op – <code>x</code> is a 4D array – is optimized using KernelAbstractions and has a fast custom backwards pass implemented. All other cases have a fallback implementation which is not especially optimized.</p><p>We have tested the code path for <code>Float16</code> and it works, but gradient accumulation is extremely fragile. Hence, for <code>Float16</code> inputs, it uses the fallback implementation.</p><p>If the batch size is small (&lt; 16), then the fallback implementation will be faster than the KA version. However, this customization is not possible using the direct <code>groupnorm</code> interface.</p><p><strong>References</strong></p><p>[1] Wu, Yuxin, and Kaiming He. &quot;Group normalization.&quot; Proceedings of the European conference on computer vision (ECCV). 2018.</p><p><a href="https://github.com/LuxDL/LuxLib.jl/blob/v0.3.17/src/api/groupnorm.jl#L1" target="_blank" rel="noreferrer">source</a></p>', 11);
const _hoisted_111 = /* @__PURE__ */ createBaseVNode("br", null, null, -1);
const _hoisted_112 = { style: { "border-width": "1px", "border-style": "solid", "border-color": "black", "padding": "1em", "border-radius": "25px" } };
const _hoisted_113 = /* @__PURE__ */ createBaseVNode("a", {
  id: "LuxLib.instancenorm",
  href: "#LuxLib.instancenorm"
}, "#", -1);
const _hoisted_114 = /* @__PURE__ */ createBaseVNode("b", null, [
  /* @__PURE__ */ createBaseVNode("u", null, "LuxLib.instancenorm")
], -1);
const _hoisted_115 = /* @__PURE__ */ createBaseVNode("i", null, "Function", -1);
const _hoisted_116 = /* @__PURE__ */ createStaticVNode('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">instancenorm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, scale, bias, σ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> identity; epsilon, training)</span></span></code></pre></div><p>Instance Normalization. For details see [1].</p>', 2);
const _hoisted_118 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_119 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.471ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "22.72ex",
  height: "2.016ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -683 10042 891",
  "aria-hidden": "true"
};
const _hoisted_120 = /* @__PURE__ */ createStaticVNode('<g stroke="currentColor" fill="currentColor" stroke-width="0" transform="scale(1,-1)"><g data-mml-node="math"><g data-mml-node="msub"><g data-mml-node="mi"><path data-c="1D437" d="M287 628Q287 635 230 637Q207 637 200 638T193 647Q193 655 197 667T204 682Q206 683 403 683Q570 682 590 682T630 676Q702 659 752 597T803 431Q803 275 696 151T444 3L430 1L236 0H125H72Q48 0 41 2T33 11Q33 13 36 25Q40 41 44 43T67 46Q94 46 127 49Q141 52 146 61Q149 65 218 339T287 628ZM703 469Q703 507 692 537T666 584T629 613T590 629T555 636Q553 636 541 636T512 636T479 637H436Q392 637 386 627Q384 623 313 339T242 52Q242 48 253 48T330 47Q335 47 349 47T373 46Q499 46 581 128Q617 164 640 212T683 339T703 469Z" style="stroke-width:3;"></path></g><g data-mml-node="mn" transform="translate(861,-150) scale(0.707)"><path data-c="31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z" style="stroke-width:3;"></path></g></g><g data-mml-node="mo" transform="translate(1264.6,0)"><path data-c="D7" d="M630 29Q630 9 609 9Q604 9 587 25T493 118L389 222L284 117Q178 13 175 11Q171 9 168 9Q160 9 154 15T147 29Q147 36 161 51T255 146L359 250L255 354Q174 435 161 449T147 471Q147 480 153 485T168 490Q173 490 175 489Q178 487 284 383L389 278L493 382Q570 459 587 475T609 491Q630 491 630 471Q630 464 620 453T522 355L418 250L522 145Q606 61 618 48T630 29Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(2042.6,0)"><path data-c="2E" d="M78 60Q78 84 95 102T138 120Q162 120 180 104T199 61Q199 36 182 18T139 0T96 17T78 60Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(2487.2,0)"><path data-c="2E" d="M78 60Q78 84 95 102T138 120Q162 120 180 104T199 61Q199 36 182 18T139 0T96 17T78 60Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(2931.9,0)"><path data-c="2E" d="M78 60Q78 84 95 102T138 120Q162 120 180 104T199 61Q199 36 182 18T139 0T96 17T78 60Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(3376.6,0)"><path data-c="D7" d="M630 29Q630 9 609 9Q604 9 587 25T493 118L389 222L284 117Q178 13 175 11Q171 9 168 9Q160 9 154 15T147 29Q147 36 161 51T255 146L359 250L255 354Q174 435 161 449T147 471Q147 480 153 485T168 490Q173 490 175 489Q178 487 284 383L389 278L493 382Q570 459 587 475T609 491Q630 491 630 471Q630 464 620 453T522 355L418 250L522 145Q606 61 618 48T630 29Z" style="stroke-width:3;"></path></g><g data-mml-node="msub" transform="translate(4154.6,0)"><g data-mml-node="mi"><path data-c="1D437" d="M287 628Q287 635 230 637Q207 637 200 638T193 647Q193 655 197 667T204 682Q206 683 403 683Q570 682 590 682T630 676Q702 659 752 597T803 431Q803 275 696 151T444 3L430 1L236 0H125H72Q48 0 41 2T33 11Q33 13 36 25Q40 41 44 43T67 46Q94 46 127 49Q141 52 146 61Q149 65 218 339T287 628ZM703 469Q703 507 692 537T666 584T629 613T590 629T555 636Q553 636 541 636T512 636T479 637H436Q392 637 386 627Q384 623 313 339T242 52Q242 48 253 48T330 47Q335 47 349 47T373 46Q499 46 581 128Q617 164 640 212T683 339T703 469Z" style="stroke-width:3;"></path></g><g data-mml-node="TeXAtom" transform="translate(861,-150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><path data-c="1D441" d="M234 637Q231 637 226 637Q201 637 196 638T191 649Q191 676 202 682Q204 683 299 683Q376 683 387 683T401 677Q612 181 616 168L670 381Q723 592 723 606Q723 633 659 637Q635 637 635 648Q635 650 637 660Q641 676 643 679T653 683Q656 683 684 682T767 680Q817 680 843 681T873 682Q888 682 888 672Q888 650 880 642Q878 637 858 637Q787 633 769 597L620 7Q618 0 599 0Q585 0 582 2Q579 5 453 305L326 604L261 344Q196 88 196 79Q201 46 268 46H278Q284 41 284 38T282 19Q278 6 272 0H259Q228 2 151 2Q123 2 100 2T63 2T46 1Q31 1 31 10Q31 14 34 26T39 40Q41 46 62 46Q130 49 150 85Q154 91 221 362L289 634Q287 635 234 637Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(888,0)"><path data-c="2212" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z" style="stroke-width:3;"></path></g><g data-mml-node="mn" transform="translate(1666,0)"><path data-c="32" d="M109 429Q82 429 66 447T50 491Q50 562 103 614T235 666Q326 666 387 610T449 465Q449 422 429 383T381 315T301 241Q265 210 201 149L142 93L218 92Q375 92 385 97Q392 99 409 186V189H449V186Q448 183 436 95T421 3V0H50V19V31Q50 38 56 46T86 81Q115 113 136 137Q145 147 170 174T204 211T233 244T261 278T284 308T305 340T320 369T333 401T340 431T343 464Q343 527 309 573T212 619Q179 619 154 602T119 569T109 550Q109 549 114 549Q132 549 151 535T170 489Q170 464 154 447T109 429Z" style="stroke-width:3;"></path></g></g></g><g data-mml-node="mo" transform="translate(6819.4,0)"><path data-c="D7" d="M630 29Q630 9 609 9Q604 9 587 25T493 118L389 222L284 117Q178 13 175 11Q171 9 168 9Q160 9 154 15T147 29Q147 36 161 51T255 146L359 250L255 354Q174 435 161 449T147 471Q147 480 153 485T168 490Q173 490 175 489Q178 487 284 383L389 278L493 382Q570 459 587 475T609 491Q630 491 630 471Q630 464 620 453T522 355L418 250L522 145Q606 61 618 48T630 29Z" style="stroke-width:3;"></path></g><g data-mml-node="mn" transform="translate(7819.6,0)"><path data-c="31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(8541.8,0)"><path data-c="D7" d="M630 29Q630 9 609 9Q604 9 587 25T493 118L389 222L284 117Q178 13 175 11Q171 9 168 9Q160 9 154 15T147 29Q147 36 161 51T255 146L359 250L255 354Q174 435 161 449T147 471Q147 480 153 485T168 490Q173 490 175 489Q178 487 284 383L389 278L493 382Q570 459 587 475T609 491Q630 491 630 471Q630 464 620 453T522 355L418 250L522 145Q606 61 618 48T630 29Z" style="stroke-width:3;"></path></g><g data-mml-node="mn" transform="translate(9542,0)"><path data-c="31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z" style="stroke-width:3;"></path></g></g></g>', 1);
const _hoisted_121 = [
  _hoisted_120
];
const _hoisted_122 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "D"),
      /* @__PURE__ */ createBaseVNode("mn", null, "1")
    ]),
    /* @__PURE__ */ createBaseVNode("mo", null, "×"),
    /* @__PURE__ */ createBaseVNode("mo", null, "."),
    /* @__PURE__ */ createBaseVNode("mo", null, "."),
    /* @__PURE__ */ createBaseVNode("mo", null, "."),
    /* @__PURE__ */ createBaseVNode("mo", null, "×"),
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "D"),
      /* @__PURE__ */ createBaseVNode("mrow", { "data-mjx-texclass": "ORD" }, [
        /* @__PURE__ */ createBaseVNode("mi", null, "N"),
        /* @__PURE__ */ createBaseVNode("mo", null, "−"),
        /* @__PURE__ */ createBaseVNode("mn", null, "2")
      ])
    ]),
    /* @__PURE__ */ createBaseVNode("mo", null, "×"),
    /* @__PURE__ */ createBaseVNode("mn", null, "1"),
    /* @__PURE__ */ createBaseVNode("mo", null, "×"),
    /* @__PURE__ */ createBaseVNode("mn", null, "1")
  ])
], -1);
const _hoisted_123 = /* @__PURE__ */ createBaseVNode("p", null, [
  /* @__PURE__ */ createBaseVNode("strong", null, "Arguments")
], -1);
const _hoisted_124 = /* @__PURE__ */ createBaseVNode("li", null, [
  /* @__PURE__ */ createBaseVNode("p", null, [
    /* @__PURE__ */ createBaseVNode("code", null, "x"),
    /* @__PURE__ */ createTextVNode(": Input to be Normalized (must be atleast 3D)")
  ])
], -1);
const _hoisted_125 = /* @__PURE__ */ createBaseVNode("code", null, "scale", -1);
const _hoisted_126 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_127 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.489ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.229ex",
  height: "1.486ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -441 543 657",
  "aria-hidden": "true"
};
const _hoisted_128 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D6FE",
        d: "M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_129 = [
  _hoisted_128
];
const _hoisted_130 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "γ")
  ])
], -1);
const _hoisted_131 = /* @__PURE__ */ createBaseVNode("code", null, "nothing", -1);
const _hoisted_132 = /* @__PURE__ */ createBaseVNode("code", null, "bias", -1);
const _hoisted_133 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_134 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.439ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.281ex",
  height: "2.034ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -705 566 899",
  "aria-hidden": "true"
};
const _hoisted_135 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D6FD",
        d: "M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_136 = [
  _hoisted_135
];
const _hoisted_137 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "β")
  ])
], -1);
const _hoisted_138 = /* @__PURE__ */ createBaseVNode("code", null, "nothing", -1);
const _hoisted_139 = /* @__PURE__ */ createBaseVNode("li", null, [
  /* @__PURE__ */ createBaseVNode("p", null, [
    /* @__PURE__ */ createBaseVNode("code", null, "σ"),
    /* @__PURE__ */ createTextVNode(": Activation function (default: "),
    /* @__PURE__ */ createBaseVNode("code", null, "identity"),
    /* @__PURE__ */ createTextVNode(")")
  ])
], -1);
const _hoisted_140 = /* @__PURE__ */ createStaticVNode('<p><strong>Keyword Arguments</strong></p><ul><li><p><code>epsilon</code>: Value added to the denominator for numerical stability</p></li><li><p><code>training</code>: Set to <code>Val(true)</code> if running in training mode</p></li></ul><p><strong>Returns</strong></p><p>Normalized Array of same size as <code>x</code>. And a Named Tuple containing the updated running mean and variance.</p><p><strong>References</strong></p><p>[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. &quot;Instance normalization: The missing ingredient for fast stylization.&quot; arXiv preprint arXiv:1607.08022 (2016).</p><p><a href="https://github.com/LuxDL/LuxLib.jl/blob/v0.3.17/src/api/instancenorm.jl#L1" target="_blank" rel="noreferrer">source</a></p>', 7);
const _hoisted_147 = /* @__PURE__ */ createBaseVNode("br", null, null, -1);
const _hoisted_148 = { style: { "border-width": "1px", "border-style": "solid", "border-color": "black", "padding": "1em", "border-radius": "25px" } };
const _hoisted_149 = /* @__PURE__ */ createBaseVNode("a", {
  id: "LuxLib.layernorm",
  href: "#LuxLib.layernorm"
}, "#", -1);
const _hoisted_150 = /* @__PURE__ */ createBaseVNode("b", null, [
  /* @__PURE__ */ createBaseVNode("u", null, "LuxLib.layernorm")
], -1);
const _hoisted_151 = /* @__PURE__ */ createBaseVNode("i", null, "Function", -1);
const _hoisted_152 = /* @__PURE__ */ createStaticVNode('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">layernorm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, scale, bias, σ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> identity; dims, epsilon)</span></span></code></pre></div><p>Layer Normalization. For details see [1].</p>', 2);
const _hoisted_154 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_155 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.025ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.294ex",
  height: "1.025ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -442 572 453",
  "aria-hidden": "true"
};
const _hoisted_156 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D465",
        d: "M52 289Q59 331 106 386T222 442Q257 442 286 424T329 379Q371 442 430 442Q467 442 494 420T522 361Q522 332 508 314T481 292T458 288Q439 288 427 299T415 328Q415 374 465 391Q454 404 425 404Q412 404 406 402Q368 386 350 336Q290 115 290 78Q290 50 306 38T341 26Q378 26 414 59T463 140Q466 150 469 151T485 153H489Q504 153 504 145Q504 144 502 134Q486 77 440 33T333 -11Q263 -11 227 52Q186 -10 133 -10H127Q78 -10 57 16T35 71Q35 103 54 123T99 143Q142 143 142 101Q142 81 130 66T107 46T94 41L91 40Q91 39 97 36T113 29T132 26Q168 26 194 71Q203 87 217 139T245 247T261 313Q266 340 266 352Q266 380 251 392T217 404Q177 404 142 372T93 290Q91 281 88 280T72 278H58Q52 284 52 289Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_157 = [
  _hoisted_156
];
const _hoisted_158 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "x")
  ])
], -1);
const _hoisted_159 = {
  class: "MathJax",
  jax: "SVG",
  display: "true",
  style: { "direction": "ltr", "display": "block", "text-align": "center", "margin": "1em 0", "position": "relative" }
};
const _hoisted_160 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-2.76ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "25.034ex",
  height: "6.063ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -1460 11064.9 2680",
  "aria-hidden": "true"
};
const _hoisted_161 = /* @__PURE__ */ createStaticVNode('<g stroke="currentColor" fill="currentColor" stroke-width="0" transform="scale(1,-1)"><g data-mml-node="math"><g data-mml-node="mi"><path data-c="1D466" d="M21 287Q21 301 36 335T84 406T158 442Q199 442 224 419T250 355Q248 336 247 334Q247 331 231 288T198 191T182 105Q182 62 196 45T238 27Q261 27 281 38T312 61T339 94Q339 95 344 114T358 173T377 247Q415 397 419 404Q432 431 462 431Q475 431 483 424T494 412T496 403Q496 390 447 193T391 -23Q363 -106 294 -155T156 -205Q111 -205 77 -183T43 -117Q43 -95 50 -80T69 -58T89 -48T106 -45Q150 -45 150 -87Q150 -107 138 -122T115 -142T102 -147L99 -148Q101 -153 118 -160T152 -167H160Q177 -167 186 -165Q219 -156 247 -127T290 -65T313 -9T321 21L315 17Q309 13 296 6T270 -6Q250 -11 231 -11Q185 -11 150 11T104 82Q103 89 103 113Q103 170 138 262T173 379Q173 380 173 381Q173 390 173 393T169 400T158 404H154Q131 404 112 385T82 344T65 302T57 280Q55 278 41 278H27Q21 284 21 287Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(767.8,0)"><path data-c="3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z" style="stroke-width:3;"></path></g><g data-mml-node="mfrac" transform="translate(1823.6,0)"><g data-mml-node="mrow" transform="translate(1188,710)"><g data-mml-node="mi"><path data-c="1D465" d="M52 289Q59 331 106 386T222 442Q257 442 286 424T329 379Q371 442 430 442Q467 442 494 420T522 361Q522 332 508 314T481 292T458 288Q439 288 427 299T415 328Q415 374 465 391Q454 404 425 404Q412 404 406 402Q368 386 350 336Q290 115 290 78Q290 50 306 38T341 26Q378 26 414 59T463 140Q466 150 469 151T485 153H489Q504 153 504 145Q504 144 502 134Q486 77 440 33T333 -11Q263 -11 227 52Q186 -10 133 -10H127Q78 -10 57 16T35 71Q35 103 54 123T99 143Q142 143 142 101Q142 81 130 66T107 46T94 41L91 40Q91 39 97 36T113 29T132 26Q168 26 194 71Q203 87 217 139T245 247T261 313Q266 340 266 352Q266 380 251 392T217 404Q177 404 142 372T93 290Q91 281 88 280T72 278H58Q52 284 52 289Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(794.2,0)"><path data-c="2212" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z" style="stroke-width:3;"></path></g><g data-mml-node="TeXAtom" data-mjx-texclass="ORD" transform="translate(1794.4,0)"><g data-mml-node="mi"><path data-c="1D53C" d="M12 666Q12 675 24 683H582Q590 680 593 672V588Q593 514 591 502T575 490Q567 490 563 495T555 517Q552 556 517 590Q486 623 445 634T340 648H282Q266 636 264 620T260 492V370H277Q329 375 358 391T404 439Q420 480 420 506Q420 529 436 529Q445 529 451 521Q455 517 455 361Q455 333 455 298T456 253Q456 217 453 207T437 197Q420 196 420 217Q420 240 406 270Q377 328 284 335H260V201Q261 174 261 134Q262 73 264 61T278 38Q281 36 282 35H331Q400 35 449 50Q571 93 602 179Q605 203 622 203Q629 203 634 197T640 183Q638 181 624 95T604 3L600 -1H24Q12 5 12 16Q12 35 51 35Q92 38 97 52Q102 60 102 341T97 632Q91 645 51 648Q12 648 12 666ZM137 341Q137 131 136 89T130 37Q129 36 129 35H235Q233 41 231 48L226 61V623L231 635L235 648H129Q132 641 133 638T135 603T137 517T137 341ZM557 603V648H504Q504 646 515 639Q527 634 542 619L557 603ZM420 317V397L406 383Q394 370 380 363L366 355Q373 350 382 346Q400 333 409 328L420 317ZM582 61L586 88Q585 88 582 83Q557 61 526 46L511 37L542 35H577Q577 36 578 39T580 49T582 61Z" style="stroke-width:3;"></path></g></g><g data-mml-node="mo" transform="translate(2461.4,0)"><path data-c="5B" d="M118 -250V750H255V710H158V-210H255V-250H118Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(2739.4,0)"><path data-c="1D465" d="M52 289Q59 331 106 386T222 442Q257 442 286 424T329 379Q371 442 430 442Q467 442 494 420T522 361Q522 332 508 314T481 292T458 288Q439 288 427 299T415 328Q415 374 465 391Q454 404 425 404Q412 404 406 402Q368 386 350 336Q290 115 290 78Q290 50 306 38T341 26Q378 26 414 59T463 140Q466 150 469 151T485 153H489Q504 153 504 145Q504 144 502 134Q486 77 440 33T333 -11Q263 -11 227 52Q186 -10 133 -10H127Q78 -10 57 16T35 71Q35 103 54 123T99 143Q142 143 142 101Q142 81 130 66T107 46T94 41L91 40Q91 39 97 36T113 29T132 26Q168 26 194 71Q203 87 217 139T245 247T261 313Q266 340 266 352Q266 380 251 392T217 404Q177 404 142 372T93 290Q91 281 88 280T72 278H58Q52 284 52 289Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(3311.4,0)"><path data-c="5D" d="M22 710V750H159V-250H22V-210H119V710H22Z" style="stroke-width:3;"></path></g></g><g data-mml-node="msqrt" transform="translate(220,-937.5)"><g transform="translate(1020,0)"><g data-mml-node="mi"><path data-c="1D449" d="M52 648Q52 670 65 683H76Q118 680 181 680Q299 680 320 683H330Q336 677 336 674T334 656Q329 641 325 637H304Q282 635 274 635Q245 630 242 620Q242 618 271 369T301 118L374 235Q447 352 520 471T595 594Q599 601 599 609Q599 633 555 637Q537 637 537 648Q537 649 539 661Q542 675 545 679T558 683Q560 683 570 683T604 682T668 681Q737 681 755 683H762Q769 676 769 672Q769 655 760 640Q757 637 743 637Q730 636 719 635T698 630T682 623T670 615T660 608T652 599T645 592L452 282Q272 -9 266 -16Q263 -18 259 -21L241 -22H234Q216 -22 216 -15Q213 -9 177 305Q139 623 138 626Q133 637 76 637H59Q52 642 52 648Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(769,0)"><path data-c="1D44E" d="M33 157Q33 258 109 349T280 441Q331 441 370 392Q386 422 416 422Q429 422 439 414T449 394Q449 381 412 234T374 68Q374 43 381 35T402 26Q411 27 422 35Q443 55 463 131Q469 151 473 152Q475 153 483 153H487Q506 153 506 144Q506 138 501 117T481 63T449 13Q436 0 417 -8Q409 -10 393 -10Q359 -10 336 5T306 36L300 51Q299 52 296 50Q294 48 292 46Q233 -10 172 -10Q117 -10 75 30T33 157ZM351 328Q351 334 346 350T323 385T277 405Q242 405 210 374T160 293Q131 214 119 129Q119 126 119 118T118 106Q118 61 136 44T179 26Q217 26 254 59T298 110Q300 114 325 217T351 328Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(1298,0)"><path data-c="1D45F" d="M21 287Q22 290 23 295T28 317T38 348T53 381T73 411T99 433T132 442Q161 442 183 430T214 408T225 388Q227 382 228 382T236 389Q284 441 347 441H350Q398 441 422 400Q430 381 430 363Q430 333 417 315T391 292T366 288Q346 288 334 299T322 328Q322 376 378 392Q356 405 342 405Q286 405 239 331Q229 315 224 298T190 165Q156 25 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 114 189T154 366Q154 405 128 405Q107 405 92 377T68 316T57 280Q55 278 41 278H27Q21 284 21 287Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(1749,0)"><path data-c="5B" d="M118 -250V750H255V710H158V-210H255V-250H118Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(2027,0)"><path data-c="1D465" d="M52 289Q59 331 106 386T222 442Q257 442 286 424T329 379Q371 442 430 442Q467 442 494 420T522 361Q522 332 508 314T481 292T458 288Q439 288 427 299T415 328Q415 374 465 391Q454 404 425 404Q412 404 406 402Q368 386 350 336Q290 115 290 78Q290 50 306 38T341 26Q378 26 414 59T463 140Q466 150 469 151T485 153H489Q504 153 504 145Q504 144 502 134Q486 77 440 33T333 -11Q263 -11 227 52Q186 -10 133 -10H127Q78 -10 57 16T35 71Q35 103 54 123T99 143Q142 143 142 101Q142 81 130 66T107 46T94 41L91 40Q91 39 97 36T113 29T132 26Q168 26 194 71Q203 87 217 139T245 247T261 313Q266 340 266 352Q266 380 251 392T217 404Q177 404 142 372T93 290Q91 281 88 280T72 278H58Q52 284 52 289Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(2599,0)"><path data-c="5D" d="M22 710V750H159V-250H22V-210H119V710H22Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(3099.2,0)"><path data-c="2B" d="M56 237T56 250T70 270H369V420L370 570Q380 583 389 583Q402 583 409 568V270H707Q722 262 722 250T707 230H409V-68Q401 -82 391 -82H389H387Q375 -82 369 -68V230H70Q56 237 56 250Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(4099.4,0)"><path data-c="1D716" d="M227 -11Q149 -11 95 41T40 174Q40 262 87 322Q121 367 173 396T287 430Q289 431 329 431H367Q382 426 382 411Q382 385 341 385H325H312Q191 385 154 277L150 265H327Q340 256 340 246Q340 228 320 219H138V217Q128 187 128 143Q128 77 160 52T231 26Q258 26 284 36T326 57T343 68Q350 68 354 58T358 39Q358 36 357 35Q354 31 337 21T289 0T227 -11Z" style="stroke-width:3;"></path></g></g><g data-mml-node="mo" transform="translate(0,67.5)"><path data-c="221A" d="M263 249Q264 249 315 130T417 -108T470 -228L725 302Q981 837 982 839Q989 850 1001 850Q1008 850 1013 844T1020 832V826L741 243Q645 43 540 -176Q479 -303 469 -324T453 -348Q449 -350 436 -350L424 -349L315 -96Q206 156 205 156L171 130Q138 104 137 104L111 130L263 249Z" style="stroke-width:3;"></path></g><rect width="4505.4" height="60" x="1020" y="857.5"></rect></g><rect width="5725.4" height="60" x="120" y="220"></rect></g><g data-mml-node="mo" transform="translate(8011.2,0)"><path data-c="2217" d="M229 286Q216 420 216 436Q216 454 240 464Q241 464 245 464T251 465Q263 464 273 456T283 436Q283 419 277 356T270 286L328 328Q384 369 389 372T399 375Q412 375 423 365T435 338Q435 325 425 315Q420 312 357 282T289 250L355 219L425 184Q434 175 434 161Q434 146 425 136T401 125Q393 125 383 131T328 171L270 213Q283 79 283 63Q283 53 276 44T250 35Q231 35 224 44T216 63Q216 80 222 143T229 213L171 171Q115 130 110 127Q106 124 100 124Q87 124 76 134T64 161Q64 166 64 169T67 175T72 181T81 188T94 195T113 204T138 215T170 230T210 250L74 315Q65 324 65 338Q65 353 74 363T98 374Q106 374 116 368T171 328L229 286Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(8733.4,0)"><path data-c="1D6FE" d="M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(9498.7,0)"><path data-c="2B" d="M56 237T56 250T70 270H369V420L370 570Q380 583 389 583Q402 583 409 568V270H707Q722 262 722 250T707 230H409V-68Q401 -82 391 -82H389H387Q375 -82 369 -68V230H70Q56 237 56 250Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(10498.9,0)"><path data-c="1D6FD" d="M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z" style="stroke-width:3;"></path></g></g></g>', 1);
const _hoisted_162 = [
  _hoisted_161
];
const _hoisted_163 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "block",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "overflow": "hidden", "width": "100%" }
}, [
  /* @__PURE__ */ createBaseVNode("math", {
    xmlns: "http://www.w3.org/1998/Math/MathML",
    display: "block"
  }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "y"),
    /* @__PURE__ */ createBaseVNode("mo", null, "="),
    /* @__PURE__ */ createBaseVNode("mfrac", null, [
      /* @__PURE__ */ createBaseVNode("mrow", null, [
        /* @__PURE__ */ createBaseVNode("mi", null, "x"),
        /* @__PURE__ */ createBaseVNode("mo", null, "−"),
        /* @__PURE__ */ createBaseVNode("mrow", { "data-mjx-texclass": "ORD" }, [
          /* @__PURE__ */ createBaseVNode("mi", { mathvariant: "double-struck" }, "E")
        ]),
        /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "["),
        /* @__PURE__ */ createBaseVNode("mi", null, "x"),
        /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "]")
      ]),
      /* @__PURE__ */ createBaseVNode("msqrt", null, [
        /* @__PURE__ */ createBaseVNode("mi", null, "V"),
        /* @__PURE__ */ createBaseVNode("mi", null, "a"),
        /* @__PURE__ */ createBaseVNode("mi", null, "r"),
        /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "["),
        /* @__PURE__ */ createBaseVNode("mi", null, "x"),
        /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "]"),
        /* @__PURE__ */ createBaseVNode("mo", null, "+"),
        /* @__PURE__ */ createBaseVNode("mi", null, "ϵ")
      ])
    ]),
    /* @__PURE__ */ createBaseVNode("mo", null, "∗"),
    /* @__PURE__ */ createBaseVNode("mi", null, "γ"),
    /* @__PURE__ */ createBaseVNode("mo", null, "+"),
    /* @__PURE__ */ createBaseVNode("mi", null, "β")
  ])
], -1);
const _hoisted_164 = /* @__PURE__ */ createBaseVNode("p", null, [
  /* @__PURE__ */ createTextVNode("and applies the activation function "),
  /* @__PURE__ */ createBaseVNode("code", null, "σ"),
  /* @__PURE__ */ createTextVNode(" elementwise to "),
  /* @__PURE__ */ createBaseVNode("code", null, "y"),
  /* @__PURE__ */ createTextVNode(".")
], -1);
const _hoisted_165 = /* @__PURE__ */ createBaseVNode("p", null, [
  /* @__PURE__ */ createBaseVNode("strong", null, "Arguments")
], -1);
const _hoisted_166 = /* @__PURE__ */ createBaseVNode("li", null, [
  /* @__PURE__ */ createBaseVNode("p", null, [
    /* @__PURE__ */ createBaseVNode("code", null, "x"),
    /* @__PURE__ */ createTextVNode(": Input to be Normalized")
  ])
], -1);
const _hoisted_167 = /* @__PURE__ */ createBaseVNode("code", null, "scale", -1);
const _hoisted_168 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_169 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.489ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.229ex",
  height: "1.486ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -441 543 657",
  "aria-hidden": "true"
};
const _hoisted_170 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D6FE",
        d: "M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_171 = [
  _hoisted_170
];
const _hoisted_172 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "γ")
  ])
], -1);
const _hoisted_173 = /* @__PURE__ */ createBaseVNode("code", null, "nothing", -1);
const _hoisted_174 = /* @__PURE__ */ createBaseVNode("code", null, "bias", -1);
const _hoisted_175 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_176 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.439ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.281ex",
  height: "2.034ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -705 566 899",
  "aria-hidden": "true"
};
const _hoisted_177 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D6FD",
        d: "M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_178 = [
  _hoisted_177
];
const _hoisted_179 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "β")
  ])
], -1);
const _hoisted_180 = /* @__PURE__ */ createBaseVNode("code", null, "nothing", -1);
const _hoisted_181 = /* @__PURE__ */ createBaseVNode("li", null, [
  /* @__PURE__ */ createBaseVNode("p", null, [
    /* @__PURE__ */ createBaseVNode("code", null, "σ"),
    /* @__PURE__ */ createTextVNode(": Activation function (default: "),
    /* @__PURE__ */ createBaseVNode("code", null, "identity"),
    /* @__PURE__ */ createTextVNode(")")
  ])
], -1);
const _hoisted_182 = /* @__PURE__ */ createStaticVNode('<p><strong>Keyword Arguments</strong></p><ul><li><p><code>dims</code>: Dimensions along which the mean and std of <code>x</code> is computed</p></li><li><p><code>epsilon</code>: Value added to the denominator for numerical stability</p></li></ul><p><strong>Returns</strong></p><p>Normalized Array of same size as <code>x</code>.</p><p><strong>References</strong></p><p>[1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. &quot;Layer normalization.&quot; arXiv preprint arXiv:1607.06450 (2016).</p><p><a href="https://github.com/LuxDL/LuxLib.jl/blob/v0.3.17/src/api/layernorm.jl#L1" target="_blank" rel="noreferrer">source</a></p>', 7);
const _hoisted_189 = /* @__PURE__ */ createStaticVNode('<br><h2 id="Apply-Activation" tabindex="-1">Apply Activation <a class="header-anchor" href="#Apply-Activation" aria-label="Permalink to &quot;Apply Activation {#Apply-Activation}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxLib.fast_activation!!" href="#LuxLib.fast_activation!!">#</a> <b><u>LuxLib.fast_activation!!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">fast_activation!!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(σ</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">F</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {F}</span></span></code></pre></div><p>Compute <code>σ.(x)</code> with the best possible implementation available. If it is possible to rewrite <code>x</code> in-place, it does so. If <code>x</code> is an immutable array, it falls back to the generic implementation.</p><div class="tip custom-block"><p class="custom-block-title">Note</p><p>This function doesn&#39;t replace <code>σ</code> with <code>NNlib.fast_act(σ, ...)</code>, that needs to be done by the user if needed.</p></div><p><strong>Arguments</strong></p><ul><li><p><code>σ</code>: Activation function</p></li><li><p><code>x</code>: Input array</p></li></ul><p><strong>Returns</strong></p><ul><li>Output Array with the same size as <code>x</code></li></ul><p><a href="https://github.com/LuxDL/LuxLib.jl/blob/v0.3.17/src/api/fast_activation.jl#L1-L21" target="_blank" rel="noreferrer">source</a></p></div><br>', 4);
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, [
    _hoisted_1,
    createBaseVNode("div", _hoisted_14, [
      _hoisted_15,
      createTextVNode(" "),
      _hoisted_16,
      createTextVNode(" — "),
      _hoisted_17,
      createTextVNode(". "),
      _hoisted_18,
      createBaseVNode("ul", null, [
        _hoisted_23,
        createBaseVNode("li", null, [
          createBaseVNode("p", null, [
            _hoisted_24,
            createTextVNode(": Inverse of the probability ("),
            createBaseVNode("mjx-container", _hoisted_25, [
              (openBlock(), createElementBlock("svg", _hoisted_26, _hoisted_28)),
              _hoisted_29
            ]),
            createTextVNode(")")
          ])
        ])
      ]),
      _hoisted_30
    ]),
    _hoisted_35,
    _hoisted_36,
    createBaseVNode("div", _hoisted_37, [
      _hoisted_38,
      createTextVNode(" "),
      _hoisted_39,
      createTextVNode(" — "),
      _hoisted_40,
      createTextVNode(". "),
      _hoisted_41,
      createBaseVNode("p", null, [
        createTextVNode("Batch Normalization computes the mean and variance for each "),
        createBaseVNode("mjx-container", _hoisted_43, [
          (openBlock(), createElementBlock("svg", _hoisted_44, _hoisted_46)),
          _hoisted_47
        ]),
        createTextVNode(" input slice and normalises the input accordingly.")
      ]),
      _hoisted_48,
      createBaseVNode("ul", null, [
        _hoisted_49,
        createBaseVNode("li", null, [
          createBaseVNode("p", null, [
            _hoisted_50,
            createTextVNode(": Scale factor ("),
            createBaseVNode("mjx-container", _hoisted_51, [
              (openBlock(), createElementBlock("svg", _hoisted_52, _hoisted_54)),
              _hoisted_55
            ]),
            createTextVNode(") (can be "),
            _hoisted_56,
            createTextVNode(")")
          ])
        ]),
        createBaseVNode("li", null, [
          createBaseVNode("p", null, [
            _hoisted_57,
            createTextVNode(": Bias factor ("),
            createBaseVNode("mjx-container", _hoisted_58, [
              (openBlock(), createElementBlock("svg", _hoisted_59, _hoisted_61)),
              _hoisted_62
            ]),
            createTextVNode(") (can be "),
            _hoisted_63,
            createTextVNode(")")
          ])
        ]),
        _hoisted_64
      ]),
      _hoisted_67
    ]),
    _hoisted_76,
    createBaseVNode("div", _hoisted_77, [
      _hoisted_78,
      createTextVNode(" "),
      _hoisted_79,
      createTextVNode(" — "),
      _hoisted_80,
      createTextVNode(". "),
      _hoisted_81,
      createBaseVNode("ul", null, [
        _hoisted_85,
        createBaseVNode("li", null, [
          createBaseVNode("p", null, [
            _hoisted_86,
            createTextVNode(": Scale factor ("),
            createBaseVNode("mjx-container", _hoisted_87, [
              (openBlock(), createElementBlock("svg", _hoisted_88, _hoisted_90)),
              _hoisted_91
            ]),
            createTextVNode(") (can be "),
            _hoisted_92,
            createTextVNode(")")
          ])
        ]),
        createBaseVNode("li", null, [
          createBaseVNode("p", null, [
            _hoisted_93,
            createTextVNode(": Bias factor ("),
            createBaseVNode("mjx-container", _hoisted_94, [
              (openBlock(), createElementBlock("svg", _hoisted_95, _hoisted_97)),
              _hoisted_98
            ]),
            createTextVNode(") (can be "),
            _hoisted_99,
            createTextVNode(")")
          ])
        ])
      ]),
      _hoisted_100
    ]),
    _hoisted_111,
    createBaseVNode("div", _hoisted_112, [
      _hoisted_113,
      createTextVNode(" "),
      _hoisted_114,
      createTextVNode(" — "),
      _hoisted_115,
      createTextVNode(". "),
      _hoisted_116,
      createBaseVNode("p", null, [
        createTextVNode("Instance Normalization computes the mean and variance for each "),
        createBaseVNode("mjx-container", _hoisted_118, [
          (openBlock(), createElementBlock("svg", _hoisted_119, _hoisted_121)),
          _hoisted_122
        ]),
        createTextVNode(" input slice and normalises the input accordingly.")
      ]),
      _hoisted_123,
      createBaseVNode("ul", null, [
        _hoisted_124,
        createBaseVNode("li", null, [
          createBaseVNode("p", null, [
            _hoisted_125,
            createTextVNode(": Scale factor ("),
            createBaseVNode("mjx-container", _hoisted_126, [
              (openBlock(), createElementBlock("svg", _hoisted_127, _hoisted_129)),
              _hoisted_130
            ]),
            createTextVNode(") (can be "),
            _hoisted_131,
            createTextVNode(")")
          ])
        ]),
        createBaseVNode("li", null, [
          createBaseVNode("p", null, [
            _hoisted_132,
            createTextVNode(": Bias factor ("),
            createBaseVNode("mjx-container", _hoisted_133, [
              (openBlock(), createElementBlock("svg", _hoisted_134, _hoisted_136)),
              _hoisted_137
            ]),
            createTextVNode(") (can be "),
            _hoisted_138,
            createTextVNode(")")
          ])
        ]),
        _hoisted_139
      ]),
      _hoisted_140
    ]),
    _hoisted_147,
    createBaseVNode("div", _hoisted_148, [
      _hoisted_149,
      createTextVNode(" "),
      _hoisted_150,
      createTextVNode(" — "),
      _hoisted_151,
      createTextVNode(". "),
      _hoisted_152,
      createBaseVNode("p", null, [
        createTextVNode("Given an input array "),
        createBaseVNode("mjx-container", _hoisted_154, [
          (openBlock(), createElementBlock("svg", _hoisted_155, _hoisted_157)),
          _hoisted_158
        ]),
        createTextVNode(", this layer computes")
      ]),
      createBaseVNode("mjx-container", _hoisted_159, [
        (openBlock(), createElementBlock("svg", _hoisted_160, _hoisted_162)),
        _hoisted_163
      ]),
      _hoisted_164,
      _hoisted_165,
      createBaseVNode("ul", null, [
        _hoisted_166,
        createBaseVNode("li", null, [
          createBaseVNode("p", null, [
            _hoisted_167,
            createTextVNode(": Scale factor ("),
            createBaseVNode("mjx-container", _hoisted_168, [
              (openBlock(), createElementBlock("svg", _hoisted_169, _hoisted_171)),
              _hoisted_172
            ]),
            createTextVNode(") (can be "),
            _hoisted_173,
            createTextVNode(")")
          ])
        ]),
        createBaseVNode("li", null, [
          createBaseVNode("p", null, [
            _hoisted_174,
            createTextVNode(": Bias factor ("),
            createBaseVNode("mjx-container", _hoisted_175, [
              (openBlock(), createElementBlock("svg", _hoisted_176, _hoisted_178)),
              _hoisted_179
            ]),
            createTextVNode(") (can be "),
            _hoisted_180,
            createTextVNode(")")
          ])
        ]),
        _hoisted_181
      ]),
      _hoisted_182
    ]),
    _hoisted_189
  ]);
}
const LuxLib = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  LuxLib as default
};
