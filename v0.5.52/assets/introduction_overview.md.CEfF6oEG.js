import{_ as e,c as t,o as r,a3 as a}from"./chunks/framework.B5P1ePk0.js";const g=JSON.parse('{"title":"Why we wrote Lux?","description":"","frontmatter":{},"headers":[],"relativePath":"introduction/overview.md","filePath":"introduction/overview.md","lastUpdated":null}'),s={name:"introduction/overview.md"},i=a('<h1 id="Why-we-wrote-Lux?" tabindex="-1">Why we wrote Lux? <a class="header-anchor" href="#Why-we-wrote-Lux?" aria-label="Permalink to &quot;Why we wrote Lux? {#Why-we-wrote-Lux?}&quot;">​</a></h1><p>Julia already has quite a few well established Neural Network Frameworks – <a href="https://fluxml.ai/" target="_blank" rel="noreferrer">Flux</a> &amp; <a href="https://denizyuret.github.io/Knet.jl/latest/" target="_blank" rel="noreferrer">KNet</a>. However, certain design elements – <strong>Coupled Model and Parameters</strong> &amp; <strong>Internal Mutations</strong> – associated with these frameworks make them less compiler and user friendly. Making changes to address these problems in the respective frameworks would be too disruptive for users. Here comes in <code>Lux</code>: a neural network framework built completely using pure functions to make it both compiler and autodiff friendly.</p><h2 id="Design-Principles" tabindex="-1">Design Principles <a class="header-anchor" href="#Design-Principles" aria-label="Permalink to &quot;Design Principles {#Design-Principles}&quot;">​</a></h2><ul><li><p><strong>Layers must be immutable</strong> – cannot store any parameter/state but rather store the information to construct them</p></li><li><p><strong>Layers are pure functions</strong></p></li><li><p><strong>Layers return a Tuple containing the result and the updated state</strong></p></li><li><p><strong>Given same inputs the outputs must be same</strong> – yes this must hold true even for stochastic functions. Randomness must be controlled using <code>rng</code>s passed in the state.</p></li><li><p><strong>Easily extensible</strong></p></li><li><p><strong>Extensive Testing</strong> – All layers and features are tested across all supported AD backends across all supported hardware backends.</p></li></ul><h2 id="Why-use-Lux-over-Flux?" tabindex="-1">Why use Lux over Flux? <a class="header-anchor" href="#Why-use-Lux-over-Flux?" aria-label="Permalink to &quot;Why use Lux over Flux? {#Why-use-Lux-over-Flux?}&quot;">​</a></h2><ul><li><p><strong>Neural Networks for SciML</strong>: For SciML Applications (Neural ODEs, Deep Equilibrium Models) solvers typically expect a monolithic parameter vector. Flux enables this via its <code>destructure</code> mechanism, but <code>destructure</code> comes with various <a href="https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.destructure" target="_blank" rel="noreferrer">edge cases and limitations</a>. Lux forces users to make an explicit distinction between state variables and parameter variables to avoid these issues. Also, it comes battery-included for distributed training.</p></li><li><p><strong>Sensible display of Custom Layers</strong> – Ever wanted to see Pytorch like Network printouts or wondered how to extend the pretty printing of Flux&#39;s layers? Lux handles all of that by default.</p></li><li><p><strong>Truly immutable models</strong> - No <em>unexpected internal mutations</em> since all layers are implemented as pure functions. All layers are also <em>deterministic</em> given the parameters and state: if a layer is supposed to be stochastic (say <code>Dropout</code>), the state must contain a seed which is then updated after the function call.</p></li><li><p><strong>Easy Parameter Manipulation</strong> – By separating parameter data and layer structures, Lux makes implementing <code>WeightNorm</code>, <code>SpectralNorm</code>, etc. downright trivial. Without this separation, it is much harder to pass such parameters around without mutations which AD systems don&#39;t like.</p></li><li><p><strong>Small Neural Networks on CPU</strong> – Lux is developed for training large neural networks. For smaller architectures, we recommend using <a href="https://github.com/PumasAI/SimpleChains.jl" target="_blank" rel="noreferrer">SimpleChains.jl</a> or even better use it in conjunction with Lux via <a href="/v0.5.52/api/Lux/interop#Lux.ToSimpleChainsAdaptor"><code>ToSimpleChainsAdaptor</code></a>.</p></li><li><p><strong>Reliability</strong> – We have learned from the mistakes of the past with Flux and everything in our core framework is extensively tested, along with downstream CI to ensure that everything works as expected.</p></li></ul><h2 id="Why-not-use-Lux-(and-Julia-for-traditional-Deep-Learning-in-general)-?" tabindex="-1">Why not use Lux (and Julia for traditional Deep Learning in general) ? <a class="header-anchor" href="#Why-not-use-Lux-(and-Julia-for-traditional-Deep-Learning-in-general)-?" aria-label="Permalink to &quot;Why not use Lux (and Julia for traditional Deep Learning in general) ? {#Why-not-use-Lux-(and-Julia-for-traditional-Deep-Learning-in-general)-?}&quot;">​</a></h2><ul><li><p><strong>Lack of Large Models Support</strong> – Classical deep learning is not Lux&#39;s primary focus. For these, python frameworks like PyTorch and Jax are better suited.</p></li><li><p><strong>XLA Support</strong> – Lux doesn&#39;t compile to XLA which means no TPU support unfortunately.</p></li></ul>',8),o=[i];function n(l,u,d,p,c,h){return r(),t("div",null,o)}const f=e(s,[["render",n]]);export{g as __pageData,f as default};
