import{_ as e,c as l,o as a,al as o}from"./chunks/framework.BL7q4BmR.js";const p=JSON.parse('{"title":"Automatic Differentiation","description":"","frontmatter":{},"headers":[],"relativePath":"manual/autodiff.md","filePath":"manual/autodiff.md","lastUpdated":null}'),s={name:"manual/autodiff.md"};function r(n,t,i,f,d,c){return a(),l("div",null,t[0]||(t[0]=[o('<h1 id="autodiff-lux" tabindex="-1">Automatic Differentiation <a class="header-anchor" href="#autodiff-lux" aria-label="Permalink to &quot;Automatic Differentiation {#autodiff-lux}&quot;">​</a></h1><p>Lux is not an AD package, but it composes well with most of the AD packages available in the Julia ecosystem. This document lists the current level of support for various AD packages in Lux. Additionally, we provide some convenience functions for working with AD.</p><h2 id="overview" tabindex="-1">Overview <a class="header-anchor" href="#overview" aria-label="Permalink to &quot;Overview&quot;">​</a></h2><table tabindex="0"><thead><tr><th style="text-align:left;">AD Package</th><th style="text-align:left;">Mode</th><th style="text-align:left;">CPU</th><th style="text-align:left;">GPU</th><th style="text-align:left;">TPU</th><th style="text-align:left;">Nested 2nd Order AD</th><th style="text-align:left;">Support Class</th></tr></thead><tbody><tr><td style="text-align:left;"><a href="https://github.com/EnzymeAD/Reactant.jl" target="_blank" rel="noreferrer"><code>Reactant.jl</code></a><sup class="footnote-ref"><a href="#fn1" id="fnref1">[1]</a></sup> + <a href="https://github.com/EnzymeAD/Enzyme.jl" target="_blank" rel="noreferrer"><code>Enzyme.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">Tier I</td></tr><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/ChainRules.jl" target="_blank" rel="noreferrer"><code>ChainRules.jl</code></a><sup class="footnote-ref"><a href="#fn2" id="fnref2">[2]</a></sup></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">❌</td><td style="text-align:left;">✔️</td><td style="text-align:left;">Tier I</td></tr><tr><td style="text-align:left;"><a href="https://github.com/EnzymeAD/Enzyme.jl" target="_blank" rel="noreferrer"><code>Enzyme.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn3" id="fnref3">[3]</a></sup></td><td style="text-align:left;">❌</td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn3" id="fnref3:1">[3:1]</a></sup></td><td style="text-align:left;">Tier I<sup class="footnote-ref"><a href="#fn4" id="fnref4">[4]</a></sup></td></tr><tr><td style="text-align:left;"><a href="https://github.com/FluxML/Zygote.jl" target="_blank" rel="noreferrer"><code>Zygote.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">❌</td><td style="text-align:left;">✔️</td><td style="text-align:left;">Tier I</td></tr><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/ForwardDiff.jl" target="_blank" rel="noreferrer"><code>ForwardDiff.jl</code></a></td><td style="text-align:left;">Forward</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">❌</td><td style="text-align:left;">✔️</td><td style="text-align:left;">Tier I</td></tr><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/ReverseDiff.jl" target="_blank" rel="noreferrer"><code>ReverseDiff.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">❌</td><td style="text-align:left;">❌</td><td style="text-align:left;">❌</td><td style="text-align:left;">Tier II</td></tr><tr><td style="text-align:left;"><a href="https://github.com/FluxML/Tracker.jl" target="_blank" rel="noreferrer"><code>Tracker.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">❌</td><td style="text-align:left;">❌</td><td style="text-align:left;">Tier II</td></tr><tr><td style="text-align:left;"><a href="https://github.com/compintell/Mooncake.jl" target="_blank" rel="noreferrer"><code>Mooncake.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn3" id="fnref3:2">[3:2]</a></sup></td><td style="text-align:left;">❌</td><td style="text-align:left;">❌</td><td style="text-align:left;">❌</td><td style="text-align:left;">Tier III</td></tr><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/Diffractor.jl" target="_blank" rel="noreferrer"><code>Diffractor.jl</code></a></td><td style="text-align:left;">Forward</td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn3" id="fnref3:3">[3:3]</a></sup></td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn3" id="fnref3:4">[3:4]</a></sup></td><td style="text-align:left;">❌</td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn3" id="fnref3:5">[3:5]</a></sup></td><td style="text-align:left;">Tier III</td></tr></tbody></table><h2 id="autodiff-recommendations" tabindex="-1">Recommendations <a class="header-anchor" href="#autodiff-recommendations" aria-label="Permalink to &quot;Recommendations {#autodiff-recommendations}&quot;">​</a></h2><ul><li><p>For CPU Use cases:</p><ol><li><p>Use <code>Reactant.jl</code> + <code>Enzyme.jl</code> for the best performance as well as mutation-support. When available, this is the most reliable and fastest option.</p></li><li><p>Use <code>Zygote.jl</code> for the best performance without <code>Reactant.jl</code>. This is the most reliable and fastest option for CPU for the time-being. (We are working on faster Enzyme support for CPU)</p></li><li><p>Use <code>Enzyme.jl</code>, if there are mutations in the code and/or <code>Zygote.jl</code> fails.</p></li><li><p>If <code>Enzyme.jl</code> fails for some reason, (open an issue and) try <code>ReverseDiff.jl</code> (<a href="https://juliadiff.org/ReverseDiff.jl/dev/api/#ReverseDiff.compile" target="_blank" rel="noreferrer">possibly with compiled mode</a>).</p></li></ol></li><li><p>For GPU Use cases:</p><ol><li><p>Use <code>Reactant.jl</code> + <code>Enzyme.jl</code> for the best performance. This is the most reliable and fastest option, but presently only supports NVIDIA GPU&#39;s. AMD GPUs are currently not supported.</p></li><li><p>Use <code>Zygote.jl</code> for the best performance on non-NVIDIA GPUs. This is the most reliable and fastest non-<code>Reactant.jl</code> option for GPU for the time-being. We are working on supporting <code>Enzyme.jl</code> without <code>Reactant.jl</code> for GPU as well.</p></li></ol></li><li><p>For TPU Use cases:</p><ol><li>Use <code>Reactant.jl</code>. This is the only supported (and fastest) option.</li></ol></li></ul><h2 id="Support-Class" tabindex="-1">Support Class <a class="header-anchor" href="#Support-Class" aria-label="Permalink to &quot;Support Class {#Support-Class}&quot;">​</a></h2><ol><li><p><strong>Tier I</strong>: These packages are fully supported and have been tested extensively. Often have special rules to enhance performance. Issues for these backends take the highest priority.</p></li><li><p><strong>Tier II</strong>: These packages are supported and extensively tested but often don&#39;t have the best performance. Issues against these backends are less critical, but we fix them when possible. (Some specific edge cases, especially with AMDGPU, are known to fail here)</p></li><li><p><strong>Tier III</strong>: We don&#39;t know if these packages currently work with Lux. We&#39;d love to add tests for these backends, but currently these are not our priority.</p></li></ol><h2 id="footnotes" tabindex="-1">Footnotes <a class="header-anchor" href="#footnotes" aria-label="Permalink to &quot;Footnotes&quot;">​</a></h2><hr class="footnotes-sep"><section class="footnotes"><ol class="footnotes-list"><li id="fn1" class="footnote-item"><p>Note that <code>Reactant.jl</code> is not really an AD package, but a tool for compiling functions, including the use of EnzymeMLIR for AD via <code>Enzyme.jl</code>. We have first-class support for the usage of <code>Reactant.jl</code> for inference and training when using <code>Enzyme.jl</code> for differentiation. <a href="#fnref1" class="footnote-backref">↩︎</a></p></li><li id="fn2" class="footnote-item"><p>Note that <code>ChainRules.jl</code> is not really an AD package, but we have first-class support for packages that use <code>rrules</code>. <a href="#fnref2" class="footnote-backref">↩︎</a></p></li><li id="fn3" class="footnote-item"><p>This feature is supported downstream, but we don&#39;t extensively test it to ensure that it works with Lux. <a href="#fnref3" class="footnote-backref">↩︎</a> <a href="#fnref3:1" class="footnote-backref">↩︎</a> <a href="#fnref3:2" class="footnote-backref">↩︎</a> <a href="#fnref3:3" class="footnote-backref">↩︎</a> <a href="#fnref3:4" class="footnote-backref">↩︎</a> <a href="#fnref3:5" class="footnote-backref">↩︎</a></p></li><li id="fn4" class="footnote-item"><p>Currently Enzyme outperforms other AD packages in terms of CPU performance. However, there are some edge cases where it might not work with Lux when not using Reactant. We are working on improving the compatibility. Please report any issues you encounter and try Reactant if something fails. <a href="#fnref4" class="footnote-backref">↩︎</a></p></li></ol></section>',11)]))}const u=e(s,[["render",r]]);export{p as __pageData,u as default};
