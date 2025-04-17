import{_ as t,o as a,c as l,a2 as o}from"./chunks/framework.ZBMMAXEM.js";const p=JSON.parse('{"title":"Automatic Differentiation","description":"","frontmatter":{},"headers":[],"relativePath":"manual/autodiff.md","filePath":"manual/autodiff.md","lastUpdated":null}'),r={name:"manual/autodiff.md"};function s(i,e,f,n,d,c){return a(),l("div",null,e[0]||(e[0]=[o('<h1 id="autodiff-lux" tabindex="-1">Automatic Differentiation <a class="header-anchor" href="#autodiff-lux" aria-label="Permalink to &quot;Automatic Differentiation {#autodiff-lux}&quot;">​</a></h1><p>Lux is not an AD package, but it composes well with most of the AD packages available in the Julia ecosystem. This document lists the current level of support for various AD packages in Lux. Additionally, we provide some convenience functions for working with AD.</p><h2 id="overview" tabindex="-1">Overview <a class="header-anchor" href="#overview" aria-label="Permalink to &quot;Overview&quot;">​</a></h2><table tabindex="0"><thead><tr><th style="text-align:left;">AD Package</th><th style="text-align:left;">Mode</th><th style="text-align:left;">CPU</th><th style="text-align:left;">GPU</th><th style="text-align:left;">Nested 2nd Order AD</th><th style="text-align:left;">Support Class</th></tr></thead><tbody><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/ChainRules.jl" target="_blank" rel="noreferrer"><code>ChainRules.jl</code></a><sup class="footnote-ref"><a href="#fn1" id="fnref1">[1]</a></sup></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">Tier I</td></tr><tr><td style="text-align:left;"><a href="https://github.com/EnzymeAD/Enzyme.jl" target="_blank" rel="noreferrer"><code>Enzyme.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2">[2]</a></sup></td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:1">[2:1]</a></sup></td><td style="text-align:left;">Tier I<sup class="footnote-ref"><a href="#fn3" id="fnref3">[3]</a></sup></td></tr><tr><td style="text-align:left;"><a href="https://github.com/FluxML/Zygote.jl" target="_blank" rel="noreferrer"><code>Zygote.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">Tier I</td></tr><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/ForwardDiff.jl" target="_blank" rel="noreferrer"><code>ForwardDiff.jl</code></a></td><td style="text-align:left;">Forward</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">Tier I</td></tr><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/ReverseDiff.jl" target="_blank" rel="noreferrer"><code>ReverseDiff.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">❌</td><td style="text-align:left;">❌</td><td style="text-align:left;">Tier II</td></tr><tr><td style="text-align:left;"><a href="https://github.com/FluxML/Tracker.jl" target="_blank" rel="noreferrer"><code>Tracker.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">❌</td><td style="text-align:left;">Tier II</td></tr><tr><td style="text-align:left;"><a href="https://github.com/compintell/Mooncake.jl" target="_blank" rel="noreferrer"><code>Mooncake.jl</code></a></td><td style="text-align:left;">Reverse</td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:2">[2:2]</a></sup></td><td style="text-align:left;">❌</td><td style="text-align:left;">❌</td><td style="text-align:left;">Tier III</td></tr><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/Diffractor.jl" target="_blank" rel="noreferrer"><code>Diffractor.jl</code></a></td><td style="text-align:left;">Forward</td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:3">[2:3]</a></sup></td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:4">[2:4]</a></sup></td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:5">[2:5]</a></sup></td><td style="text-align:left;">Tier III</td></tr></tbody></table><h2 id="autodiff-recommendations" tabindex="-1">Recommendations <a class="header-anchor" href="#autodiff-recommendations" aria-label="Permalink to &quot;Recommendations {#autodiff-recommendations}&quot;">​</a></h2><ul><li><p>For CPU Usacases:</p><ol><li><p>Use <code>Zygote.jl</code> for the best performance. This is the most reliable and fastest option for CPU for the time-being. (We are working on faster Enzyme support for CPU)</p></li><li><p>Use <code>Enzyme.jl</code>, if there are mutations in the code and/or <code>Zygote.jl</code> fails.</p></li><li><p>If <code>Enzyme.jl</code> fails for some reason, (open an issue and) try <code>ReverseDiff.jl</code> (<a href="https://juliadiff.org/ReverseDiff.jl/dev/api/#ReverseDiff.compile" target="_blank" rel="noreferrer">possibly with compiled mode</a>).</p></li></ol></li><li><p>For GPU Usacases:</p><ol><li>Use <code>Zygote.jl</code> for the best performance. This is the most reliable and fastest option for GPU for the time-being. We are working on supporting <code>Enzyme.jl</code> for GPU as well.</li></ol></li></ul><h2 id="Support-Class" tabindex="-1">Support Class <a class="header-anchor" href="#Support-Class" aria-label="Permalink to &quot;Support Class {#Support-Class}&quot;">​</a></h2><ol><li><p><strong>Tier I</strong>: These packages are fully supported and have been tested extensively. Often have special rules to enhance performance. Issues for these backends take the highest priority.</p></li><li><p><strong>Tier II</strong>: These packages are supported and extensively tested but often don&#39;t have the best performance. Issues against these backends are less critical, but we fix them when possible. (Some specific edge cases, especially with AMDGPU, are known to fail here)</p></li><li><p><strong>Tier III</strong>: We don&#39;t know if these packages currently work with Lux. We&#39;d love to add tests for these backends, but currently these are not our priority.</p></li></ol><h2 id="footnotes" tabindex="-1">Footnotes <a class="header-anchor" href="#footnotes" aria-label="Permalink to &quot;Footnotes&quot;">​</a></h2><hr class="footnotes-sep"><section class="footnotes"><ol class="footnotes-list"><li id="fn1" class="footnote-item"><p>Note that <code>ChainRules.jl</code> is not really an AD package, but we have first-class support for packages that use <code>rrules</code>. <a href="#fnref1" class="footnote-backref">↩︎</a></p></li><li id="fn2" class="footnote-item"><p>This feature is supported downstream, but we don&#39;t extensively test it to ensure that it works with Lux. <a href="#fnref2" class="footnote-backref">↩︎</a> <a href="#fnref2:1" class="footnote-backref">↩︎</a> <a href="#fnref2:2" class="footnote-backref">↩︎</a> <a href="#fnref2:3" class="footnote-backref">↩︎</a> <a href="#fnref2:4" class="footnote-backref">↩︎</a> <a href="#fnref2:5" class="footnote-backref">↩︎</a></p></li><li id="fn3" class="footnote-item"><p>Currently Enzyme outperforms other AD packages in terms of CPU performance. However, there are some edge cases where it might not work with Lux. We are working on improving the compatibility. Please report any issues you encounter. <a href="#fnref3" class="footnote-backref">↩︎</a></p></li></ol></section>',11)]))}const u=t(r,[["render",s]]);export{p as __pageData,u as default};
