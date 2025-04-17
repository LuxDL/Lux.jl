import{_ as e,c as t,o as a,a4 as i}from"./chunks/framework.Bfzm2VQf.js";const k=JSON.parse('{"title":"Preferences for Lux.jl","description":"","frontmatter":{},"headers":[],"relativePath":"manual/preferences.md","filePath":"manual/preferences.md","lastUpdated":null}'),o={name:"manual/preferences.md"},s=i('<h1 id="Preferences-for-Lux.jl" tabindex="-1">Preferences for Lux.jl <a class="header-anchor" href="#Preferences-for-Lux.jl" aria-label="Permalink to &quot;Preferences for Lux.jl {#Preferences-for-Lux.jl}&quot;">​</a></h1><div class="tip custom-block"><p class="custom-block-title">How to set Preferences</p><p><a href="https://github.com/cjdoris/PreferenceTools.jl" target="_blank" rel="noreferrer">PreferenceTools.jl</a> provides an interactive way to set preferences. First run the following command:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> PreferenceTools</span></span></code></pre></div><p>Then in the pkg mode (press <code>]</code> in the REPL), run the following command:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> preference add Lux </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">preference</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;=&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">value</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span></span></code></pre></div></div><p>Lux.jl relies on several preferences to make decision on how to run your code. Here is an exhaustive list of preferences that Lux.jl uses.</p><h2 id="Nested-Automatic-Differentiation" tabindex="-1">Nested Automatic Differentiation <a class="header-anchor" href="#Nested-Automatic-Differentiation" aria-label="Permalink to &quot;Nested Automatic Differentiation {#Nested-Automatic-Differentiation}&quot;">​</a></h2><ol><li><code>automatic_nested_ad_switching</code> - Set this to <code>false</code> to disable automatic switching of backends for nested automatic differentiation. See the manual section on <a href="/v0.5.64/manual/nested_autodiff#nested_autodiff">nested automatic differentiation</a> for more details.</li></ol><h2 id="gpu-aware-mpi-preferences" tabindex="-1">GPU-Aware MPI Support <a class="header-anchor" href="#gpu-aware-mpi-preferences" aria-label="Permalink to &quot;GPU-Aware MPI Support {#gpu-aware-mpi-preferences}&quot;">​</a></h2><p>If you are using a custom MPI build that supports CUDA or ROCM, you can use the following preferences with <a href="https://github.com/JuliaPackaging/Preferences.jl" target="_blank" rel="noreferrer">Preferences.jl</a>:</p><ol><li><p><code>cuda_aware_mpi</code> - Set this to <code>true</code> if your MPI build is CUDA aware.</p></li><li><p><code>rocm_aware_mpi</code> - Set this to <code>true</code> if your MPI build is ROCM aware.</p></li></ol><p>By default, both of these preferences are set to <code>false</code>.</p><h2 id="GPU-Backend-Selection" tabindex="-1">GPU Backend Selection <a class="header-anchor" href="#GPU-Backend-Selection" aria-label="Permalink to &quot;GPU Backend Selection {#GPU-Backend-Selection}&quot;">​</a></h2><ol><li><code>gpu_backend</code> - Set this to bypass the automatic backend selection and use a specific gpu backend. Valid options are &quot;cuda&quot;, &quot;rocm&quot;, &quot;metal&quot;, and &quot;oneapi&quot;. This preference needs to be set for <code>LuxDeviceUtils</code> package. It is recommended to use <a href="/v0.5.64/api/Accelerator_Support/LuxDeviceUtils#LuxDeviceUtils.gpu_backend!"><code>LuxDeviceUtils.gpu_backend!</code></a> to set this preference.</li></ol><h2 id="automatic-eltypes-preference" tabindex="-1">Automatic Eltype Conversion <a class="header-anchor" href="#automatic-eltypes-preference" aria-label="Permalink to &quot;Automatic Eltype Conversion {#automatic-eltypes-preference}&quot;">​</a></h2><ol><li><code>eltype_mismatch_handling</code> - Preference controlling what happens when layers get different eltypes as input. See the documentation on <a href="/v0.5.64/api/Lux/utilities#Lux.match_eltype"><code>match_eltype</code></a> for more details.</li></ol><h2 id="dispatch-doctor-preference" tabindex="-1">Dispatch Doctor <a class="header-anchor" href="#dispatch-doctor-preference" aria-label="Permalink to &quot;Dispatch Doctor {#dispatch-doctor-preference}&quot;">​</a></h2><ol><li><code>instability_check</code> - Preference controlling the dispatch doctor. See the documentation on <a href="/v0.5.64/api/Lux/utilities#Lux.set_dispatch_doctor_preferences!"><code>Lux.set_dispatch_doctor_preferences!</code></a> for more details. The preferences need to be set for <code>LuxCore</code> and <code>LuxLib</code> packages. Both of them default to <code>disable</code>.</li></ol><ul><li><p>Setting the <code>LuxCore</code> preference sets the check at the level of <code>LuxCore.apply</code>. This essentially activates the dispatch doctor for all Lux layers.</p></li><li><p>Setting the <code>LuxLib</code> preference sets the check at the level of functional layer of Lux, for example, <a href="/v0.5.64/api/Building_Blocks/LuxLib#LuxLib.API.fused_dense_bias_activation"><code>fused_dense_bias_activation</code></a>. These functions are supposed to be type stable for common input types and can be used to guarantee type stability.</p></li></ul>',16),r=[s];function c(n,l,d,p,h,u){return a(),t("div",null,r)}const m=e(o,[["render",c]]);export{k as __pageData,m as default};
