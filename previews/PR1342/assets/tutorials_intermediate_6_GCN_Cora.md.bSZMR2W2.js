import{_ as a,c as n,o as e,al as i}from"./chunks/framework.BZqo-lGB.js";const o=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),p={name:"tutorials/intermediate/6_GCN_Cora.md"};function t(c,s,l,r,h,f){return e(),n("div",null,s[0]||(s[0]=[i(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Reactant,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    MLDatasets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Statistics,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Enzyme,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    GNNGraphs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ConcreteStructs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Printf,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    OneHotArrays,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Optimisers</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; force</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(::MLDataDevices.CPUDevice) (generic function with 1 method)</span></span></code></pre></div><h2 id="Loading-Cora-Dataset" tabindex="-1">Loading Cora Dataset <a class="header-anchor" href="#Loading-Cora-Dataset" aria-label="Permalink to &quot;Loading Cora Dataset {#Loading-Cora-Dataset}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Cora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">graphs[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gnngraph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GNNGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_index; ndata</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data, edata</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_data, gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">num_nodes</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">features,</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">targets, data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">metadata[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;classes&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # We use a dense matrix here to avoid incompatibility with Reactant</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">adjacency_matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gnngraph)),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # We use this since Reactant doesn&#39;t yet support gather adjoint</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">140</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">141</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">640</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1709</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2708</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>loadcora (generic function with 1 method)</span></span></code></pre></div><h2 id="Model-Definition" tabindex="-1">Model Definition <a class="header-anchor" href="#Model-Definition" aria-label="Permalink to &quot;Model Definition {#Model-Definition}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dense</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> adj</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, h_dim, out_dim; nb_layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dropout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    layer_sizes </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, [h_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nb_layers])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn_layers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (in_dim, out_dim) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> zip</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)], layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    last_layer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dropout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dropout)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; gcn_layers, dropout, last_layer) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj, mask)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> layer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gcn_layers</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> relu</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> last_layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj))[:, mask]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>GCN (generic function with 1 method)</span></span></code></pre></div><h2 id="Helper-Functions" tabindex="-1">Helper Functions <a class="header-anchor" href="#Helper-Functions" aria-label="Permalink to &quot;Helper Functions {#Helper-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y, adj, mask))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj, mask), ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; agg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">mean, logits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))(y_pred, y[:, mask])</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st, (; y_pred)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    hidden_dim</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dropout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    nb_layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    use_bias</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lr</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.001</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    weight_decay</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    patience</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    epochs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">200</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">seed!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    features, targets, adj, (train_idx, val_idx, test_idx) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(features, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), hidden_dim, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); nb_layers, dropout, use_bias)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, gcn))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    opt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(weight_decay) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lr) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AdamW</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; eta</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lr, lambda</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">weight_decay)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gcn, ps, st, opt)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Total Trainable Parameters: %0.4f M</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">parameterlength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ps) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0e6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_loss_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gcn, ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st), (features, targets, adj, val_idx))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, train_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_config</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        dot_general_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        convolution_precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PrecisionConfig</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HIGH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, val_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Inf</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">epochs</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss_function,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (features, targets, adj, train_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            return_gradients</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                train_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, train_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, train_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            val_loss_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                gcn,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                (features, targets, adj, val_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                val_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, val_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, val_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch %3d</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Train Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Train Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Val Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 Val Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss train_acc val_loss val_acc</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> best_loss_val</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loss</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> patience</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Early Stopping at Epoch %d</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                break</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    test_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            gcn,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (features, targets, adj, test_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    test_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, test_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, test_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Test Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Test Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> test_loss test_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-05-29 04:05:12.738367: I external/xla/xla/service/service.cc:152] XLA service 0x3a7c4910 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-05-29 04:05:12.738507: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1748491512.739585  128140 se_gpu_pjrt_client.cc:1026] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1748491512.739722  128140 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1748491512.739773  128140 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1748491512.754186  128140 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-9/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-9/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>Epoch   1	Train Loss: 16.619959	Train Acc: 10.0000%	Val Loss: 12.740502	Val Acc: 7.0000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 14.923183	Train Acc: 10.0000%	Val Loss: 13.620877	Val Acc: 7.8000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 14.522738	Train Acc: 10.0000%	Val Loss: 14.875621	Val Acc: 10.0000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 19.493610	Train Acc: 12.8571%	Val Loss: 16.247404	Val Acc: 12.0000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 20.604319	Train Acc: 11.4286%	Val Loss: 17.335497	Val Acc: 12.6000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 22.558479	Train Acc: 12.8571%	Val Loss: 18.298180	Val Acc: 12.4000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 22.433399	Train Acc: 12.1429%	Val Loss: 19.138405	Val Acc: 12.2000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 23.016899	Train Acc: 12.8571%	Val Loss: 19.768303	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 25.803362	Train Acc: 12.8571%	Val Loss: 20.177584	Val Acc: 11.0000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 28.268957	Train Acc: 13.5714%	Val Loss: 20.745937	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 33.114326	Train Acc: 13.5714%	Val Loss: 21.624283	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 24.938705	Train Acc: 13.5714%	Val Loss: 23.480240	Val Acc: 11.6000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 34.395966	Train Acc: 14.2857%	Val Loss: 26.109270	Val Acc: 13.4000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 31.589798	Train Acc: 15.0000%	Val Loss: 29.332098	Val Acc: 12.0000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 41.751286	Train Acc: 14.2857%	Val Loss: 33.007336	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 46.926792	Train Acc: 12.8571%	Val Loss: 37.315723	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 51.316521	Train Acc: 13.5714%	Val Loss: 41.900848	Val Acc: 11.6000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 63.408539	Train Acc: 12.8571%	Val Loss: 46.612488	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 57.699268	Train Acc: 13.5714%	Val Loss: 51.350391	Val Acc: 11.0000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 77.929756	Train Acc: 14.2857%	Val Loss: 56.246231	Val Acc: 11.0000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 65.499527	Train Acc: 15.0000%	Val Loss: 61.205891	Val Acc: 10.6000%</span></span>
<span class="line"><span>Early Stopping at Epoch 21</span></span>
<span class="line"><span>2025-05-29 04:07:20.826648: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_23&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:21.239759: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_21&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:21.300268: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_21&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:21.367304: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_18&#39;, 48 bytes spill stores, 48 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:21.569959: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_32&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:21.728602: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_18&#39;, 284 bytes spill stores, 284 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:21.794519: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_21&#39;, 360 bytes spill stores, 356 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:21.930788: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_18&#39;, 272 bytes spill stores, 272 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:22.087909: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_18&#39;, 48 bytes spill stores, 48 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:22.142106: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_18&#39;, 604 bytes spill stores, 608 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:22.385169: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_32&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:22.630293: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_18&#39;, 1212 bytes spill stores, 976 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:22.916916: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_18&#39;, 980 bytes spill stores, 976 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:22.986138: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_21&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:22.995425: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_21&#39;, 104 bytes spill stores, 104 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:23.940881: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_21&#39;, 32 bytes spill stores, 32 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:24.902776: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_32&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:25.520213: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_21&#39;, 996 bytes spill stores, 968 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:25.967776: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_21&#39;, 292 bytes spill stores, 292 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-29 04:07:27.089421: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_32&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1748491647.136849  128140 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748491647.138435  128140 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748491647.138451  128140 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748491647.138458  128140 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.138466  128140 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748491647.138473  128140 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748491647.138480  128140 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748491647.138487  128140 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748491647.138494  128140 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748491647.138500  128140 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-29 04:07:27.138520: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.142442  128140 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748491647.142469  128140 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748491647.142477  128140 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748491647.142484  128140 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.142491  128140 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748491647.142498  128140 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748491647.142505  128140 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748491647.142512  128140 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748491647.142519  128140 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748491647.142526  128140 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-29 04:07:27.142538: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.146091  128140 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748491647.146117  128140 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748491647.146125  128140 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748491647.146132  128140 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.146139  128140 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748491647.146146  128140 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748491647.146153  128140 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748491647.146160  128140 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748491647.146169  128140 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748491647.146177  128140 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-29 04:07:27.146187: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.149616  128140 buffer_comparator.cc:145] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748491647.149648  128140 buffer_comparator.cc:145] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748491647.149656  128140 buffer_comparator.cc:145] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748491647.149663  128140 buffer_comparator.cc:145] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.149670  128140 buffer_comparator.cc:145] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748491647.149677  128140 buffer_comparator.cc:145] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748491647.149684  128140 buffer_comparator.cc:145] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748491647.149691  128140 buffer_comparator.cc:145] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748491647.149698  128140 buffer_comparator.cc:145] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748491647.149705  128140 buffer_comparator.cc:145] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-05-29 04:07:27.149715: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.153313  128140 buffer_comparator.cc:145] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748491647.153339  128140 buffer_comparator.cc:145] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748491647.153347  128140 buffer_comparator.cc:145] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748491647.153354  128140 buffer_comparator.cc:145] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.153361  128140 buffer_comparator.cc:145] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748491647.153368  128140 buffer_comparator.cc:145] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748491647.153375  128140 buffer_comparator.cc:145] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748491647.153382  128140 buffer_comparator.cc:145] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748491647.153389  128140 buffer_comparator.cc:145] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748491647.153396  128140 buffer_comparator.cc:145] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-05-29 04:07:27.153407: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.156873  128140 buffer_comparator.cc:145] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1748491647.156899  128140 buffer_comparator.cc:145] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1748491647.156907  128140 buffer_comparator.cc:145] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1748491647.156914  128140 buffer_comparator.cc:145] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1748491647.156921  128140 buffer_comparator.cc:145] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1748491647.156928  128140 buffer_comparator.cc:145] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.156935  128140 buffer_comparator.cc:145] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1748491647.156941  128140 buffer_comparator.cc:145] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1748491647.156948  128140 buffer_comparator.cc:145] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1748491647.156955  128140 buffer_comparator.cc:145] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-05-29 04:07:27.156968: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.160521  128140 buffer_comparator.cc:145] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1748491647.160546  128140 buffer_comparator.cc:145] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1748491647.160554  128140 buffer_comparator.cc:145] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1748491647.160562  128140 buffer_comparator.cc:145] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1748491647.160569  128140 buffer_comparator.cc:145] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.160576  128140 buffer_comparator.cc:145] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1748491647.160583  128140 buffer_comparator.cc:145] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1748491647.160590  128140 buffer_comparator.cc:145] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1748491647.160597  128140 buffer_comparator.cc:145] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1748491647.160604  128140 buffer_comparator.cc:145] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-05-29 04:07:27.160614: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.164078  128140 buffer_comparator.cc:145] Difference at 0: 1084.56, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748491647.164104  128140 buffer_comparator.cc:145] Difference at 1: 1350.61, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748491647.164112  128140 buffer_comparator.cc:145] Difference at 2: 2009.8, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748491647.164120  128140 buffer_comparator.cc:145] Difference at 3: 1768.04, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748491647.164127  128140 buffer_comparator.cc:145] Difference at 4: 1240.61, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.164134  128140 buffer_comparator.cc:145] Difference at 6: 1407.03, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748491647.164141  128140 buffer_comparator.cc:145] Difference at 7: 1138.83, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748491647.164147  128140 buffer_comparator.cc:145] Difference at 8: 1417.44, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.164154  128140 buffer_comparator.cc:145] Difference at 9: 2084.44, expected 1833.77</span></span>
<span class="line"><span>E0000 00:00:1748491647.164161  128140 buffer_comparator.cc:145] Difference at 10: 1844.73, expected 1592.38</span></span>
<span class="line"><span>2025-05-29 04:07:27.164172: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.167754  128140 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.167779  128140 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748491647.167787  128140 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.167794  128140 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748491647.167801  128140 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748491647.167808  128140 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748491647.167815  128140 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748491647.167822  128140 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.167828  128140 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748491647.167835  128140 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-29 04:07:27.167846: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.171371  128140 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.171396  128140 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748491647.171404  128140 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.171411  128140 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748491647.171418  128140 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748491647.171425  128140 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748491647.171432  128140 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748491647.171439  128140 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.171446  128140 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748491647.171453  128140 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-29 04:07:27.171463: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.174940  128140 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.174965  128140 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748491647.174973  128140 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.174981  128140 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748491647.174988  128140 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748491647.174994  128140 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748491647.175001  128140 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748491647.175008  128140 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.175015  128140 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748491647.175022  128140 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-29 04:07:27.175032: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.178440  128140 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.178465  128140 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748491647.178473  128140 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.178480  128140 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748491647.178487  128140 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748491647.178494  128140 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748491647.178501  128140 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748491647.178508  128140 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.178515  128140 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748491647.178521  128140 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-29 04:07:27.178532: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.181908  128140 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.181927  128140 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748491647.181932  128140 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.181937  128140 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748491647.181942  128140 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748491647.181947  128140 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748491647.181951  128140 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748491647.181956  128140 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.181960  128140 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748491647.181965  128140 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-29 04:07:27.181972: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.185247  128140 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.185265  128140 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748491647.185270  128140 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.185275  128140 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748491647.185280  128140 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748491647.185284  128140 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748491647.185289  128140 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748491647.185293  128140 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.185298  128140 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748491647.185303  128140 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-29 04:07:27.185310: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.188584  128140 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.188603  128140 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748491647.188609  128140 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.188614  128140 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748491647.188619  128140 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748491647.188623  128140 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748491647.188639  128140 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748491647.188644  128140 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.188649  128140 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748491647.188653  128140 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-29 04:07:27.188661: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.192050  128140 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.192072  128140 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748491647.192078  128140 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.192084  128140 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748491647.192089  128140 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748491647.192094  128140 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748491647.192099  128140 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748491647.192103  128140 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748491647.192108  128140 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748491647.192112  128140 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-29 04:07:27.192120: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.195513  128140 buffer_comparator.cc:145] Difference at 0: 1100.47, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748491647.195532  128140 buffer_comparator.cc:145] Difference at 1: 1361.33, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748491647.195538  128140 buffer_comparator.cc:145] Difference at 2: 2059.82, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748491647.195543  128140 buffer_comparator.cc:145] Difference at 3: 1808.05, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748491647.195547  128140 buffer_comparator.cc:145] Difference at 4: 1265.06, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.195552  128140 buffer_comparator.cc:145] Difference at 5: 1986, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.195557  128140 buffer_comparator.cc:145] Difference at 6: 1409.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748491647.195561  128140 buffer_comparator.cc:145] Difference at 7: 1173.38, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748491647.195566  128140 buffer_comparator.cc:145] Difference at 8: 1420.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.195571  128140 buffer_comparator.cc:145] Difference at 9: 2114.57, expected 1833.77</span></span>
<span class="line"><span>2025-05-29 04:07:27.195578: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.198872  128140 buffer_comparator.cc:145] Difference at 0: 1100.47, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748491647.198889  128140 buffer_comparator.cc:145] Difference at 1: 1361.33, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748491647.198894  128140 buffer_comparator.cc:145] Difference at 2: 2059.82, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748491647.198899  128140 buffer_comparator.cc:145] Difference at 3: 1808.05, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748491647.198904  128140 buffer_comparator.cc:145] Difference at 4: 1265.06, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.198908  128140 buffer_comparator.cc:145] Difference at 5: 1986, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.198913  128140 buffer_comparator.cc:145] Difference at 6: 1409.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748491647.198918  128140 buffer_comparator.cc:145] Difference at 7: 1173.38, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748491647.198922  128140 buffer_comparator.cc:145] Difference at 8: 1420.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.198927  128140 buffer_comparator.cc:145] Difference at 9: 2114.57, expected 1833.77</span></span>
<span class="line"><span>2025-05-29 04:07:27.198934: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.202155  128140 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748491647.202172  128140 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748491647.202178  128140 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748491647.202183  128140 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748491647.202187  128140 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.202194  128140 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.202199  128140 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748491647.202204  128140 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748491647.202208  128140 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.202213  128140 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-29 04:07:27.202220: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.205500  128140 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748491647.205517  128140 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748491647.205522  128140 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748491647.205527  128140 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748491647.205532  128140 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.205536  128140 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.205541  128140 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748491647.205545  128140 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748491647.205550  128140 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.205554  128140 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-29 04:07:27.205562: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.208871  128140 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748491647.208889  128140 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748491647.208894  128140 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748491647.208899  128140 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748491647.208904  128140 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.208908  128140 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.208913  128140 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748491647.208917  128140 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748491647.208922  128140 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.208927  128140 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-29 04:07:27.208934: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.212266  128140 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748491647.212284  128140 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748491647.212290  128140 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748491647.212294  128140 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748491647.212299  128140 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.212304  128140 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.212308  128140 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748491647.212313  128140 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748491647.212318  128140 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.212323  128140 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-29 04:07:27.212331: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.215680  128140 buffer_comparator.cc:145] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1748491647.215699  128140 buffer_comparator.cc:145] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.215704  128140 buffer_comparator.cc:145] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1748491647.215709  128140 buffer_comparator.cc:145] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1748491647.215714  128140 buffer_comparator.cc:145] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1748491647.215719  128140 buffer_comparator.cc:145] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1748491647.215723  128140 buffer_comparator.cc:145] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1748491647.215728  128140 buffer_comparator.cc:145] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1748491647.215733  128140 buffer_comparator.cc:145] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1748491647.215737  128140 buffer_comparator.cc:145] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-05-29 04:07:27.215744: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.219012  128140 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748491647.219030  128140 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1748491647.219037  128140 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1748491647.219042  128140 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1748491647.219047  128140 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1748491647.219053  128140 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1748491647.219058  128140 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1748491647.219063  128140 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.77</span></span>
<span class="line"><span>E0000 00:00:1748491647.219069  128140 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.09</span></span>
<span class="line"><span>E0000 00:00:1748491647.219074  128140 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.62</span></span>
<span class="line"><span>2025-05-29 04:07:27.219081: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.222602  128140 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748491647.222620  128140 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1748491647.222637  128140 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1748491647.222643  128140 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1748491647.222647  128140 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1748491647.222653  128140 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1748491647.222659  128140 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1748491647.222663  128140 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.77</span></span>
<span class="line"><span>E0000 00:00:1748491647.222669  128140 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.09</span></span>
<span class="line"><span>E0000 00:00:1748491647.222674  128140 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.62</span></span>
<span class="line"><span>2025-05-29 04:07:27.222683: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.226003  128140 buffer_comparator.cc:145] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1748491647.226021  128140 buffer_comparator.cc:145] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.226026  128140 buffer_comparator.cc:145] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1748491647.226031  128140 buffer_comparator.cc:145] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1748491647.226036  128140 buffer_comparator.cc:145] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1748491647.226041  128140 buffer_comparator.cc:145] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1748491647.226045  128140 buffer_comparator.cc:145] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1748491647.226050  128140 buffer_comparator.cc:145] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1748491647.226054  128140 buffer_comparator.cc:145] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1748491647.226059  128140 buffer_comparator.cc:145] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-05-29 04:07:27.226066: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.229287  128140 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748491647.229304  128140 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748491647.229310  128140 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748491647.229314  128140 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748491647.229319  128140 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.229324  128140 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748491647.229328  128140 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748491647.229333  128140 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748491647.229337  128140 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748491647.229342  128140 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.77</span></span>
<span class="line"><span>2025-05-29 04:07:27.229349: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.232567  128140 buffer_comparator.cc:145] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1748491647.232585  128140 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1748491647.232591  128140 buffer_comparator.cc:145] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1748491647.232595  128140 buffer_comparator.cc:145] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1748491647.232599  128140 buffer_comparator.cc:145] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1748491647.232604  128140 buffer_comparator.cc:145] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1748491647.232608  128140 buffer_comparator.cc:145] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1748491647.232612  128140 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1748491647.232616  128140 buffer_comparator.cc:145] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1748491647.232620  128140 buffer_comparator.cc:145] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-05-29 04:07:27.232637: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.234815  128140 buffer_comparator.cc:145] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1748491647.234834  128140 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1748491647.234839  128140 buffer_comparator.cc:145] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1748491647.234843  128140 buffer_comparator.cc:145] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1748491647.234848  128140 buffer_comparator.cc:145] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1748491647.234852  128140 buffer_comparator.cc:145] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1748491647.234856  128140 buffer_comparator.cc:145] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1748491647.234860  128140 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1748491647.234864  128140 buffer_comparator.cc:145] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1748491647.234869  128140 buffer_comparator.cc:145] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-05-29 04:07:27.234875: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.237035  128140 buffer_comparator.cc:145] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1748491647.237047  128140 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1748491647.237050  128140 buffer_comparator.cc:145] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1748491647.237053  128140 buffer_comparator.cc:145] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1748491647.237056  128140 buffer_comparator.cc:145] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1748491647.237059  128140 buffer_comparator.cc:145] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1748491647.237062  128140 buffer_comparator.cc:145] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1748491647.237064  128140 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1748491647.237067  128140 buffer_comparator.cc:145] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1748491647.237070  128140 buffer_comparator.cc:145] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-05-29 04:07:27.237075: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.239157  128140 buffer_comparator.cc:145] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1748491647.239169  128140 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1748491647.239172  128140 buffer_comparator.cc:145] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1748491647.239175  128140 buffer_comparator.cc:145] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1748491647.239178  128140 buffer_comparator.cc:145] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1748491647.239181  128140 buffer_comparator.cc:145] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1748491647.239183  128140 buffer_comparator.cc:145] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1748491647.239186  128140 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1748491647.239189  128140 buffer_comparator.cc:145] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1748491647.239192  128140 buffer_comparator.cc:145] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-05-29 04:07:27.239196: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.241283  128140 buffer_comparator.cc:145] Difference at 32: -nan, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1748491647.241295  128140 buffer_comparator.cc:145] Difference at 33: -nan, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1748491647.241299  128140 buffer_comparator.cc:145] Difference at 34: -nan, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1748491647.241302  128140 buffer_comparator.cc:145] Difference at 35: -nan, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1748491647.241306  128140 buffer_comparator.cc:145] Difference at 36: -nan, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1748491647.241309  128140 buffer_comparator.cc:145] Difference at 37: -nan, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1748491647.241311  128140 buffer_comparator.cc:145] Difference at 38: -nan, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1748491647.241314  128140 buffer_comparator.cc:145] Difference at 39: -nan, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1748491647.241317  128140 buffer_comparator.cc:145] Difference at 40: -nan, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1748491647.241320  128140 buffer_comparator.cc:145] Difference at 41: -nan, expected 13.7427</span></span>
<span class="line"><span>2025-05-29 04:07:27.241324: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.243410  128140 buffer_comparator.cc:145] Difference at 32: -nan, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1748491647.243424  128140 buffer_comparator.cc:145] Difference at 33: -nan, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1748491647.243427  128140 buffer_comparator.cc:145] Difference at 34: -nan, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1748491647.243430  128140 buffer_comparator.cc:145] Difference at 35: -nan, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1748491647.243433  128140 buffer_comparator.cc:145] Difference at 36: -nan, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1748491647.243435  128140 buffer_comparator.cc:145] Difference at 37: -nan, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1748491647.243438  128140 buffer_comparator.cc:145] Difference at 38: -nan, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1748491647.243441  128140 buffer_comparator.cc:145] Difference at 39: -nan, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1748491647.243444  128140 buffer_comparator.cc:145] Difference at 40: -nan, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1748491647.243447  128140 buffer_comparator.cc:145] Difference at 41: -nan, expected 13.7427</span></span>
<span class="line"><span>2025-05-29 04:07:27.243451: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.245530  128140 buffer_comparator.cc:145] Difference at 32: -nan, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1748491647.245542  128140 buffer_comparator.cc:145] Difference at 33: -nan, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1748491647.245546  128140 buffer_comparator.cc:145] Difference at 34: -nan, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1748491647.245549  128140 buffer_comparator.cc:145] Difference at 35: -nan, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1748491647.245551  128140 buffer_comparator.cc:145] Difference at 36: -nan, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1748491647.245554  128140 buffer_comparator.cc:145] Difference at 37: -nan, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1748491647.245557  128140 buffer_comparator.cc:145] Difference at 38: -nan, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1748491647.245560  128140 buffer_comparator.cc:145] Difference at 39: -nan, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1748491647.245563  128140 buffer_comparator.cc:145] Difference at 40: -nan, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1748491647.245565  128140 buffer_comparator.cc:145] Difference at 41: -nan, expected 13.7427</span></span>
<span class="line"><span>2025-05-29 04:07:27.245570: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.247667  128140 buffer_comparator.cc:145] Difference at 64: -nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1748491647.247679  128140 buffer_comparator.cc:145] Difference at 65: -nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1748491647.247682  128140 buffer_comparator.cc:145] Difference at 66: -nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1748491647.247685  128140 buffer_comparator.cc:145] Difference at 67: -nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1748491647.247688  128140 buffer_comparator.cc:145] Difference at 68: -nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1748491647.247691  128140 buffer_comparator.cc:145] Difference at 69: -nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1748491647.247693  128140 buffer_comparator.cc:145] Difference at 70: -nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1748491647.247698  128140 buffer_comparator.cc:145] Difference at 71: -nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1748491647.247701  128140 buffer_comparator.cc:145] Difference at 72: -nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1748491647.247703  128140 buffer_comparator.cc:145] Difference at 73: -nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-29 04:07:27.247708: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.249782  128140 buffer_comparator.cc:145] Difference at 64: -nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1748491647.249794  128140 buffer_comparator.cc:145] Difference at 65: -nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1748491647.249797  128140 buffer_comparator.cc:145] Difference at 66: -nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1748491647.249800  128140 buffer_comparator.cc:145] Difference at 67: -nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1748491647.249803  128140 buffer_comparator.cc:145] Difference at 68: -nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1748491647.249806  128140 buffer_comparator.cc:145] Difference at 69: -nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1748491647.249809  128140 buffer_comparator.cc:145] Difference at 70: -nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1748491647.249812  128140 buffer_comparator.cc:145] Difference at 71: -nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1748491647.249814  128140 buffer_comparator.cc:145] Difference at 72: -nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1748491647.249817  128140 buffer_comparator.cc:145] Difference at 73: -nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-29 04:07:27.249822: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.251907  128140 buffer_comparator.cc:145] Difference at 64: -nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1748491647.251920  128140 buffer_comparator.cc:145] Difference at 65: -nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1748491647.251923  128140 buffer_comparator.cc:145] Difference at 66: -nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1748491647.251926  128140 buffer_comparator.cc:145] Difference at 67: -nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1748491647.251929  128140 buffer_comparator.cc:145] Difference at 68: -nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1748491647.251932  128140 buffer_comparator.cc:145] Difference at 69: -nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1748491647.251935  128140 buffer_comparator.cc:145] Difference at 70: -nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1748491647.251938  128140 buffer_comparator.cc:145] Difference at 71: -nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1748491647.251940  128140 buffer_comparator.cc:145] Difference at 72: -nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1748491647.251943  128140 buffer_comparator.cc:145] Difference at 73: -nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-29 04:07:27.251948: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.254052  128140 buffer_comparator.cc:145] Difference at 0: 16.5369, expected 14.4011</span></span>
<span class="line"><span>E0000 00:00:1748491647.254064  128140 buffer_comparator.cc:145] Difference at 1: 19.4176, expected 15.9904</span></span>
<span class="line"><span>E0000 00:00:1748491647.254067  128140 buffer_comparator.cc:145] Difference at 2: 16.204, expected 13.4103</span></span>
<span class="line"><span>E0000 00:00:1748491647.254071  128140 buffer_comparator.cc:145] Difference at 6: 13.1759, expected 11.4953</span></span>
<span class="line"><span>E0000 00:00:1748491647.254074  128140 buffer_comparator.cc:145] Difference at 9: 16.3002, expected 14.2452</span></span>
<span class="line"><span>E0000 00:00:1748491647.254077  128140 buffer_comparator.cc:145] Difference at 11: 15.6508, expected 13.739</span></span>
<span class="line"><span>E0000 00:00:1748491647.254080  128140 buffer_comparator.cc:145] Difference at 12: 20.6885, expected 16.297</span></span>
<span class="line"><span>E0000 00:00:1748491647.254083  128140 buffer_comparator.cc:145] Difference at 13: 17.247, expected 14.372</span></span>
<span class="line"><span>E0000 00:00:1748491647.254086  128140 buffer_comparator.cc:145] Difference at 14: 14.7694, expected 12.4213</span></span>
<span class="line"><span>E0000 00:00:1748491647.254089  128140 buffer_comparator.cc:145] Difference at 16: 17.2743, expected 15.1227</span></span>
<span class="line"><span>2025-05-29 04:07:27.254095: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.256178  128140 buffer_comparator.cc:145] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1748491647.256191  128140 buffer_comparator.cc:145] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1748491647.256194  128140 buffer_comparator.cc:145] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1748491647.256197  128140 buffer_comparator.cc:145] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1748491647.256200  128140 buffer_comparator.cc:145] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1748491647.256203  128140 buffer_comparator.cc:145] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1748491647.256205  128140 buffer_comparator.cc:145] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1748491647.256208  128140 buffer_comparator.cc:145] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1748491647.256211  128140 buffer_comparator.cc:145] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1748491647.256214  128140 buffer_comparator.cc:145] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-29 04:07:27.256218: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.258319  128140 buffer_comparator.cc:145] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1748491647.258331  128140 buffer_comparator.cc:145] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1748491647.258334  128140 buffer_comparator.cc:145] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1748491647.258337  128140 buffer_comparator.cc:145] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1748491647.258340  128140 buffer_comparator.cc:145] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1748491647.258343  128140 buffer_comparator.cc:145] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1748491647.258345  128140 buffer_comparator.cc:145] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1748491647.258348  128140 buffer_comparator.cc:145] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1748491647.258351  128140 buffer_comparator.cc:145] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1748491647.258354  128140 buffer_comparator.cc:145] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-29 04:07:27.258358: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.260423  128140 buffer_comparator.cc:145] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1748491647.260435  128140 buffer_comparator.cc:145] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1748491647.260438  128140 buffer_comparator.cc:145] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1748491647.260441  128140 buffer_comparator.cc:145] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1748491647.260444  128140 buffer_comparator.cc:145] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1748491647.260446  128140 buffer_comparator.cc:145] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1748491647.260449  128140 buffer_comparator.cc:145] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1748491647.260452  128140 buffer_comparator.cc:145] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1748491647.260455  128140 buffer_comparator.cc:145] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1748491647.260458  128140 buffer_comparator.cc:145] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-29 04:07:27.260463: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.262528  128140 buffer_comparator.cc:145] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1748491647.262540  128140 buffer_comparator.cc:145] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1748491647.262545  128140 buffer_comparator.cc:145] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1748491647.262548  128140 buffer_comparator.cc:145] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1748491647.262551  128140 buffer_comparator.cc:145] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1748491647.262553  128140 buffer_comparator.cc:145] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1748491647.262556  128140 buffer_comparator.cc:145] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1748491647.262559  128140 buffer_comparator.cc:145] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1748491647.262562  128140 buffer_comparator.cc:145] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1748491647.262564  128140 buffer_comparator.cc:145] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-29 04:07:27.262569: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.264640  128140 buffer_comparator.cc:145] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1748491647.264652  128140 buffer_comparator.cc:145] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1748491647.264655  128140 buffer_comparator.cc:145] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1748491647.264658  128140 buffer_comparator.cc:145] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1748491647.264661  128140 buffer_comparator.cc:145] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1748491647.264664  128140 buffer_comparator.cc:145] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1748491647.264667  128140 buffer_comparator.cc:145] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1748491647.264669  128140 buffer_comparator.cc:145] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1748491647.264672  128140 buffer_comparator.cc:145] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1748491647.264675  128140 buffer_comparator.cc:145] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-29 04:07:27.264680: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.266766  128140 buffer_comparator.cc:145] Difference at 128: nan, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1748491647.266778  128140 buffer_comparator.cc:145] Difference at 129: nan, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1748491647.266781  128140 buffer_comparator.cc:145] Difference at 130: nan, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1748491647.266784  128140 buffer_comparator.cc:145] Difference at 131: nan, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1748491647.266787  128140 buffer_comparator.cc:145] Difference at 132: nan, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1748491647.266790  128140 buffer_comparator.cc:145] Difference at 133: nan, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1748491647.266792  128140 buffer_comparator.cc:145] Difference at 134: nan, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1748491647.266795  128140 buffer_comparator.cc:145] Difference at 135: nan, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1748491647.266798  128140 buffer_comparator.cc:145] Difference at 136: nan, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1748491647.266801  128140 buffer_comparator.cc:145] Difference at 137: nan, expected 12.9584</span></span>
<span class="line"><span>2025-05-29 04:07:27.266805: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.268903  128140 buffer_comparator.cc:145] Difference at 128: nan, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1748491647.268916  128140 buffer_comparator.cc:145] Difference at 129: nan, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1748491647.268919  128140 buffer_comparator.cc:145] Difference at 130: nan, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1748491647.268922  128140 buffer_comparator.cc:145] Difference at 131: nan, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1748491647.268925  128140 buffer_comparator.cc:145] Difference at 132: nan, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1748491647.268929  128140 buffer_comparator.cc:145] Difference at 133: nan, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1748491647.268932  128140 buffer_comparator.cc:145] Difference at 134: nan, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1748491647.268934  128140 buffer_comparator.cc:145] Difference at 135: nan, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1748491647.268937  128140 buffer_comparator.cc:145] Difference at 136: nan, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1748491647.268940  128140 buffer_comparator.cc:145] Difference at 137: nan, expected 12.9584</span></span>
<span class="line"><span>2025-05-29 04:07:27.268945: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.276841  128140 buffer_comparator.cc:145] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1748491647.276866  128140 buffer_comparator.cc:145] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1748491647.276870  128140 buffer_comparator.cc:145] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1748491647.276873  128140 buffer_comparator.cc:145] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1748491647.276876  128140 buffer_comparator.cc:145] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.276879  128140 buffer_comparator.cc:145] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1748491647.276881  128140 buffer_comparator.cc:145] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.276884  128140 buffer_comparator.cc:145] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1748491647.276887  128140 buffer_comparator.cc:145] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1748491647.276890  128140 buffer_comparator.cc:145] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-05-29 04:07:27.276896: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.283188  128140 buffer_comparator.cc:145] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1748491647.283210  128140 buffer_comparator.cc:145] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1748491647.283214  128140 buffer_comparator.cc:145] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1748491647.283217  128140 buffer_comparator.cc:145] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1748491647.283220  128140 buffer_comparator.cc:145] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.283222  128140 buffer_comparator.cc:145] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1748491647.283225  128140 buffer_comparator.cc:145] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.283228  128140 buffer_comparator.cc:145] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1748491647.283231  128140 buffer_comparator.cc:145] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1748491647.283234  128140 buffer_comparator.cc:145] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-05-29 04:07:27.283241: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748491647.289656  128140 buffer_comparator.cc:145] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1748491647.289674  128140 buffer_comparator.cc:145] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1748491647.289677  128140 buffer_comparator.cc:145] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1748491647.289680  128140 buffer_comparator.cc:145] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1748491647.289683  128140 buffer_comparator.cc:145] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1748491647.289686  128140 buffer_comparator.cc:145] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1748491647.289689  128140 buffer_comparator.cc:145] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1748491647.289692  128140 buffer_comparator.cc:145] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>Test Loss: 59.203743	Test Acc: 10.9000%</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.5</span></span>
<span class="line"><span>Commit 760b2e5b739 (2025-04-14 06:53 UTC)</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,21)]))}const E=a(p,[["render",t]]);export{o as __pageData,E as default};
