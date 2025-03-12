import{_ as s,c as n,o as e,al as p}from"./chunks/framework.BCN3FD2k.js";const d=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),c={name:"tutorials/intermediate/6_GCN_Cora.md"};function i(t,a,r,l,f,o){return e(),n("div",null,a[0]||(a[0]=[p(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, Reactant, MLDatasets, Random, Statistics, Enzyme, GNNGraphs, ConcreteStructs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Printf, OneHotArrays, Optimisers</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; force </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(::MLDataDevices.CPUDevice) (generic function with 1 method)</span></span></code></pre></div><h2 id="Loading-Cora-Dataset" tabindex="-1">Loading Cora Dataset <a class="header-anchor" href="#Loading-Cora-Dataset" aria-label="Permalink to &quot;Loading Cora Dataset {#Loading-Cora-Dataset}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Cora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">graphs[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gnngraph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GNNGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_index; ndata </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data, edata </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_data, gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">num_nodes</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dense </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> adj</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, h_dim, out_dim; nb_layers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dropout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    layer_sizes </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, [h_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nb_layers])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn_layers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (in_dim, out_dim) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> zip</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)], layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; agg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> mean, logits </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))(y_pred, y[:, mask])</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st, (; y_pred)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        hidden_dim</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dropout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, nb_layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, use_bias</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        lr</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.001</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, weight_decay</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, patience</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, epochs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 200</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">seed!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    features, targets, adj, (train_idx, val_idx, test_idx) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(features, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), hidden_dim, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); nb_layers, dropout, use_bias)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, gcn) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    opt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(weight_decay) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lr) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AdamW</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lr, lambda </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> weight_decay)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gcn, ps, st, opt)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Total Trainable Parameters: %0.4f M</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">parameterlength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ps) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0e6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_loss_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gcn, ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st), (features, targets, adj, val_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, train_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, val_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Inf</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">epochs</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), loss_function, (features, targets, adj, train_idx), train_state;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            return_gradients </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                train_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, train_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, train_idx]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            val_loss_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                gcn, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                (features, targets, adj, val_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                val_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, val_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, val_idx]</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            gcn, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (features, targets, adj, test_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    test_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, test_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, test_idx]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Test Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Test Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> test_loss test_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-03-11 22:48:00.783300: I external/xla/xla/service/service.cc:152] XLA service 0xb7bc380 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-11 22:48:00.783451: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1741733280.784284 1210855 se_gpu_pjrt_client.cc:951] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1741733280.784386 1210855 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741733280.784581 1210855 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741733280.799764 1210855 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-12/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-12/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:340</span></span>
<span class="line"><span>2025-03-11 22:49:11.254444: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 16 bytes spill stores, 16 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:11.338578: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:11.534964: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 648 bytes spill stores, 652 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:11.600029: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 336 bytes spill stores, 336 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:11.603853: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 1176 bytes spill stores, 1148 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:11.621918: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 320 bytes spill stores, 320 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:12.006121: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 128 bytes spill stores, 128 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:12.480334: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22_0&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:12.728782: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:12.897952: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:12.948944: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 304 bytes spill stores, 304 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:13.481090: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 276 bytes spill stores, 276 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:13.515298: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_29&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:13.813328: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:49:13.909024: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741733354.280153 1210855 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1741733354.280938 1210855 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1741733354.280947 1210855 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1741733354.280954 1210855 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1741733354.280961 1210855 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1741733354.280968 1210855 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1741733354.280976 1210855 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1741733354.280982 1210855 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1741733354.280989 1210855 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.280995 1210855 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-03-11 22:49:14.281009: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.287529 1210855 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1741733354.287544 1210855 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1741733354.287547 1210855 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1741733354.287550 1210855 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1741733354.287553 1210855 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1741733354.287556 1210855 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1741733354.287559 1210855 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1741733354.287562 1210855 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1741733354.287565 1210855 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.287568 1210855 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-03-11 22:49:14.287573: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.292510 1210855 buffer_comparator.cc:156] Difference at 64: 0, expected 1106.21</span></span>
<span class="line"><span>E0000 00:00:1741733354.292528 1210855 buffer_comparator.cc:156] Difference at 65: 0, expected 1087.83</span></span>
<span class="line"><span>E0000 00:00:1741733354.292531 1210855 buffer_comparator.cc:156] Difference at 66: 0, expected 1090.54</span></span>
<span class="line"><span>E0000 00:00:1741733354.292534 1210855 buffer_comparator.cc:156] Difference at 67: 0, expected 1104.23</span></span>
<span class="line"><span>E0000 00:00:1741733354.292537 1210855 buffer_comparator.cc:156] Difference at 68: 0, expected 1104.3</span></span>
<span class="line"><span>E0000 00:00:1741733354.292540 1210855 buffer_comparator.cc:156] Difference at 69: 0, expected 1093.45</span></span>
<span class="line"><span>E0000 00:00:1741733354.292543 1210855 buffer_comparator.cc:156] Difference at 70: 0, expected 1091.52</span></span>
<span class="line"><span>E0000 00:00:1741733354.292546 1210855 buffer_comparator.cc:156] Difference at 71: 0, expected 1110.4</span></span>
<span class="line"><span>E0000 00:00:1741733354.292549 1210855 buffer_comparator.cc:156] Difference at 72: 0, expected 1106.92</span></span>
<span class="line"><span>E0000 00:00:1741733354.292552 1210855 buffer_comparator.cc:156] Difference at 73: 0, expected 1088.44</span></span>
<span class="line"><span>2025-03-11 22:49:14.292557: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.297307 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.297321 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.297324 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.297327 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.297330 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.297333 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.297336 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.297339 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.297342 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.297345 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.297351: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.302225 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.302239 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.302242 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.302245 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.302248 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.302251 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.302254 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.302257 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.302260 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.302263 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.302267: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.306831 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.306846 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.306850 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.306853 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.306856 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.306858 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.306861 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.306864 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.306867 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.306870 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.306875: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.311320 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.311336 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.311339 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.311342 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.311345 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.311348 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.311351 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.311353 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.311356 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.311359 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.311364: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.315929 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.315945 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.315950 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.315953 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.315956 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.315959 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.315961 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.315964 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.315967 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.315970 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.315975: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.320673 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.320687 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.320690 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.320693 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.320696 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.320699 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.320702 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.320705 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.320708 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.320711 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.320716: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.325287 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.325302 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.325305 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.325308 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.325311 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.325314 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.325317 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.325320 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.325323 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.325326 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.325331: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.329718 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.329731 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.329735 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.329738 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.329741 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.329744 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.329748 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.329751 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.329754 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.329757 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.329761: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.334125 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.334140 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.334143 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.334146 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.334149 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.334152 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.334155 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.334158 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.334161 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.334163 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.334168: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.338547 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.338565 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.338568 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.338571 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.338574 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.338577 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.338580 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.338583 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.338586 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.338589 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.338593: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.342943 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.342958 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.342961 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.342964 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.342967 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.342970 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.342973 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.342976 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.342978 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.342981 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.342987: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.347239 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.347254 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.347257 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.347260 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.347263 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.347266 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.347269 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.347272 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.347275 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.347278 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.347283: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.351834 1210855 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741733354.351849 1210855 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741733354.351853 1210855 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741733354.351856 1210855 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741733354.351858 1210855 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741733354.351861 1210855 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741733354.351864 1210855 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.351867 1210855 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741733354.351870 1210855 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741733354.351873 1210855 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.351878: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.356285 1210855 buffer_comparator.cc:156] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1741733354.356300 1210855 buffer_comparator.cc:156] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1741733354.356303 1210855 buffer_comparator.cc:156] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1741733354.356306 1210855 buffer_comparator.cc:156] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1741733354.356309 1210855 buffer_comparator.cc:156] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1741733354.356312 1210855 buffer_comparator.cc:156] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1741733354.356315 1210855 buffer_comparator.cc:156] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1741733354.356318 1210855 buffer_comparator.cc:156] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1741733354.356321 1210855 buffer_comparator.cc:156] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1741733354.356324 1210855 buffer_comparator.cc:156] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-03-11 22:49:14.356328: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.395439 1210855 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741733354.395485 1210855 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741733354.395492 1210855 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741733354.395496 1210855 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741733354.395500 1210855 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741733354.395505 1210855 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741733354.395509 1210855 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733354.395513 1210855 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741733354.395518 1210855 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741733354.395522 1210855 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.395532: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.398773 1210855 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741733354.398793 1210855 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741733354.398798 1210855 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741733354.398802 1210855 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741733354.398807 1210855 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741733354.398811 1210855 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741733354.398815 1210855 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733354.398819 1210855 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741733354.398824 1210855 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741733354.398828 1210855 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.398834: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.401888 1210855 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741733354.401903 1210855 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741733354.401906 1210855 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741733354.401909 1210855 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741733354.401912 1210855 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741733354.401915 1210855 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741733354.401918 1210855 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733354.401921 1210855 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741733354.401924 1210855 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741733354.401927 1210855 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.401932: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.404886 1210855 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741733354.404907 1210855 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741733354.404911 1210855 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741733354.404914 1210855 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741733354.404919 1210855 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741733354.404922 1210855 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741733354.404925 1210855 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733354.404928 1210855 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741733354.404931 1210855 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741733354.404934 1210855 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.404939: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.407912 1210855 buffer_comparator.cc:156] Difference at 0: 1139.71, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1741733354.407926 1210855 buffer_comparator.cc:156] Difference at 1: 1404.8, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1741733354.407929 1210855 buffer_comparator.cc:156] Difference at 2: 2132.23, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1741733354.407932 1210855 buffer_comparator.cc:156] Difference at 3: 1838.84, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1741733354.407935 1210855 buffer_comparator.cc:156] Difference at 4: 1307.39, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1741733354.407938 1210855 buffer_comparator.cc:156] Difference at 5: 2064.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1741733354.407941 1210855 buffer_comparator.cc:156] Difference at 6: 1480.82, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1741733354.407944 1210855 buffer_comparator.cc:156] Difference at 7: 1113.19, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1741733354.407947 1210855 buffer_comparator.cc:156] Difference at 8: 1358.65, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.407950 1210855 buffer_comparator.cc:156] Difference at 9: 2048.24, expected 1833.76</span></span>
<span class="line"><span>2025-03-11 22:49:14.407955: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.410949 1210855 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741733354.410964 1210855 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741733354.410967 1210855 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741733354.410970 1210855 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741733354.410973 1210855 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741733354.410976 1210855 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741733354.410979 1210855 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733354.410982 1210855 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741733354.410985 1210855 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741733354.410988 1210855 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-11 22:49:14.410993: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.414003 1210855 buffer_comparator.cc:156] Difference at 224: 1186.14, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1741733354.414017 1210855 buffer_comparator.cc:156] Difference at 225: 1033.68, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741733354.414020 1210855 buffer_comparator.cc:156] Difference at 226: 723.67, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1741733354.414023 1210855 buffer_comparator.cc:156] Difference at 227: 1156.29, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1741733354.414026 1210855 buffer_comparator.cc:156] Difference at 228: 843.86, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741733354.414029 1210855 buffer_comparator.cc:156] Difference at 229: 633.168, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741733354.414034 1210855 buffer_comparator.cc:156] Difference at 230: 810.302, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1741733354.414038 1210855 buffer_comparator.cc:156] Difference at 231: 1218.15, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1741733354.414041 1210855 buffer_comparator.cc:156] Difference at 232: 1064.04, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1741733354.414044 1210855 buffer_comparator.cc:156] Difference at 233: 741.156, expected 1803.13</span></span>
<span class="line"><span>2025-03-11 22:49:14.414054: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.416982 1210855 buffer_comparator.cc:156] Difference at 224: 1186.14, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1741733354.416997 1210855 buffer_comparator.cc:156] Difference at 225: 1033.68, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741733354.417000 1210855 buffer_comparator.cc:156] Difference at 226: 723.67, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1741733354.417003 1210855 buffer_comparator.cc:156] Difference at 227: 1156.29, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1741733354.417006 1210855 buffer_comparator.cc:156] Difference at 228: 843.86, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741733354.417009 1210855 buffer_comparator.cc:156] Difference at 229: 633.168, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741733354.417012 1210855 buffer_comparator.cc:156] Difference at 230: 810.302, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1741733354.417015 1210855 buffer_comparator.cc:156] Difference at 231: 1218.15, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1741733354.417018 1210855 buffer_comparator.cc:156] Difference at 232: 1064.04, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1741733354.417021 1210855 buffer_comparator.cc:156] Difference at 233: 741.156, expected 1803.13</span></span>
<span class="line"><span>2025-03-11 22:49:14.417026: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.419985 1210855 buffer_comparator.cc:156] Difference at 224: 1186.14, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1741733354.419999 1210855 buffer_comparator.cc:156] Difference at 225: 1033.68, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741733354.420002 1210855 buffer_comparator.cc:156] Difference at 226: 723.67, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1741733354.420005 1210855 buffer_comparator.cc:156] Difference at 227: 1156.29, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1741733354.420008 1210855 buffer_comparator.cc:156] Difference at 228: 843.86, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741733354.420011 1210855 buffer_comparator.cc:156] Difference at 229: 633.168, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741733354.420014 1210855 buffer_comparator.cc:156] Difference at 230: 810.302, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1741733354.420017 1210855 buffer_comparator.cc:156] Difference at 231: 1218.15, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1741733354.420020 1210855 buffer_comparator.cc:156] Difference at 232: 1064.04, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1741733354.420023 1210855 buffer_comparator.cc:156] Difference at 233: 741.156, expected 1803.13</span></span>
<span class="line"><span>2025-03-11 22:49:14.420028: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.423039 1210855 buffer_comparator.cc:156] Difference at 448: 1214.22, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741733354.423053 1210855 buffer_comparator.cc:156] Difference at 449: 1056.45, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733354.423056 1210855 buffer_comparator.cc:156] Difference at 450: 736.847, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741733354.423059 1210855 buffer_comparator.cc:156] Difference at 451: 1184.91, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741733354.423062 1210855 buffer_comparator.cc:156] Difference at 452: 859.942, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.423065 1210855 buffer_comparator.cc:156] Difference at 453: 620.77, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741733354.423068 1210855 buffer_comparator.cc:156] Difference at 454: 796.75, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741733354.423071 1210855 buffer_comparator.cc:156] Difference at 455: 1201.02, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741733354.423076 1210855 buffer_comparator.cc:156] Difference at 456: 1045.45, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741733354.423079 1210855 buffer_comparator.cc:156] Difference at 457: 732.834, expected 1821.28</span></span>
<span class="line"><span>2025-03-11 22:49:14.423084: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.426013 1210855 buffer_comparator.cc:156] Difference at 0: 1057.27, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1741733354.426026 1210855 buffer_comparator.cc:156] Difference at 1: 1319.15, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1741733354.426029 1210855 buffer_comparator.cc:156] Difference at 2: 2004.43, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1741733354.426032 1210855 buffer_comparator.cc:156] Difference at 3: 1745.74, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1741733354.426035 1210855 buffer_comparator.cc:156] Difference at 4: 1252.2, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1741733354.426039 1210855 buffer_comparator.cc:156] Difference at 7: 1175.57, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1741733354.426042 1210855 buffer_comparator.cc:156] Difference at 8: 1398.75, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1741733354.426045 1210855 buffer_comparator.cc:156] Difference at 9: 2125.62, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1741733354.426054 1210855 buffer_comparator.cc:156] Difference at 10: 1878.38, expected 1592.37</span></span>
<span class="line"><span>E0000 00:00:1741733354.426057 1210855 buffer_comparator.cc:156] Difference at 11: 1362.67, expected 1121.95</span></span>
<span class="line"><span>2025-03-11 22:49:14.426061: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.429070 1210855 buffer_comparator.cc:156] Difference at 448: 1221.14, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741733354.429086 1210855 buffer_comparator.cc:156] Difference at 449: 1061.5, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733354.429089 1210855 buffer_comparator.cc:156] Difference at 450: 743.315, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741733354.429092 1210855 buffer_comparator.cc:156] Difference at 451: 1192.79, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741733354.429095 1210855 buffer_comparator.cc:156] Difference at 452: 864.899, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.429098 1210855 buffer_comparator.cc:156] Difference at 453: 626.203, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741733354.429101 1210855 buffer_comparator.cc:156] Difference at 454: 803.97, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741733354.429104 1210855 buffer_comparator.cc:156] Difference at 455: 1208.29, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741733354.429108 1210855 buffer_comparator.cc:156] Difference at 456: 1052.01, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741733354.429112 1210855 buffer_comparator.cc:156] Difference at 457: 737.437, expected 1821.28</span></span>
<span class="line"><span>2025-03-11 22:49:14.429117: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.432110 1210855 buffer_comparator.cc:156] Difference at 448: 1221.14, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741733354.432123 1210855 buffer_comparator.cc:156] Difference at 449: 1061.5, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733354.432127 1210855 buffer_comparator.cc:156] Difference at 450: 743.315, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741733354.432130 1210855 buffer_comparator.cc:156] Difference at 451: 1192.79, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741733354.432133 1210855 buffer_comparator.cc:156] Difference at 452: 864.899, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.432136 1210855 buffer_comparator.cc:156] Difference at 453: 626.203, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741733354.432139 1210855 buffer_comparator.cc:156] Difference at 454: 803.97, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741733354.432142 1210855 buffer_comparator.cc:156] Difference at 455: 1208.29, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741733354.432145 1210855 buffer_comparator.cc:156] Difference at 456: 1052.01, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741733354.432148 1210855 buffer_comparator.cc:156] Difference at 457: 737.437, expected 1821.28</span></span>
<span class="line"><span>2025-03-11 22:49:14.432154: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.435141 1210855 buffer_comparator.cc:156] Difference at 448: 1221.14, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741733354.435155 1210855 buffer_comparator.cc:156] Difference at 449: 1061.5, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733354.435158 1210855 buffer_comparator.cc:156] Difference at 450: 743.315, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741733354.435161 1210855 buffer_comparator.cc:156] Difference at 451: 1192.79, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741733354.435164 1210855 buffer_comparator.cc:156] Difference at 452: 864.899, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.435167 1210855 buffer_comparator.cc:156] Difference at 453: 626.203, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741733354.435170 1210855 buffer_comparator.cc:156] Difference at 454: 803.97, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741733354.435174 1210855 buffer_comparator.cc:156] Difference at 455: 1208.29, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741733354.435177 1210855 buffer_comparator.cc:156] Difference at 456: 1052.01, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741733354.435180 1210855 buffer_comparator.cc:156] Difference at 457: 737.437, expected 1821.28</span></span>
<span class="line"><span>2025-03-11 22:49:14.435184: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.438142 1210855 buffer_comparator.cc:156] Difference at 448: 1221.14, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741733354.438158 1210855 buffer_comparator.cc:156] Difference at 449: 1061.5, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733354.438162 1210855 buffer_comparator.cc:156] Difference at 450: 743.315, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741733354.438165 1210855 buffer_comparator.cc:156] Difference at 451: 1192.79, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741733354.438168 1210855 buffer_comparator.cc:156] Difference at 452: 864.899, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.438171 1210855 buffer_comparator.cc:156] Difference at 453: 626.203, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741733354.438174 1210855 buffer_comparator.cc:156] Difference at 454: 803.97, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741733354.438177 1210855 buffer_comparator.cc:156] Difference at 455: 1208.29, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741733354.438180 1210855 buffer_comparator.cc:156] Difference at 456: 1052.01, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741733354.438183 1210855 buffer_comparator.cc:156] Difference at 457: 737.437, expected 1821.28</span></span>
<span class="line"><span>2025-03-11 22:49:14.438188: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.441291 1210855 buffer_comparator.cc:156] Difference at 896: 1204.66, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1741733354.441305 1210855 buffer_comparator.cc:156] Difference at 897: 1053.28, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741733354.441308 1210855 buffer_comparator.cc:156] Difference at 898: 740.998, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1741733354.441311 1210855 buffer_comparator.cc:156] Difference at 899: 1185.71, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1741733354.441314 1210855 buffer_comparator.cc:156] Difference at 900: 850.478, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741733354.441317 1210855 buffer_comparator.cc:156] Difference at 901: 634.712, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.441320 1210855 buffer_comparator.cc:156] Difference at 902: 799.593, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741733354.441324 1210855 buffer_comparator.cc:156] Difference at 903: 1208.15, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1741733354.441327 1210855 buffer_comparator.cc:156] Difference at 904: 1055.09, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1741733354.441330 1210855 buffer_comparator.cc:156] Difference at 905: 746.267, expected 1817.41</span></span>
<span class="line"><span>2025-03-11 22:49:14.441334: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.444432 1210855 buffer_comparator.cc:156] Difference at 896: 1204.66, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1741733354.444446 1210855 buffer_comparator.cc:156] Difference at 897: 1053.28, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741733354.444449 1210855 buffer_comparator.cc:156] Difference at 898: 740.998, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1741733354.444452 1210855 buffer_comparator.cc:156] Difference at 899: 1185.71, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1741733354.444455 1210855 buffer_comparator.cc:156] Difference at 900: 850.478, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741733354.444458 1210855 buffer_comparator.cc:156] Difference at 901: 634.712, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.444461 1210855 buffer_comparator.cc:156] Difference at 902: 799.593, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741733354.444464 1210855 buffer_comparator.cc:156] Difference at 903: 1208.15, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1741733354.444467 1210855 buffer_comparator.cc:156] Difference at 904: 1055.09, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1741733354.444470 1210855 buffer_comparator.cc:156] Difference at 905: 746.267, expected 1817.41</span></span>
<span class="line"><span>2025-03-11 22:49:14.444475: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.447523 1210855 buffer_comparator.cc:156] Difference at 896: 1204.66, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1741733354.447536 1210855 buffer_comparator.cc:156] Difference at 897: 1053.28, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741733354.447540 1210855 buffer_comparator.cc:156] Difference at 898: 740.998, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1741733354.447543 1210855 buffer_comparator.cc:156] Difference at 899: 1185.71, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1741733354.447546 1210855 buffer_comparator.cc:156] Difference at 900: 850.478, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741733354.447549 1210855 buffer_comparator.cc:156] Difference at 901: 634.712, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.447552 1210855 buffer_comparator.cc:156] Difference at 902: 799.593, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741733354.447555 1210855 buffer_comparator.cc:156] Difference at 903: 1208.15, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1741733354.447558 1210855 buffer_comparator.cc:156] Difference at 904: 1055.09, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1741733354.447561 1210855 buffer_comparator.cc:156] Difference at 905: 746.267, expected 1817.41</span></span>
<span class="line"><span>2025-03-11 22:49:14.447566: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.450549 1210855 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1741733354.450565 1210855 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741733354.450569 1210855 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1741733354.450572 1210855 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1741733354.450575 1210855 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741733354.450578 1210855 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.450581 1210855 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741733354.450584 1210855 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1741733354.450587 1210855 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1741733354.450590 1210855 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-03-11 22:49:14.450595: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.453812 1210855 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1741733354.453827 1210855 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741733354.453831 1210855 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.453834 1210855 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741733354.453837 1210855 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1741733354.453840 1210855 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1741733354.453843 1210855 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1741733354.453847 1210855 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1741733354.453850 1210855 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1741733354.453853 1210855 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-03-11 22:49:14.453857: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.457088 1210855 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1741733354.457101 1210855 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741733354.457105 1210855 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.457108 1210855 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741733354.457111 1210855 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1741733354.457114 1210855 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1741733354.457117 1210855 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1741733354.457120 1210855 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1741733354.457123 1210855 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1741733354.457126 1210855 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-03-11 22:49:14.457131: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.460309 1210855 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1741733354.460331 1210855 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741733354.460335 1210855 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741733354.460338 1210855 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741733354.460341 1210855 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1741733354.460344 1210855 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1741733354.460347 1210855 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1741733354.460350 1210855 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1741733354.460353 1210855 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1741733354.460356 1210855 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-03-11 22:49:14.460361: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.464809 1210855 buffer_comparator.cc:156] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741733354.464824 1210855 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741733354.464827 1210855 buffer_comparator.cc:156] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741733354.464831 1210855 buffer_comparator.cc:156] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741733354.464834 1210855 buffer_comparator.cc:156] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741733354.464837 1210855 buffer_comparator.cc:156] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741733354.464840 1210855 buffer_comparator.cc:156] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741733354.464843 1210855 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741733354.464845 1210855 buffer_comparator.cc:156] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741733354.464848 1210855 buffer_comparator.cc:156] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-03-11 22:49:14.464853: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.467227 1210855 buffer_comparator.cc:156] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741733354.467241 1210855 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741733354.467244 1210855 buffer_comparator.cc:156] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741733354.467247 1210855 buffer_comparator.cc:156] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741733354.467250 1210855 buffer_comparator.cc:156] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741733354.467253 1210855 buffer_comparator.cc:156] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741733354.467255 1210855 buffer_comparator.cc:156] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741733354.467258 1210855 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741733354.467261 1210855 buffer_comparator.cc:156] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741733354.467264 1210855 buffer_comparator.cc:156] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-03-11 22:49:14.467268: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.469607 1210855 buffer_comparator.cc:156] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741733354.469621 1210855 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741733354.469624 1210855 buffer_comparator.cc:156] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741733354.469627 1210855 buffer_comparator.cc:156] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741733354.469630 1210855 buffer_comparator.cc:156] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741733354.469633 1210855 buffer_comparator.cc:156] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741733354.469636 1210855 buffer_comparator.cc:156] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741733354.469638 1210855 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741733354.469641 1210855 buffer_comparator.cc:156] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741733354.469644 1210855 buffer_comparator.cc:156] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-03-11 22:49:14.469648: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.472062 1210855 buffer_comparator.cc:156] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741733354.472077 1210855 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741733354.472081 1210855 buffer_comparator.cc:156] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741733354.472083 1210855 buffer_comparator.cc:156] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741733354.472086 1210855 buffer_comparator.cc:156] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741733354.472089 1210855 buffer_comparator.cc:156] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741733354.472093 1210855 buffer_comparator.cc:156] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741733354.472096 1210855 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741733354.472099 1210855 buffer_comparator.cc:156] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741733354.472102 1210855 buffer_comparator.cc:156] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-03-11 22:49:14.472106: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.474480 1210855 buffer_comparator.cc:156] Difference at 32: -nan, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741733354.474494 1210855 buffer_comparator.cc:156] Difference at 33: -nan, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741733354.474497 1210855 buffer_comparator.cc:156] Difference at 34: -nan, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741733354.474500 1210855 buffer_comparator.cc:156] Difference at 35: -nan, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741733354.474503 1210855 buffer_comparator.cc:156] Difference at 36: -nan, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741733354.474505 1210855 buffer_comparator.cc:156] Difference at 37: -nan, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741733354.474508 1210855 buffer_comparator.cc:156] Difference at 38: -nan, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741733354.474511 1210855 buffer_comparator.cc:156] Difference at 39: -nan, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741733354.474514 1210855 buffer_comparator.cc:156] Difference at 40: -nan, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741733354.474516 1210855 buffer_comparator.cc:156] Difference at 41: -nan, expected 20.3484</span></span>
<span class="line"><span>2025-03-11 22:49:14.474521: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.476877 1210855 buffer_comparator.cc:156] Difference at 32: -nan, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741733354.476895 1210855 buffer_comparator.cc:156] Difference at 33: -nan, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741733354.476898 1210855 buffer_comparator.cc:156] Difference at 34: -nan, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741733354.476900 1210855 buffer_comparator.cc:156] Difference at 35: -nan, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741733354.476903 1210855 buffer_comparator.cc:156] Difference at 36: -nan, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741733354.476906 1210855 buffer_comparator.cc:156] Difference at 37: -nan, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741733354.476909 1210855 buffer_comparator.cc:156] Difference at 38: -nan, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741733354.476911 1210855 buffer_comparator.cc:156] Difference at 39: -nan, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741733354.476914 1210855 buffer_comparator.cc:156] Difference at 40: -nan, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741733354.476917 1210855 buffer_comparator.cc:156] Difference at 41: -nan, expected 20.3484</span></span>
<span class="line"><span>2025-03-11 22:49:14.476921: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.479309 1210855 buffer_comparator.cc:156] Difference at 32: -nan, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741733354.479324 1210855 buffer_comparator.cc:156] Difference at 33: -nan, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741733354.479327 1210855 buffer_comparator.cc:156] Difference at 34: -nan, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741733354.479330 1210855 buffer_comparator.cc:156] Difference at 35: -nan, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741733354.479333 1210855 buffer_comparator.cc:156] Difference at 36: -nan, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741733354.479336 1210855 buffer_comparator.cc:156] Difference at 37: -nan, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741733354.479339 1210855 buffer_comparator.cc:156] Difference at 38: -nan, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741733354.479341 1210855 buffer_comparator.cc:156] Difference at 39: -nan, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741733354.479344 1210855 buffer_comparator.cc:156] Difference at 40: -nan, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741733354.479348 1210855 buffer_comparator.cc:156] Difference at 41: -nan, expected 20.3484</span></span>
<span class="line"><span>2025-03-11 22:49:14.479353: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.481720 1210855 buffer_comparator.cc:156] Difference at 32: -nan, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741733354.481736 1210855 buffer_comparator.cc:156] Difference at 33: -nan, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741733354.481739 1210855 buffer_comparator.cc:156] Difference at 34: -nan, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741733354.481742 1210855 buffer_comparator.cc:156] Difference at 35: -nan, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741733354.481745 1210855 buffer_comparator.cc:156] Difference at 36: -nan, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741733354.481748 1210855 buffer_comparator.cc:156] Difference at 37: -nan, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741733354.481750 1210855 buffer_comparator.cc:156] Difference at 38: -nan, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741733354.481753 1210855 buffer_comparator.cc:156] Difference at 39: -nan, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741733354.481756 1210855 buffer_comparator.cc:156] Difference at 40: -nan, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741733354.481759 1210855 buffer_comparator.cc:156] Difference at 41: -nan, expected 20.3484</span></span>
<span class="line"><span>2025-03-11 22:49:14.481763: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.484130 1210855 buffer_comparator.cc:156] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741733354.484144 1210855 buffer_comparator.cc:156] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741733354.484147 1210855 buffer_comparator.cc:156] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741733354.484150 1210855 buffer_comparator.cc:156] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741733354.484152 1210855 buffer_comparator.cc:156] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741733354.484155 1210855 buffer_comparator.cc:156] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741733354.484158 1210855 buffer_comparator.cc:156] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741733354.484161 1210855 buffer_comparator.cc:156] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741733354.484164 1210855 buffer_comparator.cc:156] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741733354.484166 1210855 buffer_comparator.cc:156] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-03-11 22:49:14.484171: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733354.486557 1210855 buffer_comparator.cc:156] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741733354.486572 1210855 buffer_comparator.cc:156] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741733354.486575 1210855 buffer_comparator.cc:156] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741733354.486578 1210855 buffer_comparator.cc:156] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741733354.486580 1210855 buffer_comparator.cc:156] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741733354.486583 1210855 buffer_comparator.cc:156] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741733354.486586 1210855 buffer_comparator.cc:156] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741733354.486589 1210855 buffer_comparator.cc:156] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741733354.486591 1210855 buffer_comparator.cc:156] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741733354.486594 1210855 buffer_comparator.cc:156] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>E0000 00:00:1741733354.488955 1210855 buffer_comparator.cc:156] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>Epoch   1	Train Loss: 15.773692	Train Acc: 19.2857%	Val Loss: 7.023974	Val Acc: 20.8000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 8.867982	Train Acc: 24.2857%	Val Loss: 2.967125	Val Acc: 29.4000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 3.609741	Train Acc: 42.8571%	Val Loss: 2.001948	Val Acc: 40.0000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 1.742898	Train Acc: 54.2857%	Val Loss: 2.041071	Val Acc: 44.0000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 1.592426	Train Acc: 58.5714%	Val Loss: 1.836263	Val Acc: 49.4000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.353780	Train Acc: 65.7143%	Val Loss: 1.630883	Val Acc: 54.2000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.132901	Train Acc: 70.0000%	Val Loss: 1.536177	Val Acc: 60.2000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 0.975761	Train Acc: 72.8571%	Val Loss: 1.527029	Val Acc: 61.6000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 0.911684	Train Acc: 75.7143%	Val Loss: 1.535078	Val Acc: 62.0000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 0.937093	Train Acc: 77.1429%	Val Loss: 1.510086	Val Acc: 63.6000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 0.833613	Train Acc: 78.5714%	Val Loss: 1.489088	Val Acc: 66.0000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 0.753915	Train Acc: 82.8571%	Val Loss: 1.485898	Val Acc: 66.6000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.710028	Train Acc: 85.0000%	Val Loss: 1.491980	Val Acc: 68.0000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 0.664639	Train Acc: 84.2857%	Val Loss: 1.507280	Val Acc: 66.8000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 0.613682	Train Acc: 85.0000%	Val Loss: 1.527489	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.562765	Train Acc: 86.4286%	Val Loss: 1.545437	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.532903	Train Acc: 86.4286%	Val Loss: 1.553837	Val Acc: 67.8000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.514571	Train Acc: 87.1429%	Val Loss: 1.550445	Val Acc: 67.4000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.492412	Train Acc: 87.8571%	Val Loss: 1.544771	Val Acc: 67.8000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.468656	Train Acc: 88.5714%	Val Loss: 1.544334	Val Acc: 68.2000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.446536	Train Acc: 88.5714%	Val Loss: 1.551339	Val Acc: 68.6000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.426508	Train Acc: 87.8571%	Val Loss: 1.567150	Val Acc: 68.6000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.408416	Train Acc: 88.5714%	Val Loss: 1.589547	Val Acc: 67.8000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.391937	Train Acc: 88.5714%	Val Loss: 1.616495	Val Acc: 67.4000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.377947	Train Acc: 88.5714%	Val Loss: 1.647271	Val Acc: 67.6000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.365843	Train Acc: 89.2857%	Val Loss: 1.679508	Val Acc: 68.0000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.355406	Train Acc: 91.4286%	Val Loss: 1.711879	Val Acc: 67.6000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.345920	Train Acc: 91.4286%	Val Loss: 1.740336	Val Acc: 67.6000%</span></span>
<span class="line"><span>Epoch  29	Train Loss: 0.336177	Train Acc: 92.1429%	Val Loss: 1.763298	Val Acc: 67.8000%</span></span>
<span class="line"><span>Epoch  30	Train Loss: 0.325666	Train Acc: 92.1429%	Val Loss: 1.780414	Val Acc: 67.8000%</span></span>
<span class="line"><span>Epoch  31	Train Loss: 0.314244	Train Acc: 92.8571%	Val Loss: 1.791988	Val Acc: 67.8000%</span></span>
<span class="line"><span>Epoch  32	Train Loss: 0.302450	Train Acc: 93.5714%	Val Loss: 1.799915	Val Acc: 67.6000%</span></span>
<span class="line"><span>Early Stopping at Epoch 32</span></span>
<span class="line"><span>2025-03-11 22:49:59.844301: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:50:00.185120: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:50:00.324121: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741733400.331691 1210855 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741733400.331762 1210855 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741733400.331770 1210855 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741733400.331777 1210855 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.331785 1210855 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741733400.331792 1210855 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741733400.331799 1210855 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733400.331806 1210855 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741733400.331813 1210855 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741733400.331819 1210855 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-11 22:50:00.331835: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.335455 1210855 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741733400.335483 1210855 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741733400.335491 1210855 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741733400.335498 1210855 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.335505 1210855 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741733400.335512 1210855 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741733400.335519 1210855 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733400.335526 1210855 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741733400.335533 1210855 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741733400.335540 1210855 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-11 22:50:00.335551: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.339160 1210855 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741733400.339174 1210855 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741733400.339177 1210855 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741733400.339180 1210855 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.339183 1210855 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741733400.339186 1210855 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741733400.339190 1210855 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733400.339193 1210855 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741733400.339197 1210855 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741733400.339200 1210855 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-11 22:50:00.339205: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.342573 1210855 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741733400.342587 1210855 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741733400.342590 1210855 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741733400.342593 1210855 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.342596 1210855 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741733400.342599 1210855 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741733400.342603 1210855 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733400.342606 1210855 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741733400.342609 1210855 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741733400.342612 1210855 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-11 22:50:00.342616: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.346008 1210855 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741733400.346023 1210855 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741733400.346027 1210855 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741733400.346030 1210855 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.346033 1210855 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741733400.346036 1210855 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741733400.346039 1210855 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733400.346042 1210855 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741733400.346045 1210855 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741733400.346054 1210855 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-11 22:50:00.346059: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.349473 1210855 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741733400.349488 1210855 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741733400.349491 1210855 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741733400.349494 1210855 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.349497 1210855 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741733400.349500 1210855 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741733400.349504 1210855 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741733400.349507 1210855 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741733400.349510 1210855 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741733400.349513 1210855 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-11 22:50:00.349519: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.352846 1210855 buffer_comparator.cc:156] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1741733400.352863 1210855 buffer_comparator.cc:156] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741733400.352866 1210855 buffer_comparator.cc:156] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741733400.352869 1210855 buffer_comparator.cc:156] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741733400.352872 1210855 buffer_comparator.cc:156] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741733400.352875 1210855 buffer_comparator.cc:156] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741733400.352878 1210855 buffer_comparator.cc:156] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741733400.352881 1210855 buffer_comparator.cc:156] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1741733400.352884 1210855 buffer_comparator.cc:156] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741733400.352887 1210855 buffer_comparator.cc:156] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-03-11 22:50:00.352892: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.356185 1210855 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741733400.356201 1210855 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741733400.356204 1210855 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741733400.356207 1210855 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741733400.356210 1210855 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741733400.356213 1210855 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741733400.356216 1210855 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741733400.356219 1210855 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1741733400.356222 1210855 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1741733400.356225 1210855 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-11 22:50:00.356230: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.359555 1210855 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741733400.359573 1210855 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741733400.359576 1210855 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741733400.359579 1210855 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741733400.359582 1210855 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741733400.359585 1210855 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741733400.359589 1210855 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741733400.359592 1210855 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1741733400.359595 1210855 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1741733400.359598 1210855 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-11 22:50:00.359602: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.362951 1210855 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733400.362966 1210855 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741733400.362970 1210855 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741733400.362973 1210855 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741733400.362976 1210855 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741733400.362979 1210855 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741733400.362982 1210855 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.362985 1210855 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.362988 1210855 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741733400.362991 1210855 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-11 22:50:00.362996: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.366268 1210855 buffer_comparator.cc:156] Difference at 7: 1058.92, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1741733400.366286 1210855 buffer_comparator.cc:156] Difference at 11: 1263.92, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1741733400.366290 1210855 buffer_comparator.cc:156] Difference at 179: 1223.75, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1741733400.366294 1210855 buffer_comparator.cc:156] Difference at 266: 1047.35, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1741733400.366297 1210855 buffer_comparator.cc:156] Difference at 270: 1246.8, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1741733400.366301 1210855 buffer_comparator.cc:156] Difference at 417: 1222.47, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1741733400.366304 1210855 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733400.366307 1210855 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741733400.366310 1210855 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741733400.366313 1210855 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>2025-03-11 22:50:00.366318: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.369629 1210855 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733400.369642 1210855 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741733400.369646 1210855 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741733400.369649 1210855 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741733400.369652 1210855 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741733400.369655 1210855 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741733400.369658 1210855 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.369661 1210855 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.369664 1210855 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741733400.369667 1210855 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-11 22:50:00.369672: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.373013 1210855 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733400.373028 1210855 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741733400.373031 1210855 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741733400.373035 1210855 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741733400.373038 1210855 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741733400.373041 1210855 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741733400.373044 1210855 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.373047 1210855 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.373050 1210855 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741733400.373053 1210855 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-11 22:50:00.373058: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.376357 1210855 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733400.376371 1210855 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741733400.376375 1210855 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741733400.376378 1210855 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741733400.376381 1210855 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741733400.376384 1210855 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741733400.376387 1210855 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.376390 1210855 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.376393 1210855 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741733400.376396 1210855 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-11 22:50:00.376401: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.379710 1210855 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741733400.379724 1210855 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741733400.379727 1210855 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741733400.379731 1210855 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741733400.379734 1210855 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741733400.379737 1210855 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741733400.379740 1210855 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.379743 1210855 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741733400.379746 1210855 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741733400.379749 1210855 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-11 22:50:00.379753: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.383176 1210855 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741733400.383190 1210855 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741733400.383193 1210855 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741733400.383197 1210855 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741733400.383201 1210855 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741733400.383204 1210855 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741733400.383207 1210855 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741733400.383210 1210855 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741733400.383213 1210855 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741733400.383216 1210855 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-11 22:50:00.383221: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.386630 1210855 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741733400.386646 1210855 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741733400.386650 1210855 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741733400.386653 1210855 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741733400.386656 1210855 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741733400.386659 1210855 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741733400.386662 1210855 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741733400.386665 1210855 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741733400.386668 1210855 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741733400.386671 1210855 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-11 22:50:00.386676: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.390034 1210855 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741733400.390063 1210855 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741733400.390066 1210855 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741733400.390069 1210855 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741733400.390072 1210855 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741733400.390076 1210855 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741733400.390079 1210855 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741733400.390082 1210855 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741733400.390085 1210855 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741733400.390088 1210855 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-11 22:50:00.390092: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.393422 1210855 buffer_comparator.cc:156] Difference at 896: 485.098, expected 958.133</span></span>
<span class="line"><span>E0000 00:00:1741733400.393439 1210855 buffer_comparator.cc:156] Difference at 897: 732.587, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741733400.393442 1210855 buffer_comparator.cc:156] Difference at 898: 635.29, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741733400.393445 1210855 buffer_comparator.cc:156] Difference at 899: 446.948, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741733400.393448 1210855 buffer_comparator.cc:156] Difference at 900: 712.745, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741733400.393452 1210855 buffer_comparator.cc:156] Difference at 901: 516.07, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741733400.393456 1210855 buffer_comparator.cc:156] Difference at 902: 373.095, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741733400.393459 1210855 buffer_comparator.cc:156] Difference at 903: 483.905, expected 941.483</span></span>
<span class="line"><span>E0000 00:00:1741733400.393462 1210855 buffer_comparator.cc:156] Difference at 904: 721.412, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741733400.393465 1210855 buffer_comparator.cc:156] Difference at 905: 633.571, expected 1817.42</span></span>
<span class="line"><span>2025-03-11 22:50:00.393470: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.397191 1210855 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741733400.397205 1210855 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741733400.397208 1210855 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741733400.397211 1210855 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.397214 1210855 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741733400.397218 1210855 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741733400.397221 1210855 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741733400.397224 1210855 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741733400.397227 1210855 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.397230 1210855 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-11 22:50:00.397235: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.400808 1210855 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741733400.400824 1210855 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741733400.400827 1210855 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741733400.400830 1210855 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.400833 1210855 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741733400.400836 1210855 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741733400.400839 1210855 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741733400.400842 1210855 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741733400.400846 1210855 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.400849 1210855 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-11 22:50:00.400853: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733400.404372 1210855 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741733400.404389 1210855 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741733400.404392 1210855 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741733400.404395 1210855 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.404399 1210855 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741733400.404402 1210855 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741733400.404405 1210855 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741733400.404409 1210855 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741733400.404412 1210855 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741733400.404415 1210855 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-11 22:50:00.404420: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-03-11 22:50:01.860668: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:50:01.944930: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-11 22:50:02.332113: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741733402.339209 1210855 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741733402.339265 1210855 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741733402.339269 1210855 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741733402.339272 1210855 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741733402.339275 1210855 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741733402.339278 1210855 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741733402.339281 1210855 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741733402.339284 1210855 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741733402.339287 1210855 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741733402.339290 1210855 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-11 22:50:02.339300: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.342631 1210855 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741733402.342647 1210855 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741733402.342650 1210855 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741733402.342653 1210855 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741733402.342656 1210855 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741733402.342659 1210855 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741733402.342662 1210855 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741733402.342665 1210855 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741733402.342668 1210855 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741733402.342671 1210855 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-11 22:50:02.342676: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.346024 1210855 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741733402.346038 1210855 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741733402.346041 1210855 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741733402.346045 1210855 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741733402.346056 1210855 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741733402.346059 1210855 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741733402.346062 1210855 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741733402.346065 1210855 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741733402.346068 1210855 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741733402.346071 1210855 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-11 22:50:02.346075: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.349419 1210855 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741733402.349433 1210855 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741733402.349436 1210855 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741733402.349439 1210855 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741733402.349442 1210855 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741733402.349445 1210855 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741733402.349448 1210855 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741733402.349451 1210855 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741733402.349454 1210855 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741733402.349457 1210855 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-11 22:50:02.349461: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.352800 1210855 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741733402.352817 1210855 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741733402.352820 1210855 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741733402.352823 1210855 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741733402.352826 1210855 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741733402.352829 1210855 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741733402.352832 1210855 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741733402.352835 1210855 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741733402.352838 1210855 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741733402.352841 1210855 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-11 22:50:02.352846: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.356257 1210855 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741733402.356272 1210855 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741733402.356275 1210855 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741733402.356278 1210855 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741733402.356281 1210855 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741733402.356284 1210855 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741733402.356287 1210855 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741733402.356290 1210855 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741733402.356295 1210855 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741733402.356298 1210855 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-11 22:50:02.356302: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.359594 1210855 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741733402.359609 1210855 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741733402.359613 1210855 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741733402.359616 1210855 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741733402.359619 1210855 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741733402.359622 1210855 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741733402.359624 1210855 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741733402.359627 1210855 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741733402.359630 1210855 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741733402.359633 1210855 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-11 22:50:02.359638: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.362923 1210855 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741733402.362937 1210855 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741733402.362940 1210855 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741733402.362943 1210855 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741733402.362946 1210855 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741733402.362949 1210855 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741733402.362952 1210855 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741733402.362955 1210855 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741733402.362958 1210855 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741733402.362961 1210855 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-11 22:50:02.362966: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.366276 1210855 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741733402.366293 1210855 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741733402.366297 1210855 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741733402.366300 1210855 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741733402.366303 1210855 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741733402.366305 1210855 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741733402.366308 1210855 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741733402.366311 1210855 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741733402.366314 1210855 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741733402.366317 1210855 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-11 22:50:02.366322: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.369643 1210855 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741733402.369657 1210855 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741733402.369660 1210855 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741733402.369663 1210855 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741733402.369666 1210855 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741733402.369669 1210855 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741733402.369672 1210855 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741733402.369675 1210855 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741733402.369678 1210855 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741733402.369681 1210855 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-11 22:50:02.369685: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.372949 1210855 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741733402.372966 1210855 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741733402.372969 1210855 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741733402.372972 1210855 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741733402.372975 1210855 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741733402.372978 1210855 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741733402.372981 1210855 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741733402.372984 1210855 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741733402.372987 1210855 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741733402.372989 1210855 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-11 22:50:02.372994: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.376283 1210855 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741733402.376297 1210855 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741733402.376300 1210855 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741733402.376303 1210855 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741733402.376306 1210855 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741733402.376309 1210855 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741733402.376312 1210855 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741733402.376315 1210855 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741733402.376318 1210855 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741733402.376321 1210855 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-11 22:50:02.376325: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.379639 1210855 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741733402.379653 1210855 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741733402.379656 1210855 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741733402.379660 1210855 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741733402.379663 1210855 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741733402.379666 1210855 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741733402.379669 1210855 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741733402.379672 1210855 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741733402.379675 1210855 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741733402.379678 1210855 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-11 22:50:02.379682: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.382978 1210855 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741733402.382993 1210855 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741733402.382996 1210855 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741733402.382999 1210855 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741733402.383002 1210855 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741733402.383005 1210855 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741733402.383008 1210855 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741733402.383011 1210855 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741733402.383014 1210855 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741733402.383017 1210855 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-11 22:50:02.383022: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.386305 1210855 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741733402.386319 1210855 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741733402.386322 1210855 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741733402.386325 1210855 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741733402.386328 1210855 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741733402.386331 1210855 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741733402.386334 1210855 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741733402.386337 1210855 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741733402.386340 1210855 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741733402.386343 1210855 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-11 22:50:02.386347: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.389755 1210855 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741733402.389769 1210855 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741733402.389772 1210855 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741733402.389775 1210855 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741733402.389778 1210855 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741733402.389781 1210855 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741733402.389784 1210855 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741733402.389788 1210855 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741733402.389791 1210855 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741733402.389794 1210855 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-11 22:50:02.389798: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.393190 1210855 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741733402.393204 1210855 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741733402.393207 1210855 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741733402.393210 1210855 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741733402.393213 1210855 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741733402.393216 1210855 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741733402.393219 1210855 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741733402.393221 1210855 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741733402.393224 1210855 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741733402.393227 1210855 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-11 22:50:02.393232: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.396571 1210855 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741733402.396586 1210855 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741733402.396590 1210855 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741733402.396593 1210855 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741733402.396596 1210855 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741733402.396598 1210855 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741733402.396601 1210855 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741733402.396604 1210855 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741733402.396607 1210855 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741733402.396610 1210855 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-11 22:50:02.396615: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.399928 1210855 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741733402.399942 1210855 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741733402.399946 1210855 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741733402.399949 1210855 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741733402.399952 1210855 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741733402.399954 1210855 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741733402.399957 1210855 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741733402.399960 1210855 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741733402.399963 1210855 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741733402.399966 1210855 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-11 22:50:02.399971: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.403500 1210855 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741733402.403514 1210855 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741733402.403517 1210855 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741733402.403520 1210855 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741733402.403523 1210855 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741733402.403526 1210855 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741733402.403529 1210855 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741733402.403532 1210855 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741733402.403535 1210855 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741733402.403538 1210855 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-11 22:50:02.403542: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.407078 1210855 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741733402.407091 1210855 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741733402.407094 1210855 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741733402.407097 1210855 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741733402.407100 1210855 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741733402.407103 1210855 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741733402.407106 1210855 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741733402.407109 1210855 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741733402.407112 1210855 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741733402.407115 1210855 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-11 22:50:02.407120: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741733402.410594 1210855 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741733402.410610 1210855 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741733402.410613 1210855 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741733402.410616 1210855 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741733402.410619 1210855 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741733402.410622 1210855 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741733402.410625 1210855 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741733402.410628 1210855 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741733402.410631 1210855 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741733402.410634 1210855 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-11 22:50:02.410638: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Test Loss: 1.645231	Test Acc: 70.8000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.4</span></span>
<span class="line"><span>Commit 8561cc3d68d (2025-03-10 11:36 UTC)</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,21)]))}const E=s(c,[["render",i]]);export{d as __pageData,E as default};
