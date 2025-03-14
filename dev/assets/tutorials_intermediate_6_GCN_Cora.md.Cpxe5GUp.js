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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-03-14 20:56:12.548997: I external/xla/xla/service/service.cc:152] XLA service 0x41edf20 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-14 20:56:12.549424: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1741985772.550734 2659715 se_gpu_pjrt_client.cc:951] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1741985772.550855 2659715 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741985772.551402 2659715 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741985772.568663 2659715 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-12/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-12/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:333</span></span>
<span class="line"><span>2025-03-14 20:57:22.686776: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 16 bytes spill stores, 16 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:22.760109: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 276 bytes spill stores, 276 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:22.761075: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 336 bytes spill stores, 336 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:22.955591: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:23.107414: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:23.407235: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 1176 bytes spill stores, 1148 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:23.430136: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 320 bytes spill stores, 320 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:23.547011: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24_0&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:23.709653: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_35_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:24.388632: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:24.398106: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 648 bytes spill stores, 652 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:24.489386: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_35&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:25.155306: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 128 bytes spill stores, 128 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:25.317065: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_35&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:57:25.330621: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_26&#39;, 304 bytes spill stores, 304 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741985845.487279 2659715 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1741985845.488580 2659715 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1741985845.488590 2659715 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1741985845.488598 2659715 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1741985845.488605 2659715 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1741985845.488613 2659715 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1741985845.488622 2659715 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1741985845.488629 2659715 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1741985845.488636 2659715 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1741985845.488643 2659715 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-14 20:57:25.488659: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.491515 2659715 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1741985845.491532 2659715 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1741985845.491535 2659715 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1741985845.491538 2659715 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1741985845.491542 2659715 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1741985845.491544 2659715 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1741985845.491547 2659715 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1741985845.491550 2659715 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1741985845.491553 2659715 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1741985845.491556 2659715 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-14 20:57:25.491561: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.493761 2659715 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1741985845.493778 2659715 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1741985845.493781 2659715 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1741985845.493784 2659715 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1741985845.493787 2659715 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1741985845.493790 2659715 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1741985845.493793 2659715 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1741985845.493796 2659715 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1741985845.493799 2659715 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1741985845.493802 2659715 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-14 20:57:25.493806: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.496019 2659715 buffer_comparator.cc:156] Difference at 32: 0, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1741985845.496037 2659715 buffer_comparator.cc:156] Difference at 33: 0, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1741985845.496040 2659715 buffer_comparator.cc:156] Difference at 34: 0, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1741985845.496043 2659715 buffer_comparator.cc:156] Difference at 35: 0, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1741985845.496046 2659715 buffer_comparator.cc:156] Difference at 36: 0, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1741985845.496049 2659715 buffer_comparator.cc:156] Difference at 37: 0, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1741985845.496052 2659715 buffer_comparator.cc:156] Difference at 38: 0, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1741985845.496055 2659715 buffer_comparator.cc:156] Difference at 39: 0, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1741985845.496058 2659715 buffer_comparator.cc:156] Difference at 40: 0, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1741985845.496061 2659715 buffer_comparator.cc:156] Difference at 41: 0, expected 13.7427</span></span>
<span class="line"><span>2025-03-14 20:57:25.496067: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.498270 2659715 buffer_comparator.cc:156] Difference at 32: 0, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1741985845.498286 2659715 buffer_comparator.cc:156] Difference at 33: 0, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1741985845.498289 2659715 buffer_comparator.cc:156] Difference at 34: 0, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1741985845.498292 2659715 buffer_comparator.cc:156] Difference at 35: 0, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1741985845.498295 2659715 buffer_comparator.cc:156] Difference at 36: 0, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1741985845.498298 2659715 buffer_comparator.cc:156] Difference at 37: 0, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1741985845.498301 2659715 buffer_comparator.cc:156] Difference at 38: 0, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1741985845.498304 2659715 buffer_comparator.cc:156] Difference at 39: 0, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1741985845.498306 2659715 buffer_comparator.cc:156] Difference at 40: 0, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1741985845.498309 2659715 buffer_comparator.cc:156] Difference at 41: 0, expected 13.7427</span></span>
<span class="line"><span>2025-03-14 20:57:25.498314: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.500562 2659715 buffer_comparator.cc:156] Difference at 0: 16.5257, expected 14.4011</span></span>
<span class="line"><span>E0000 00:00:1741985845.500580 2659715 buffer_comparator.cc:156] Difference at 1: 19.4064, expected 15.9904</span></span>
<span class="line"><span>E0000 00:00:1741985845.500583 2659715 buffer_comparator.cc:156] Difference at 2: 16.1909, expected 13.4103</span></span>
<span class="line"><span>E0000 00:00:1741985845.500587 2659715 buffer_comparator.cc:156] Difference at 6: 13.1689, expected 11.4953</span></span>
<span class="line"><span>E0000 00:00:1741985845.500590 2659715 buffer_comparator.cc:156] Difference at 9: 16.2882, expected 14.2452</span></span>
<span class="line"><span>E0000 00:00:1741985845.500593 2659715 buffer_comparator.cc:156] Difference at 11: 15.6385, expected 13.739</span></span>
<span class="line"><span>E0000 00:00:1741985845.500596 2659715 buffer_comparator.cc:156] Difference at 12: 20.6748, expected 16.297</span></span>
<span class="line"><span>E0000 00:00:1741985845.500599 2659715 buffer_comparator.cc:156] Difference at 13: 17.2352, expected 14.372</span></span>
<span class="line"><span>E0000 00:00:1741985845.500602 2659715 buffer_comparator.cc:156] Difference at 14: 14.761, expected 12.4213</span></span>
<span class="line"><span>E0000 00:00:1741985845.500605 2659715 buffer_comparator.cc:156] Difference at 16: 17.262, expected 15.1227</span></span>
<span class="line"><span>2025-03-14 20:57:25.500610: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.502831 2659715 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1741985845.502847 2659715 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1741985845.502850 2659715 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1741985845.502853 2659715 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1741985845.502856 2659715 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1741985845.502859 2659715 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1741985845.502862 2659715 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1741985845.502865 2659715 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1741985845.502868 2659715 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1741985845.502870 2659715 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-14 20:57:25.502875: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.505068 2659715 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1741985845.505083 2659715 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1741985845.505088 2659715 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1741985845.505091 2659715 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1741985845.505094 2659715 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1741985845.505097 2659715 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1741985845.505100 2659715 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1741985845.505103 2659715 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1741985845.505106 2659715 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1741985845.505109 2659715 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-14 20:57:25.505113: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.507308 2659715 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1741985845.507329 2659715 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1741985845.507332 2659715 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1741985845.507335 2659715 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1741985845.507338 2659715 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1741985845.507341 2659715 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1741985845.507344 2659715 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1741985845.507347 2659715 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1741985845.507350 2659715 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1741985845.507353 2659715 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-14 20:57:25.507357: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.509550 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1741985845.509564 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1741985845.509568 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1741985845.509571 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1741985845.509574 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1741985845.509576 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1741985845.509579 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1741985845.509582 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1741985845.509585 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1741985845.509588 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-14 20:57:25.509593: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.511809 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1741985845.511822 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1741985845.511826 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1741985845.511829 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1741985845.511832 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1741985845.511834 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1741985845.511839 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1741985845.511842 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1741985845.511845 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1741985845.511848 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-14 20:57:25.511853: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.514059 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1741985845.514075 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1741985845.514078 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1741985845.514081 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1741985845.514084 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1741985845.514087 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1741985845.514090 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1741985845.514093 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1741985845.514096 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1741985845.514098 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-14 20:57:25.514103: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.516301 2659715 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1741985845.516324 2659715 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1741985845.516327 2659715 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1741985845.516330 2659715 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1741985845.516333 2659715 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1741985845.516336 2659715 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1741985845.516339 2659715 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1741985845.516342 2659715 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1741985845.516345 2659715 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1741985845.516348 2659715 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-14 20:57:25.516352: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.518580 2659715 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1741985845.518595 2659715 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1741985845.518598 2659715 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1741985845.518601 2659715 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1741985845.518604 2659715 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1741985845.518607 2659715 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1741985845.518610 2659715 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1741985845.518613 2659715 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1741985845.518616 2659715 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1741985845.518619 2659715 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-14 20:57:25.518625: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.520831 2659715 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1741985845.520851 2659715 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1741985845.520854 2659715 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1741985845.520857 2659715 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1741985845.520860 2659715 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1741985845.520863 2659715 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1741985845.520866 2659715 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1741985845.520869 2659715 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1741985845.520872 2659715 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1741985845.520875 2659715 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-14 20:57:25.520879: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.523115 2659715 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1741985845.523135 2659715 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1741985845.523138 2659715 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1741985845.523141 2659715 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1741985845.523144 2659715 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1741985845.523147 2659715 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1741985845.523150 2659715 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1741985845.523153 2659715 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1741985845.523156 2659715 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1741985845.523159 2659715 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-14 20:57:25.523164: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.532440 2659715 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1741985845.532477 2659715 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1741985845.532481 2659715 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1741985845.532484 2659715 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1741985845.532487 2659715 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1741985845.532490 2659715 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1741985845.532493 2659715 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1741985845.532496 2659715 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1741985845.532499 2659715 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.532502 2659715 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-03-14 20:57:25.532509: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.538486 2659715 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1741985845.538521 2659715 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1741985845.538525 2659715 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1741985845.538528 2659715 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1741985845.538532 2659715 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1741985845.538534 2659715 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1741985845.538537 2659715 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1741985845.538540 2659715 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1741985845.538543 2659715 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.538546 2659715 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-03-14 20:57:25.538553: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.543624 2659715 buffer_comparator.cc:156] Difference at 64: 0, expected 1106.21</span></span>
<span class="line"><span>E0000 00:00:1741985845.543665 2659715 buffer_comparator.cc:156] Difference at 65: 0, expected 1087.83</span></span>
<span class="line"><span>E0000 00:00:1741985845.543669 2659715 buffer_comparator.cc:156] Difference at 66: 0, expected 1090.54</span></span>
<span class="line"><span>E0000 00:00:1741985845.543672 2659715 buffer_comparator.cc:156] Difference at 67: 0, expected 1104.23</span></span>
<span class="line"><span>E0000 00:00:1741985845.543675 2659715 buffer_comparator.cc:156] Difference at 68: 0, expected 1104.3</span></span>
<span class="line"><span>E0000 00:00:1741985845.543678 2659715 buffer_comparator.cc:156] Difference at 69: 0, expected 1093.45</span></span>
<span class="line"><span>E0000 00:00:1741985845.543681 2659715 buffer_comparator.cc:156] Difference at 70: 0, expected 1091.52</span></span>
<span class="line"><span>E0000 00:00:1741985845.543684 2659715 buffer_comparator.cc:156] Difference at 71: 0, expected 1110.4</span></span>
<span class="line"><span>E0000 00:00:1741985845.543687 2659715 buffer_comparator.cc:156] Difference at 72: 0, expected 1106.92</span></span>
<span class="line"><span>E0000 00:00:1741985845.543690 2659715 buffer_comparator.cc:156] Difference at 73: 0, expected 1088.44</span></span>
<span class="line"><span>2025-03-14 20:57:25.543697: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.548610 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.548633 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.548636 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.548639 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.548642 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.548645 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.548648 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.548651 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.548654 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.548657 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.548662: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.553531 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.553554 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.553557 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.553560 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.553563 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.553566 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.553571 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.553573 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.553576 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.553579 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.553585: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.558175 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.558201 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.558204 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.558207 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.558210 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.558213 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.558216 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.558219 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.558222 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.558225 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.558231: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.562783 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.562818 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.562822 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.562825 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.562828 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.562831 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.562834 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.562837 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.562840 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.562843 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.562850: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.567557 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.567587 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.567590 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.567593 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.567596 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.567599 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.567602 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.567605 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.567608 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.567611 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.567619: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.572336 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.572356 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.572360 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.572363 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.572366 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.572369 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.572371 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.572374 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.572377 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.572380 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.572386: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.577014 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.577035 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.577038 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.577041 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.577044 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.577047 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.577050 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.577053 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.577056 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.577059 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.577064: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.581424 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.581451 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.581455 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.581458 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.581461 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.581464 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.581467 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.581470 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.581473 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.581476 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.581482: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.585826 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.585861 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.585866 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.585869 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.585872 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.585875 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.585878 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.585881 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.585884 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.585887 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.585894: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.590431 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.590455 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.590458 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.590461 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.590464 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.590467 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.590470 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.590473 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.590476 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.590479 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.590484: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.594703 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.594722 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.594725 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.594728 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.594731 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.594734 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.594737 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.594740 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.594743 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.594746 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.594751: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.598879 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.598906 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.598909 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.598912 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.598915 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.598918 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.598923 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.598926 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.598929 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.598932 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.598937: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.603384 2659715 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741985845.603416 2659715 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741985845.603419 2659715 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741985845.603422 2659715 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741985845.603425 2659715 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741985845.603428 2659715 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741985845.603431 2659715 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741985845.603434 2659715 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741985845.603437 2659715 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741985845.603440 2659715 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-14 20:57:25.603446: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.607811 2659715 buffer_comparator.cc:156] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1741985845.607848 2659715 buffer_comparator.cc:156] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1741985845.607852 2659715 buffer_comparator.cc:156] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1741985845.607854 2659715 buffer_comparator.cc:156] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1741985845.607857 2659715 buffer_comparator.cc:156] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1741985845.607860 2659715 buffer_comparator.cc:156] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1741985845.607863 2659715 buffer_comparator.cc:156] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1741985845.607866 2659715 buffer_comparator.cc:156] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1741985845.607869 2659715 buffer_comparator.cc:156] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1741985845.607872 2659715 buffer_comparator.cc:156] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-03-14 20:57:25.607879: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.611516 2659715 buffer_comparator.cc:156] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741985845.611536 2659715 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741985845.611539 2659715 buffer_comparator.cc:156] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741985845.611542 2659715 buffer_comparator.cc:156] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741985845.611545 2659715 buffer_comparator.cc:156] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741985845.611547 2659715 buffer_comparator.cc:156] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741985845.611550 2659715 buffer_comparator.cc:156] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741985845.611553 2659715 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741985845.611556 2659715 buffer_comparator.cc:156] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741985845.611559 2659715 buffer_comparator.cc:156] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-03-14 20:57:25.611566: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.613993 2659715 buffer_comparator.cc:156] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741985845.614009 2659715 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741985845.614012 2659715 buffer_comparator.cc:156] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741985845.614015 2659715 buffer_comparator.cc:156] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741985845.614018 2659715 buffer_comparator.cc:156] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741985845.614021 2659715 buffer_comparator.cc:156] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741985845.614024 2659715 buffer_comparator.cc:156] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741985845.614027 2659715 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741985845.614029 2659715 buffer_comparator.cc:156] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741985845.614032 2659715 buffer_comparator.cc:156] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-03-14 20:57:25.614037: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.616439 2659715 buffer_comparator.cc:156] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741985845.616454 2659715 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741985845.616457 2659715 buffer_comparator.cc:156] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741985845.616460 2659715 buffer_comparator.cc:156] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741985845.616463 2659715 buffer_comparator.cc:156] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741985845.616466 2659715 buffer_comparator.cc:156] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741985845.616468 2659715 buffer_comparator.cc:156] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741985845.616471 2659715 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741985845.616474 2659715 buffer_comparator.cc:156] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741985845.616477 2659715 buffer_comparator.cc:156] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-03-14 20:57:25.616482: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.618877 2659715 buffer_comparator.cc:156] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741985845.618893 2659715 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741985845.618897 2659715 buffer_comparator.cc:156] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741985845.618899 2659715 buffer_comparator.cc:156] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741985845.618902 2659715 buffer_comparator.cc:156] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741985845.618905 2659715 buffer_comparator.cc:156] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741985845.618908 2659715 buffer_comparator.cc:156] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741985845.618911 2659715 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741985845.618913 2659715 buffer_comparator.cc:156] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741985845.618916 2659715 buffer_comparator.cc:156] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-03-14 20:57:25.618921: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.621326 2659715 buffer_comparator.cc:156] Difference at 32: -nan, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741985845.621344 2659715 buffer_comparator.cc:156] Difference at 33: -nan, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741985845.621347 2659715 buffer_comparator.cc:156] Difference at 34: -nan, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741985845.621350 2659715 buffer_comparator.cc:156] Difference at 35: -nan, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741985845.621353 2659715 buffer_comparator.cc:156] Difference at 36: -nan, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741985845.621356 2659715 buffer_comparator.cc:156] Difference at 37: -nan, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741985845.621359 2659715 buffer_comparator.cc:156] Difference at 38: -nan, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741985845.621362 2659715 buffer_comparator.cc:156] Difference at 39: -nan, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741985845.621364 2659715 buffer_comparator.cc:156] Difference at 40: -nan, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741985845.621367 2659715 buffer_comparator.cc:156] Difference at 41: -nan, expected 20.3484</span></span>
<span class="line"><span>2025-03-14 20:57:25.621372: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.623796 2659715 buffer_comparator.cc:156] Difference at 32: -nan, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741985845.623820 2659715 buffer_comparator.cc:156] Difference at 33: -nan, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741985845.623823 2659715 buffer_comparator.cc:156] Difference at 34: -nan, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741985845.623826 2659715 buffer_comparator.cc:156] Difference at 35: -nan, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741985845.623829 2659715 buffer_comparator.cc:156] Difference at 36: -nan, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741985845.623832 2659715 buffer_comparator.cc:156] Difference at 37: -nan, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741985845.623835 2659715 buffer_comparator.cc:156] Difference at 38: -nan, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741985845.623838 2659715 buffer_comparator.cc:156] Difference at 39: -nan, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741985845.623840 2659715 buffer_comparator.cc:156] Difference at 40: -nan, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741985845.623843 2659715 buffer_comparator.cc:156] Difference at 41: -nan, expected 20.3484</span></span>
<span class="line"><span>2025-03-14 20:57:25.623848: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.626266 2659715 buffer_comparator.cc:156] Difference at 32: -nan, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741985845.626289 2659715 buffer_comparator.cc:156] Difference at 33: -nan, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741985845.626292 2659715 buffer_comparator.cc:156] Difference at 34: -nan, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741985845.626295 2659715 buffer_comparator.cc:156] Difference at 35: -nan, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741985845.626298 2659715 buffer_comparator.cc:156] Difference at 36: -nan, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741985845.626301 2659715 buffer_comparator.cc:156] Difference at 37: -nan, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741985845.626304 2659715 buffer_comparator.cc:156] Difference at 38: -nan, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741985845.626307 2659715 buffer_comparator.cc:156] Difference at 39: -nan, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741985845.626309 2659715 buffer_comparator.cc:156] Difference at 40: -nan, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741985845.626312 2659715 buffer_comparator.cc:156] Difference at 41: -nan, expected 20.3484</span></span>
<span class="line"><span>2025-03-14 20:57:25.626317: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.628755 2659715 buffer_comparator.cc:156] Difference at 32: -nan, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741985845.628777 2659715 buffer_comparator.cc:156] Difference at 33: -nan, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741985845.628780 2659715 buffer_comparator.cc:156] Difference at 34: -nan, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741985845.628783 2659715 buffer_comparator.cc:156] Difference at 35: -nan, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741985845.628788 2659715 buffer_comparator.cc:156] Difference at 36: -nan, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741985845.628791 2659715 buffer_comparator.cc:156] Difference at 37: -nan, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741985845.628793 2659715 buffer_comparator.cc:156] Difference at 38: -nan, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741985845.628796 2659715 buffer_comparator.cc:156] Difference at 39: -nan, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741985845.628799 2659715 buffer_comparator.cc:156] Difference at 40: -nan, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741985845.628802 2659715 buffer_comparator.cc:156] Difference at 41: -nan, expected 20.3484</span></span>
<span class="line"><span>2025-03-14 20:57:25.628808: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.631229 2659715 buffer_comparator.cc:156] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741985845.631247 2659715 buffer_comparator.cc:156] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741985845.631250 2659715 buffer_comparator.cc:156] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741985845.631253 2659715 buffer_comparator.cc:156] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741985845.631256 2659715 buffer_comparator.cc:156] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741985845.631259 2659715 buffer_comparator.cc:156] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741985845.631262 2659715 buffer_comparator.cc:156] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741985845.631264 2659715 buffer_comparator.cc:156] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741985845.631267 2659715 buffer_comparator.cc:156] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741985845.631270 2659715 buffer_comparator.cc:156] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-03-14 20:57:25.631274: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.633668 2659715 buffer_comparator.cc:156] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741985845.633687 2659715 buffer_comparator.cc:156] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741985845.633690 2659715 buffer_comparator.cc:156] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741985845.633693 2659715 buffer_comparator.cc:156] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741985845.633696 2659715 buffer_comparator.cc:156] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741985845.633699 2659715 buffer_comparator.cc:156] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741985845.633701 2659715 buffer_comparator.cc:156] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741985845.633704 2659715 buffer_comparator.cc:156] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741985845.633707 2659715 buffer_comparator.cc:156] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741985845.633710 2659715 buffer_comparator.cc:156] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-03-14 20:57:25.633714: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.636130 2659715 buffer_comparator.cc:156] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741985845.636153 2659715 buffer_comparator.cc:156] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741985845.636156 2659715 buffer_comparator.cc:156] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741985845.636159 2659715 buffer_comparator.cc:156] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741985845.636162 2659715 buffer_comparator.cc:156] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741985845.636164 2659715 buffer_comparator.cc:156] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741985845.636167 2659715 buffer_comparator.cc:156] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741985845.636171 2659715 buffer_comparator.cc:156] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741985845.636174 2659715 buffer_comparator.cc:156] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741985845.636177 2659715 buffer_comparator.cc:156] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-03-14 20:57:25.636181: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.638574 2659715 buffer_comparator.cc:156] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741985845.638589 2659715 buffer_comparator.cc:156] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741985845.638592 2659715 buffer_comparator.cc:156] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741985845.638595 2659715 buffer_comparator.cc:156] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741985845.638598 2659715 buffer_comparator.cc:156] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741985845.638601 2659715 buffer_comparator.cc:156] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741985845.638604 2659715 buffer_comparator.cc:156] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741985845.638606 2659715 buffer_comparator.cc:156] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741985845.638609 2659715 buffer_comparator.cc:156] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741985845.638612 2659715 buffer_comparator.cc:156] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-03-14 20:57:25.638616: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.641041 2659715 buffer_comparator.cc:156] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741985845.641059 2659715 buffer_comparator.cc:156] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741985845.641062 2659715 buffer_comparator.cc:156] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741985845.641065 2659715 buffer_comparator.cc:156] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741985845.641068 2659715 buffer_comparator.cc:156] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741985845.641071 2659715 buffer_comparator.cc:156] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741985845.641074 2659715 buffer_comparator.cc:156] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741985845.641076 2659715 buffer_comparator.cc:156] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741985845.641079 2659715 buffer_comparator.cc:156] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741985845.641082 2659715 buffer_comparator.cc:156] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-03-14 20:57:25.641087: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.643485 2659715 buffer_comparator.cc:156] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741985845.643503 2659715 buffer_comparator.cc:156] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741985845.643506 2659715 buffer_comparator.cc:156] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741985845.643509 2659715 buffer_comparator.cc:156] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741985845.643512 2659715 buffer_comparator.cc:156] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741985845.643515 2659715 buffer_comparator.cc:156] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741985845.643517 2659715 buffer_comparator.cc:156] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741985845.643520 2659715 buffer_comparator.cc:156] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741985845.643523 2659715 buffer_comparator.cc:156] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741985845.643526 2659715 buffer_comparator.cc:156] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-03-14 20:57:25.643533: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.645951 2659715 buffer_comparator.cc:156] Difference at 128: -nan, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1741985845.645971 2659715 buffer_comparator.cc:156] Difference at 129: -nan, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1741985845.645975 2659715 buffer_comparator.cc:156] Difference at 130: -nan, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1741985845.645978 2659715 buffer_comparator.cc:156] Difference at 131: -nan, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1741985845.645980 2659715 buffer_comparator.cc:156] Difference at 132: -nan, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1741985845.645983 2659715 buffer_comparator.cc:156] Difference at 133: -nan, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1741985845.645986 2659715 buffer_comparator.cc:156] Difference at 134: -nan, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1741985845.645989 2659715 buffer_comparator.cc:156] Difference at 135: -nan, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1741985845.645992 2659715 buffer_comparator.cc:156] Difference at 136: -nan, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1741985845.645994 2659715 buffer_comparator.cc:156] Difference at 137: -nan, expected 18.5916</span></span>
<span class="line"><span>2025-03-14 20:57:25.646000: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.648432 2659715 buffer_comparator.cc:156] Difference at 128: -nan, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1741985845.648455 2659715 buffer_comparator.cc:156] Difference at 129: -nan, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1741985845.648458 2659715 buffer_comparator.cc:156] Difference at 130: -nan, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1741985845.648461 2659715 buffer_comparator.cc:156] Difference at 131: -nan, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1741985845.648464 2659715 buffer_comparator.cc:156] Difference at 132: -nan, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1741985845.648466 2659715 buffer_comparator.cc:156] Difference at 133: -nan, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1741985845.648469 2659715 buffer_comparator.cc:156] Difference at 134: -nan, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1741985845.648472 2659715 buffer_comparator.cc:156] Difference at 135: -nan, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1741985845.648475 2659715 buffer_comparator.cc:156] Difference at 136: -nan, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1741985845.648478 2659715 buffer_comparator.cc:156] Difference at 137: -nan, expected 18.5916</span></span>
<span class="line"><span>2025-03-14 20:57:25.648483: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985845.650928 2659715 buffer_comparator.cc:156] Difference at 128: -nan, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1741985845.650945 2659715 buffer_comparator.cc:156] Difference at 129: -nan, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1741985845.650948 2659715 buffer_comparator.cc:156] Difference at 130: -nan, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1741985845.650951 2659715 buffer_comparator.cc:156] Difference at 131: -nan, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1741985845.650954 2659715 buffer_comparator.cc:156] Difference at 132: -nan, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1741985845.650957 2659715 buffer_comparator.cc:156] Difference at 133: -nan, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1741985845.650960 2659715 buffer_comparator.cc:156] Difference at 134: -nan, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1741985845.650962 2659715 buffer_comparator.cc:156] Difference at 135: -nan, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1741985845.650965 2659715 buffer_comparator.cc:156] Difference at 136: -nan, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1741985845.650968 2659715 buffer_comparator.cc:156] Difference at 137: -nan, expected 18.5916</span></span>
<span class="line"><span>2025-03-14 20:57:25.650973: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Epoch   1	Train Loss: 16.470459	Train Acc: 22.8571%	Val Loss: 7.062370	Val Acc: 24.8000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 8.748114	Train Acc: 23.5714%	Val Loss: 3.164325	Val Acc: 30.6000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 3.634000	Train Acc: 45.7143%	Val Loss: 1.767279	Val Acc: 43.6000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 1.844923	Train Acc: 54.2857%	Val Loss: 1.774289	Val Acc: 44.8000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 1.519016	Train Acc: 60.7143%	Val Loss: 1.720779	Val Acc: 47.8000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.320785	Train Acc: 70.0000%	Val Loss: 1.585779	Val Acc: 54.6000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.153206	Train Acc: 72.8571%	Val Loss: 1.513280	Val Acc: 59.8000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 1.070447	Train Acc: 77.1429%	Val Loss: 1.483784	Val Acc: 61.2000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 1.009521	Train Acc: 77.1429%	Val Loss: 1.505568	Val Acc: 62.0000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 0.969393	Train Acc: 77.1429%	Val Loss: 1.528395	Val Acc: 62.2000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 0.911441	Train Acc: 78.5714%	Val Loss: 1.535004	Val Acc: 63.0000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 0.876338	Train Acc: 79.2857%	Val Loss: 1.508996	Val Acc: 64.2000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.782100	Train Acc: 79.2857%	Val Loss: 1.503073	Val Acc: 64.2000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 0.688038	Train Acc: 79.2857%	Val Loss: 1.522324	Val Acc: 65.8000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 0.625099	Train Acc: 81.4286%	Val Loss: 1.555677	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.590631	Train Acc: 82.1429%	Val Loss: 1.587869	Val Acc: 67.0000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.572871	Train Acc: 82.8571%	Val Loss: 1.602481	Val Acc: 67.4000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.551022	Train Acc: 85.0000%	Val Loss: 1.595073	Val Acc: 67.2000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.518503	Train Acc: 85.0000%	Val Loss: 1.579118	Val Acc: 67.4000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.492048	Train Acc: 83.5714%	Val Loss: 1.565115	Val Acc: 68.0000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.469935	Train Acc: 84.2857%	Val Loss: 1.557257	Val Acc: 67.6000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.452317	Train Acc: 85.0000%	Val Loss: 1.557843	Val Acc: 67.0000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.440072	Train Acc: 85.0000%	Val Loss: 1.566017	Val Acc: 66.8000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.423191	Train Acc: 86.4286%	Val Loss: 1.581582	Val Acc: 66.0000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.404572	Train Acc: 86.4286%	Val Loss: 1.604523	Val Acc: 65.8000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.390305	Train Acc: 87.8571%	Val Loss: 1.629158	Val Acc: 66.0000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.378975	Train Acc: 88.5714%	Val Loss: 1.653721	Val Acc: 66.2000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.368412	Train Acc: 88.5714%	Val Loss: 1.678228	Val Acc: 66.4000%</span></span>
<span class="line"><span>Early Stopping at Epoch 28</span></span>
<span class="line"><span>2025-03-14 20:58:10.194139: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_35&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:58:10.391168: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_35&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:58:10.489203: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_35_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741985890.499284 2659715 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741985890.499349 2659715 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741985890.499358 2659715 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741985890.499365 2659715 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.499372 2659715 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741985890.499380 2659715 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741985890.499387 2659715 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741985890.499394 2659715 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741985890.499401 2659715 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741985890.499408 2659715 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-14 20:58:10.499423: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.502313 2659715 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741985890.502343 2659715 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741985890.502351 2659715 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741985890.502358 2659715 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.502365 2659715 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741985890.502372 2659715 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741985890.502380 2659715 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741985890.502387 2659715 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741985890.502394 2659715 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741985890.502401 2659715 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-14 20:58:10.502412: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.505070 2659715 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741985890.505084 2659715 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741985890.505088 2659715 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741985890.505091 2659715 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.505094 2659715 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741985890.505097 2659715 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741985890.505100 2659715 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741985890.505103 2659715 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741985890.505108 2659715 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741985890.505111 2659715 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-14 20:58:10.505116: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.507733 2659715 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741985890.507748 2659715 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741985890.507752 2659715 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741985890.507755 2659715 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.507758 2659715 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741985890.507761 2659715 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741985890.507764 2659715 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741985890.507767 2659715 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741985890.507770 2659715 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741985890.507773 2659715 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-14 20:58:10.507778: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.510394 2659715 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741985890.510409 2659715 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741985890.510412 2659715 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741985890.510415 2659715 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.510418 2659715 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741985890.510421 2659715 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741985890.510425 2659715 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741985890.510428 2659715 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741985890.510431 2659715 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741985890.510434 2659715 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-14 20:58:10.510439: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.513068 2659715 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741985890.513082 2659715 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741985890.513085 2659715 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741985890.513088 2659715 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.513091 2659715 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741985890.513094 2659715 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741985890.513098 2659715 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741985890.513101 2659715 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741985890.513104 2659715 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741985890.513107 2659715 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-14 20:58:10.513113: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.515693 2659715 buffer_comparator.cc:156] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1741985890.515708 2659715 buffer_comparator.cc:156] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741985890.515711 2659715 buffer_comparator.cc:156] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741985890.515714 2659715 buffer_comparator.cc:156] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741985890.515717 2659715 buffer_comparator.cc:156] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741985890.515720 2659715 buffer_comparator.cc:156] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741985890.515723 2659715 buffer_comparator.cc:156] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741985890.515726 2659715 buffer_comparator.cc:156] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1741985890.515730 2659715 buffer_comparator.cc:156] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741985890.515733 2659715 buffer_comparator.cc:156] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-03-14 20:58:10.515737: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.518316 2659715 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741985890.518329 2659715 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741985890.518332 2659715 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741985890.518336 2659715 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741985890.518339 2659715 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741985890.518342 2659715 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741985890.518345 2659715 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741985890.518348 2659715 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1741985890.518351 2659715 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1741985890.518354 2659715 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-14 20:58:10.518359: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.520956 2659715 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741985890.520971 2659715 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741985890.520975 2659715 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741985890.520978 2659715 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741985890.520981 2659715 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741985890.520984 2659715 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741985890.520987 2659715 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741985890.520990 2659715 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1741985890.520993 2659715 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1741985890.520997 2659715 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-14 20:58:10.521002: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.523616 2659715 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741985890.523632 2659715 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741985890.523635 2659715 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741985890.523638 2659715 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741985890.523641 2659715 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741985890.523644 2659715 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741985890.523648 2659715 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.523651 2659715 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.523654 2659715 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741985890.523657 2659715 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-14 20:58:10.523662: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.526224 2659715 buffer_comparator.cc:156] Difference at 7: 1058.92, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1741985890.526239 2659715 buffer_comparator.cc:156] Difference at 11: 1263.92, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1741985890.526243 2659715 buffer_comparator.cc:156] Difference at 179: 1223.75, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1741985890.526247 2659715 buffer_comparator.cc:156] Difference at 266: 1047.35, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1741985890.526250 2659715 buffer_comparator.cc:156] Difference at 270: 1246.8, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1741985890.526254 2659715 buffer_comparator.cc:156] Difference at 417: 1222.47, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1741985890.526258 2659715 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741985890.526261 2659715 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741985890.526264 2659715 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741985890.526267 2659715 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>2025-03-14 20:58:10.526271: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.528838 2659715 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741985890.528851 2659715 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741985890.528855 2659715 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741985890.528858 2659715 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741985890.528861 2659715 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741985890.528864 2659715 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741985890.528867 2659715 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.528870 2659715 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.528873 2659715 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741985890.528876 2659715 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-14 20:58:10.528881: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.531460 2659715 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741985890.531475 2659715 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741985890.531479 2659715 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741985890.531482 2659715 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741985890.531485 2659715 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741985890.531488 2659715 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741985890.531491 2659715 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.531494 2659715 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.531497 2659715 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741985890.531501 2659715 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-14 20:58:10.531505: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.534085 2659715 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741985890.534099 2659715 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741985890.534102 2659715 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741985890.534105 2659715 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741985890.534108 2659715 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741985890.534112 2659715 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741985890.534115 2659715 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.534118 2659715 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.534121 2659715 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741985890.534124 2659715 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-14 20:58:10.534128: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.536686 2659715 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741985890.536701 2659715 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741985890.536704 2659715 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741985890.536707 2659715 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741985890.536710 2659715 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741985890.536713 2659715 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741985890.536716 2659715 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.536719 2659715 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741985890.536723 2659715 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741985890.536726 2659715 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-14 20:58:10.536731: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.539429 2659715 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741985890.539446 2659715 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741985890.539449 2659715 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741985890.539455 2659715 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741985890.539458 2659715 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741985890.539461 2659715 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741985890.539464 2659715 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741985890.539467 2659715 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741985890.539471 2659715 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741985890.539474 2659715 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-14 20:58:10.539479: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.542145 2659715 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741985890.542162 2659715 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741985890.542166 2659715 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741985890.542169 2659715 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741985890.542172 2659715 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741985890.542175 2659715 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741985890.542178 2659715 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741985890.542181 2659715 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741985890.542184 2659715 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741985890.542187 2659715 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-14 20:58:10.542193: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.544809 2659715 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741985890.544825 2659715 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741985890.544828 2659715 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741985890.544831 2659715 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741985890.544834 2659715 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741985890.544837 2659715 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741985890.544840 2659715 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741985890.544844 2659715 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741985890.544847 2659715 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741985890.544850 2659715 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-14 20:58:10.544855: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.547463 2659715 buffer_comparator.cc:156] Difference at 896: 485.098, expected 958.133</span></span>
<span class="line"><span>E0000 00:00:1741985890.547480 2659715 buffer_comparator.cc:156] Difference at 897: 732.587, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741985890.547483 2659715 buffer_comparator.cc:156] Difference at 898: 635.29, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741985890.547487 2659715 buffer_comparator.cc:156] Difference at 899: 446.948, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741985890.547490 2659715 buffer_comparator.cc:156] Difference at 900: 712.745, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741985890.547495 2659715 buffer_comparator.cc:156] Difference at 901: 516.07, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741985890.547498 2659715 buffer_comparator.cc:156] Difference at 902: 373.095, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741985890.547501 2659715 buffer_comparator.cc:156] Difference at 903: 483.905, expected 941.483</span></span>
<span class="line"><span>E0000 00:00:1741985890.547504 2659715 buffer_comparator.cc:156] Difference at 904: 721.412, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741985890.547507 2659715 buffer_comparator.cc:156] Difference at 905: 633.571, expected 1817.42</span></span>
<span class="line"><span>2025-03-14 20:58:10.547513: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.550274 2659715 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741985890.550289 2659715 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741985890.550292 2659715 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741985890.550295 2659715 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.550299 2659715 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741985890.550302 2659715 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741985890.550305 2659715 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741985890.550308 2659715 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741985890.550311 2659715 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.550314 2659715 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-14 20:58:10.550319: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.553061 2659715 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741985890.553075 2659715 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741985890.553078 2659715 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741985890.553081 2659715 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.553084 2659715 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741985890.553088 2659715 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741985890.553091 2659715 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741985890.553094 2659715 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741985890.553097 2659715 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.553100 2659715 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-14 20:58:10.553105: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985890.555814 2659715 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741985890.555828 2659715 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741985890.555831 2659715 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741985890.555834 2659715 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.555837 2659715 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741985890.555840 2659715 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741985890.555843 2659715 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741985890.555848 2659715 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741985890.555851 2659715 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741985890.555854 2659715 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-14 20:58:10.555859: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-03-14 20:58:11.911792: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:58:12.490192: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-14 20:58:12.664956: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741985892.672373 2659715 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741985892.672434 2659715 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741985892.672443 2659715 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741985892.672451 2659715 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741985892.672457 2659715 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741985892.672464 2659715 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741985892.672471 2659715 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741985892.672478 2659715 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741985892.672484 2659715 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741985892.672491 2659715 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-14 20:58:12.672505: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.676103 2659715 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741985892.676133 2659715 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741985892.676141 2659715 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741985892.676148 2659715 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741985892.676154 2659715 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741985892.676161 2659715 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741985892.676168 2659715 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741985892.676174 2659715 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741985892.676181 2659715 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741985892.676188 2659715 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-14 20:58:12.676198: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.679790 2659715 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741985892.679807 2659715 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741985892.679810 2659715 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741985892.679813 2659715 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741985892.679818 2659715 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741985892.679821 2659715 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741985892.679824 2659715 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741985892.679827 2659715 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741985892.679830 2659715 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741985892.679833 2659715 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-14 20:58:12.679838: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.683193 2659715 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741985892.683208 2659715 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741985892.683211 2659715 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741985892.683214 2659715 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741985892.683217 2659715 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741985892.683220 2659715 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741985892.683223 2659715 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741985892.683226 2659715 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741985892.683228 2659715 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741985892.683231 2659715 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-14 20:58:12.683236: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.686784 2659715 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741985892.686799 2659715 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741985892.686802 2659715 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741985892.686805 2659715 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741985892.686808 2659715 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741985892.686811 2659715 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741985892.686814 2659715 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741985892.686817 2659715 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741985892.686820 2659715 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741985892.686823 2659715 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-14 20:58:12.686828: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.690222 2659715 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741985892.690239 2659715 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741985892.690242 2659715 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741985892.690245 2659715 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741985892.690248 2659715 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741985892.690251 2659715 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741985892.690254 2659715 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741985892.690257 2659715 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741985892.690262 2659715 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741985892.690265 2659715 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-14 20:58:12.690270: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.693589 2659715 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741985892.693602 2659715 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741985892.693606 2659715 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741985892.693609 2659715 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741985892.693612 2659715 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741985892.693615 2659715 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741985892.693618 2659715 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741985892.693620 2659715 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741985892.693623 2659715 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741985892.693626 2659715 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-14 20:58:12.693631: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.696921 2659715 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741985892.696938 2659715 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741985892.696941 2659715 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741985892.696944 2659715 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741985892.696947 2659715 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741985892.696950 2659715 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741985892.696953 2659715 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741985892.696956 2659715 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741985892.696959 2659715 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741985892.696962 2659715 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-14 20:58:12.696967: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.700330 2659715 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741985892.700344 2659715 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741985892.700347 2659715 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741985892.700350 2659715 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741985892.700353 2659715 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741985892.700356 2659715 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741985892.700359 2659715 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741985892.700362 2659715 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741985892.700365 2659715 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741985892.700368 2659715 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-14 20:58:12.700373: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.703709 2659715 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741985892.703725 2659715 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741985892.703728 2659715 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741985892.703731 2659715 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741985892.703734 2659715 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741985892.703737 2659715 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741985892.703740 2659715 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741985892.703743 2659715 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741985892.703746 2659715 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741985892.703748 2659715 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-14 20:58:12.703753: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.707036 2659715 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741985892.707052 2659715 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741985892.707056 2659715 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741985892.707059 2659715 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741985892.707062 2659715 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741985892.707065 2659715 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741985892.707068 2659715 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741985892.707071 2659715 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741985892.707074 2659715 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741985892.707076 2659715 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-14 20:58:12.707081: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.710397 2659715 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741985892.710422 2659715 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741985892.710425 2659715 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741985892.710428 2659715 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741985892.710432 2659715 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741985892.710434 2659715 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741985892.710437 2659715 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741985892.710440 2659715 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741985892.710443 2659715 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741985892.710446 2659715 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-14 20:58:12.710451: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.713787 2659715 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741985892.713802 2659715 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741985892.713805 2659715 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741985892.713810 2659715 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741985892.713813 2659715 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741985892.713816 2659715 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741985892.713819 2659715 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741985892.713822 2659715 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741985892.713825 2659715 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741985892.713828 2659715 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-14 20:58:12.713832: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.717138 2659715 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741985892.717158 2659715 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741985892.717161 2659715 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741985892.717164 2659715 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741985892.717167 2659715 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741985892.717170 2659715 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741985892.717173 2659715 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741985892.717176 2659715 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741985892.717179 2659715 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741985892.717182 2659715 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-14 20:58:12.717187: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.720499 2659715 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741985892.720514 2659715 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741985892.720518 2659715 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741985892.720521 2659715 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741985892.720524 2659715 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741985892.720527 2659715 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741985892.720530 2659715 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741985892.720532 2659715 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741985892.720535 2659715 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741985892.720538 2659715 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-14 20:58:12.720543: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.723974 2659715 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741985892.723991 2659715 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741985892.723994 2659715 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741985892.723997 2659715 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741985892.724000 2659715 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741985892.724003 2659715 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741985892.724006 2659715 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741985892.724010 2659715 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741985892.724013 2659715 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741985892.724016 2659715 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-14 20:58:12.724021: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.727438 2659715 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741985892.727455 2659715 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741985892.727459 2659715 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741985892.727462 2659715 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741985892.727465 2659715 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741985892.727468 2659715 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741985892.727471 2659715 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741985892.727473 2659715 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741985892.727476 2659715 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741985892.727479 2659715 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-14 20:58:12.727484: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.730835 2659715 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741985892.730850 2659715 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741985892.730853 2659715 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741985892.730856 2659715 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741985892.730859 2659715 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741985892.730862 2659715 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741985892.730865 2659715 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741985892.730868 2659715 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741985892.730870 2659715 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741985892.730873 2659715 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-14 20:58:12.730878: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.734216 2659715 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741985892.734232 2659715 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741985892.734235 2659715 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741985892.734238 2659715 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741985892.734241 2659715 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741985892.734244 2659715 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741985892.734247 2659715 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741985892.734250 2659715 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741985892.734253 2659715 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741985892.734256 2659715 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-14 20:58:12.734261: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.737816 2659715 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741985892.737830 2659715 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741985892.737833 2659715 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741985892.737836 2659715 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741985892.737839 2659715 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741985892.737842 2659715 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741985892.737845 2659715 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741985892.737848 2659715 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741985892.737851 2659715 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741985892.737854 2659715 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-14 20:58:12.737859: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.741424 2659715 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741985892.741440 2659715 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741985892.741443 2659715 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741985892.741446 2659715 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741985892.741449 2659715 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741985892.741452 2659715 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741985892.741455 2659715 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741985892.741458 2659715 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741985892.741461 2659715 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741985892.741464 2659715 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-14 20:58:12.741468: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741985892.745003 2659715 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741985892.745020 2659715 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741985892.745023 2659715 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741985892.745026 2659715 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741985892.745029 2659715 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741985892.745032 2659715 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741985892.745035 2659715 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741985892.745038 2659715 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741985892.745041 2659715 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741985892.745044 2659715 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-14 20:58:12.745049: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Test Loss: 1.462901	Test Acc: 69.7000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  JULIA_DEBUG = Literate</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,21)]))}const u=s(c,[["render",i]]);export{d as __pageData,u as default};
