import{_ as s,c as n,o as e,al as p}from"./chunks/framework.D6MqQydi.js";const d=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),c={name:"tutorials/intermediate/6_GCN_Cora.md"};function i(t,a,r,l,f,o){return e(),n("div",null,a[0]||(a[0]=[p(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, Reactant, MLDatasets, Random, Statistics, Enzyme, GNNGraphs, ConcreteStructs,</span></span>
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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-03-08 00:03:49.748298: I external/xla/xla/service/service.cc:152] XLA service 0x775e800 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-08 00:03:49.748431: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1741392229.749088 3994099 se_gpu_pjrt_client.cc:951] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1741392229.749133 3994099 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741392229.749163 3994099 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741392229.763382 3994099 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-14/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-14/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:340</span></span>
<span class="line"><span>2025-03-08 00:05:59.706946: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_29&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.054185: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22_0&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.088669: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 276 bytes spill stores, 276 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.098392: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.153165: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 128 bytes spill stores, 128 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.203676: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 304 bytes spill stores, 304 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.210128: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.291167: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 320 bytes spill stores, 320 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.411105: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 16 bytes spill stores, 16 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.462211: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 1176 bytes spill stores, 1148 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.594944: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 648 bytes spill stores, 652 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:00.898459: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:01.547184: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 336 bytes spill stores, 336 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:01.842776: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:02.106122: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741392362.254179 3994099 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1741392362.254249 3994099 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1741392362.254257 3994099 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1741392362.254264 3994099 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1741392362.254271 3994099 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1741392362.254277 3994099 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1741392362.254286 3994099 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1741392362.254292 3994099 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1741392362.254299 3994099 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1741392362.254305 3994099 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-08 00:06:02.254320: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.257090 3994099 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1741392362.257121 3994099 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1741392362.257129 3994099 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1741392362.257135 3994099 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1741392362.257142 3994099 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1741392362.257149 3994099 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1741392362.257155 3994099 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1741392362.257162 3994099 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1741392362.257168 3994099 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1741392362.257174 3994099 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-08 00:06:02.257185: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.259700 3994099 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1741392362.259731 3994099 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1741392362.259738 3994099 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1741392362.259745 3994099 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1741392362.259751 3994099 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1741392362.259758 3994099 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1741392362.259764 3994099 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1741392362.259771 3994099 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1741392362.259777 3994099 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1741392362.259784 3994099 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-08 00:06:02.259794: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.262298 3994099 buffer_comparator.cc:156] Difference at 32: 0, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1741392362.262325 3994099 buffer_comparator.cc:156] Difference at 33: 0, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1741392362.262332 3994099 buffer_comparator.cc:156] Difference at 34: 0, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1741392362.262339 3994099 buffer_comparator.cc:156] Difference at 35: 0, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1741392362.262345 3994099 buffer_comparator.cc:156] Difference at 36: 0, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1741392362.262352 3994099 buffer_comparator.cc:156] Difference at 37: 0, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1741392362.262358 3994099 buffer_comparator.cc:156] Difference at 38: 0, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1741392362.262365 3994099 buffer_comparator.cc:156] Difference at 39: 0, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1741392362.262371 3994099 buffer_comparator.cc:156] Difference at 40: 0, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1741392362.262378 3994099 buffer_comparator.cc:156] Difference at 41: 0, expected 13.7427</span></span>
<span class="line"><span>2025-03-08 00:06:02.262390: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.264680 3994099 buffer_comparator.cc:156] Difference at 32: 0, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1741392362.264696 3994099 buffer_comparator.cc:156] Difference at 33: 0, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1741392362.264699 3994099 buffer_comparator.cc:156] Difference at 34: 0, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1741392362.264702 3994099 buffer_comparator.cc:156] Difference at 35: 0, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1741392362.264705 3994099 buffer_comparator.cc:156] Difference at 36: 0, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1741392362.264708 3994099 buffer_comparator.cc:156] Difference at 37: 0, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1741392362.264711 3994099 buffer_comparator.cc:156] Difference at 38: 0, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1741392362.264714 3994099 buffer_comparator.cc:156] Difference at 39: 0, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1741392362.264717 3994099 buffer_comparator.cc:156] Difference at 40: 0, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1741392362.264719 3994099 buffer_comparator.cc:156] Difference at 41: 0, expected 13.7427</span></span>
<span class="line"><span>2025-03-08 00:06:02.264724: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.266949 3994099 buffer_comparator.cc:156] Difference at 0: 16.5257, expected 14.4011</span></span>
<span class="line"><span>E0000 00:00:1741392362.266965 3994099 buffer_comparator.cc:156] Difference at 1: 19.4064, expected 15.9904</span></span>
<span class="line"><span>E0000 00:00:1741392362.266968 3994099 buffer_comparator.cc:156] Difference at 2: 16.1909, expected 13.4103</span></span>
<span class="line"><span>E0000 00:00:1741392362.266971 3994099 buffer_comparator.cc:156] Difference at 6: 13.1689, expected 11.4953</span></span>
<span class="line"><span>E0000 00:00:1741392362.266974 3994099 buffer_comparator.cc:156] Difference at 9: 16.2882, expected 14.2452</span></span>
<span class="line"><span>E0000 00:00:1741392362.266977 3994099 buffer_comparator.cc:156] Difference at 11: 15.6385, expected 13.739</span></span>
<span class="line"><span>E0000 00:00:1741392362.266980 3994099 buffer_comparator.cc:156] Difference at 12: 20.6748, expected 16.297</span></span>
<span class="line"><span>E0000 00:00:1741392362.266983 3994099 buffer_comparator.cc:156] Difference at 13: 17.2352, expected 14.372</span></span>
<span class="line"><span>E0000 00:00:1741392362.266986 3994099 buffer_comparator.cc:156] Difference at 14: 14.761, expected 12.4213</span></span>
<span class="line"><span>E0000 00:00:1741392362.266989 3994099 buffer_comparator.cc:156] Difference at 16: 17.262, expected 15.1227</span></span>
<span class="line"><span>2025-03-08 00:06:02.266994: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.269189 3994099 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1741392362.269202 3994099 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1741392362.269206 3994099 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1741392362.269209 3994099 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1741392362.269211 3994099 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1741392362.269214 3994099 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1741392362.269217 3994099 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1741392362.269220 3994099 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1741392362.269223 3994099 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1741392362.269226 3994099 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-08 00:06:02.269231: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.271418 3994099 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1741392362.271431 3994099 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1741392362.271436 3994099 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1741392362.271439 3994099 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1741392362.271442 3994099 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1741392362.271445 3994099 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1741392362.271447 3994099 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1741392362.271450 3994099 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1741392362.271453 3994099 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1741392362.271456 3994099 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-08 00:06:02.271461: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.273651 3994099 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1741392362.273664 3994099 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1741392362.273667 3994099 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1741392362.273670 3994099 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1741392362.273673 3994099 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1741392362.273676 3994099 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1741392362.273679 3994099 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1741392362.273681 3994099 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1741392362.273684 3994099 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1741392362.273687 3994099 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-08 00:06:02.273692: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.275906 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1741392362.275920 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1741392362.275924 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1741392362.275927 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1741392362.275930 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1741392362.275932 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1741392362.275935 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1741392362.275938 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1741392362.275941 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1741392362.275944 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-08 00:06:02.275949: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.278191 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1741392362.278205 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1741392362.278208 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1741392362.278211 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1741392362.278214 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1741392362.278217 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1741392362.278221 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1741392362.278224 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1741392362.278227 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1741392362.278230 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-08 00:06:02.278235: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.280429 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1741392362.280445 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1741392362.280448 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1741392362.280451 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1741392362.280454 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1741392362.280457 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1741392362.280460 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1741392362.280462 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1741392362.280465 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1741392362.280468 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-08 00:06:02.280473: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.282677 3994099 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1741392362.282691 3994099 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1741392362.282694 3994099 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1741392362.282697 3994099 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1741392362.282700 3994099 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1741392362.282703 3994099 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1741392362.282706 3994099 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1741392362.282709 3994099 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1741392362.282712 3994099 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1741392362.282715 3994099 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-08 00:06:02.282719: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.284947 3994099 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1741392362.284962 3994099 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1741392362.284965 3994099 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1741392362.284968 3994099 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1741392362.284971 3994099 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1741392362.284974 3994099 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1741392362.284977 3994099 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1741392362.284980 3994099 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1741392362.284983 3994099 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1741392362.284986 3994099 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-08 00:06:02.284992: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.287188 3994099 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1741392362.287206 3994099 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1741392362.287209 3994099 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1741392362.287212 3994099 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1741392362.287215 3994099 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1741392362.287218 3994099 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1741392362.287221 3994099 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1741392362.287224 3994099 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1741392362.287227 3994099 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1741392362.287229 3994099 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-08 00:06:02.287234: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.289455 3994099 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1741392362.289469 3994099 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1741392362.289472 3994099 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1741392362.289475 3994099 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1741392362.289478 3994099 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1741392362.289481 3994099 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1741392362.289484 3994099 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1741392362.289486 3994099 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1741392362.289489 3994099 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1741392362.289492 3994099 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-08 00:06:02.289497: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.298523 3994099 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1741392362.298555 3994099 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1741392362.298558 3994099 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1741392362.298561 3994099 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1741392362.298564 3994099 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1741392362.298567 3994099 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1741392362.298570 3994099 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1741392362.298573 3994099 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1741392362.298576 3994099 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.298578 3994099 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-03-08 00:06:02.298585: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.304538 3994099 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1741392362.304568 3994099 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1741392362.304572 3994099 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1741392362.304575 3994099 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1741392362.304578 3994099 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1741392362.304581 3994099 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1741392362.304584 3994099 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1741392362.304587 3994099 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1741392362.304590 3994099 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.304593 3994099 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-03-08 00:06:02.304599: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.309664 3994099 buffer_comparator.cc:156] Difference at 64: 0, expected 1106.21</span></span>
<span class="line"><span>E0000 00:00:1741392362.309693 3994099 buffer_comparator.cc:156] Difference at 65: 0, expected 1087.83</span></span>
<span class="line"><span>E0000 00:00:1741392362.309696 3994099 buffer_comparator.cc:156] Difference at 66: 0, expected 1090.54</span></span>
<span class="line"><span>E0000 00:00:1741392362.309699 3994099 buffer_comparator.cc:156] Difference at 67: 0, expected 1104.23</span></span>
<span class="line"><span>E0000 00:00:1741392362.309702 3994099 buffer_comparator.cc:156] Difference at 68: 0, expected 1104.3</span></span>
<span class="line"><span>E0000 00:00:1741392362.309705 3994099 buffer_comparator.cc:156] Difference at 69: 0, expected 1093.45</span></span>
<span class="line"><span>E0000 00:00:1741392362.309708 3994099 buffer_comparator.cc:156] Difference at 70: 0, expected 1091.52</span></span>
<span class="line"><span>E0000 00:00:1741392362.309711 3994099 buffer_comparator.cc:156] Difference at 71: 0, expected 1110.4</span></span>
<span class="line"><span>E0000 00:00:1741392362.309714 3994099 buffer_comparator.cc:156] Difference at 72: 0, expected 1106.92</span></span>
<span class="line"><span>E0000 00:00:1741392362.309717 3994099 buffer_comparator.cc:156] Difference at 73: 0, expected 1088.44</span></span>
<span class="line"><span>2025-03-08 00:06:02.309723: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.314743 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.314781 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.314784 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.314787 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.314790 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.314793 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.314796 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.314799 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.314802 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.314805 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.314810: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.319740 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.319764 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.319767 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.319770 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.319773 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.319776 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.319781 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.319783 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.319786 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.319789 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.319795: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.324453 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.324488 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.324491 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.324494 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.324497 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.324500 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.324503 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.324506 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.324509 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.324512 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.324519: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.329388 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.329421 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.329424 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.329427 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.329430 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.329433 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.329436 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.329439 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.329442 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.329445 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.329452: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.334107 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.334127 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.334130 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.334133 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.334136 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.334139 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.334142 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.334145 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.334148 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.334150 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.334159: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.339162 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.339235 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.339238 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.339241 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.339244 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.339247 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.339250 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.339253 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.339256 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.339259 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.339269: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.344187 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.344240 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.344243 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.344246 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.344249 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.344252 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.344255 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.344258 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.344261 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.344263 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.344273: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.349012 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.349073 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.349076 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.349079 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.349082 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.349085 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.349088 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.349091 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.349094 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.349097 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.349108: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.353936 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.353995 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.354001 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.354004 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.354007 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.354010 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.354013 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.354016 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.354019 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.354021 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.354032: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.358704 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.358735 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.358738 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.358741 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.358744 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.358747 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.358750 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.358753 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.358756 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.358759 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.358765: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.363183 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.363207 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.363210 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.363213 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.363216 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.363219 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.363222 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.363225 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.363228 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.363231 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.363236: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.367556 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.367580 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.367583 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.367586 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.367589 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.367592 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.367597 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.367600 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.367603 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.367606 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.367612: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.372225 3994099 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741392362.372250 3994099 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741392362.372253 3994099 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741392362.372256 3994099 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741392362.372259 3994099 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741392362.372262 3994099 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741392362.372265 3994099 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.372268 3994099 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741392362.372271 3994099 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741392362.372274 3994099 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.372279: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.376741 3994099 buffer_comparator.cc:156] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1741392362.376760 3994099 buffer_comparator.cc:156] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1741392362.376763 3994099 buffer_comparator.cc:156] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1741392362.376766 3994099 buffer_comparator.cc:156] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1741392362.376769 3994099 buffer_comparator.cc:156] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1741392362.376772 3994099 buffer_comparator.cc:156] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1741392362.376775 3994099 buffer_comparator.cc:156] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1741392362.376778 3994099 buffer_comparator.cc:156] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1741392362.376781 3994099 buffer_comparator.cc:156] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1741392362.376784 3994099 buffer_comparator.cc:156] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-03-08 00:06:02.376788: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.388093 3994099 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741392362.388142 3994099 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741392362.388145 3994099 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741392362.388148 3994099 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741392362.388151 3994099 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741392362.388154 3994099 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741392362.388157 3994099 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392362.388160 3994099 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741392362.388164 3994099 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741392362.388168 3994099 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.388178: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.391351 3994099 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741392362.391375 3994099 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741392362.391378 3994099 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741392362.391381 3994099 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741392362.391384 3994099 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741392362.391387 3994099 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741392362.391390 3994099 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392362.391393 3994099 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741392362.391396 3994099 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741392362.391399 3994099 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.391405: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.394576 3994099 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741392362.394591 3994099 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741392362.394594 3994099 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741392362.394597 3994099 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741392362.394600 3994099 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741392362.394603 3994099 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741392362.394606 3994099 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392362.394609 3994099 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741392362.394612 3994099 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741392362.394615 3994099 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.394620: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.397782 3994099 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741392362.397798 3994099 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741392362.397801 3994099 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741392362.397805 3994099 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741392362.397808 3994099 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741392362.397811 3994099 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741392362.397814 3994099 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392362.397817 3994099 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741392362.397820 3994099 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741392362.397823 3994099 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.397828: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.401028 3994099 buffer_comparator.cc:156] Difference at 0: 1139.71, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1741392362.401046 3994099 buffer_comparator.cc:156] Difference at 1: 1404.8, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1741392362.401049 3994099 buffer_comparator.cc:156] Difference at 2: 2132.23, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1741392362.401052 3994099 buffer_comparator.cc:156] Difference at 3: 1838.84, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1741392362.401055 3994099 buffer_comparator.cc:156] Difference at 4: 1307.39, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1741392362.401058 3994099 buffer_comparator.cc:156] Difference at 5: 2064.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1741392362.401061 3994099 buffer_comparator.cc:156] Difference at 6: 1480.82, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1741392362.401064 3994099 buffer_comparator.cc:156] Difference at 7: 1113.19, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1741392362.401068 3994099 buffer_comparator.cc:156] Difference at 8: 1358.65, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.401071 3994099 buffer_comparator.cc:156] Difference at 9: 2048.24, expected 1833.76</span></span>
<span class="line"><span>2025-03-08 00:06:02.401076: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.404327 3994099 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741392362.404344 3994099 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741392362.404348 3994099 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741392362.404351 3994099 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741392362.404354 3994099 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741392362.404357 3994099 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741392362.404360 3994099 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392362.404363 3994099 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741392362.404366 3994099 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741392362.404369 3994099 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-08 00:06:02.404374: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.407586 3994099 buffer_comparator.cc:156] Difference at 224: 1186.14, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1741392362.407604 3994099 buffer_comparator.cc:156] Difference at 225: 1033.68, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741392362.407608 3994099 buffer_comparator.cc:156] Difference at 226: 723.67, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1741392362.407611 3994099 buffer_comparator.cc:156] Difference at 227: 1156.29, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1741392362.407614 3994099 buffer_comparator.cc:156] Difference at 228: 843.86, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741392362.407617 3994099 buffer_comparator.cc:156] Difference at 229: 633.168, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741392362.407620 3994099 buffer_comparator.cc:156] Difference at 230: 810.302, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1741392362.407623 3994099 buffer_comparator.cc:156] Difference at 231: 1218.15, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1741392362.407626 3994099 buffer_comparator.cc:156] Difference at 232: 1064.04, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1741392362.407629 3994099 buffer_comparator.cc:156] Difference at 233: 741.156, expected 1803.13</span></span>
<span class="line"><span>2025-03-08 00:06:02.407634: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.410785 3994099 buffer_comparator.cc:156] Difference at 224: 1186.14, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1741392362.410804 3994099 buffer_comparator.cc:156] Difference at 225: 1033.68, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741392362.410808 3994099 buffer_comparator.cc:156] Difference at 226: 723.67, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1741392362.410811 3994099 buffer_comparator.cc:156] Difference at 227: 1156.29, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1741392362.410814 3994099 buffer_comparator.cc:156] Difference at 228: 843.86, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741392362.410817 3994099 buffer_comparator.cc:156] Difference at 229: 633.168, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741392362.410820 3994099 buffer_comparator.cc:156] Difference at 230: 810.302, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1741392362.410823 3994099 buffer_comparator.cc:156] Difference at 231: 1218.15, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1741392362.410826 3994099 buffer_comparator.cc:156] Difference at 232: 1064.04, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1741392362.410829 3994099 buffer_comparator.cc:156] Difference at 233: 741.156, expected 1803.13</span></span>
<span class="line"><span>2025-03-08 00:06:02.410834: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.413999 3994099 buffer_comparator.cc:156] Difference at 224: 1186.14, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1741392362.414016 3994099 buffer_comparator.cc:156] Difference at 225: 1033.68, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741392362.414020 3994099 buffer_comparator.cc:156] Difference at 226: 723.67, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1741392362.414023 3994099 buffer_comparator.cc:156] Difference at 227: 1156.29, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1741392362.414026 3994099 buffer_comparator.cc:156] Difference at 228: 843.86, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741392362.414029 3994099 buffer_comparator.cc:156] Difference at 229: 633.168, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741392362.414032 3994099 buffer_comparator.cc:156] Difference at 230: 810.302, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1741392362.414035 3994099 buffer_comparator.cc:156] Difference at 231: 1218.15, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1741392362.414038 3994099 buffer_comparator.cc:156] Difference at 232: 1064.04, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1741392362.414041 3994099 buffer_comparator.cc:156] Difference at 233: 741.156, expected 1803.13</span></span>
<span class="line"><span>2025-03-08 00:06:02.414046: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.417285 3994099 buffer_comparator.cc:156] Difference at 448: 1214.22, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741392362.417301 3994099 buffer_comparator.cc:156] Difference at 449: 1056.45, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392362.417304 3994099 buffer_comparator.cc:156] Difference at 450: 736.847, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741392362.417307 3994099 buffer_comparator.cc:156] Difference at 451: 1184.91, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741392362.417310 3994099 buffer_comparator.cc:156] Difference at 452: 859.942, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741392362.417313 3994099 buffer_comparator.cc:156] Difference at 453: 620.77, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741392362.417316 3994099 buffer_comparator.cc:156] Difference at 454: 796.75, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741392362.417319 3994099 buffer_comparator.cc:156] Difference at 455: 1201.02, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741392362.417322 3994099 buffer_comparator.cc:156] Difference at 456: 1045.45, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741392362.417325 3994099 buffer_comparator.cc:156] Difference at 457: 732.834, expected 1821.28</span></span>
<span class="line"><span>2025-03-08 00:06:02.417330: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.420469 3994099 buffer_comparator.cc:156] Difference at 0: 1057.27, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1741392362.420486 3994099 buffer_comparator.cc:156] Difference at 1: 1319.15, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1741392362.420489 3994099 buffer_comparator.cc:156] Difference at 2: 2004.43, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1741392362.420494 3994099 buffer_comparator.cc:156] Difference at 3: 1745.74, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1741392362.420497 3994099 buffer_comparator.cc:156] Difference at 4: 1252.2, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1741392362.420500 3994099 buffer_comparator.cc:156] Difference at 7: 1175.57, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1741392362.420503 3994099 buffer_comparator.cc:156] Difference at 8: 1398.75, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1741392362.420506 3994099 buffer_comparator.cc:156] Difference at 9: 2125.62, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1741392362.420509 3994099 buffer_comparator.cc:156] Difference at 10: 1878.38, expected 1592.37</span></span>
<span class="line"><span>E0000 00:00:1741392362.420512 3994099 buffer_comparator.cc:156] Difference at 11: 1362.67, expected 1121.95</span></span>
<span class="line"><span>2025-03-08 00:06:02.420517: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.423711 3994099 buffer_comparator.cc:156] Difference at 448: 1221.14, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741392362.423730 3994099 buffer_comparator.cc:156] Difference at 449: 1061.5, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392362.423734 3994099 buffer_comparator.cc:156] Difference at 450: 743.315, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741392362.423737 3994099 buffer_comparator.cc:156] Difference at 451: 1192.79, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741392362.423740 3994099 buffer_comparator.cc:156] Difference at 452: 864.899, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741392362.423743 3994099 buffer_comparator.cc:156] Difference at 453: 626.203, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741392362.423746 3994099 buffer_comparator.cc:156] Difference at 454: 803.97, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741392362.423749 3994099 buffer_comparator.cc:156] Difference at 455: 1208.29, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741392362.423752 3994099 buffer_comparator.cc:156] Difference at 456: 1052.01, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741392362.423755 3994099 buffer_comparator.cc:156] Difference at 457: 737.437, expected 1821.28</span></span>
<span class="line"><span>2025-03-08 00:06:02.423760: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.427029 3994099 buffer_comparator.cc:156] Difference at 448: 1221.14, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741392362.427046 3994099 buffer_comparator.cc:156] Difference at 449: 1061.5, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392362.427050 3994099 buffer_comparator.cc:156] Difference at 450: 743.315, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741392362.427053 3994099 buffer_comparator.cc:156] Difference at 451: 1192.79, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741392362.427056 3994099 buffer_comparator.cc:156] Difference at 452: 864.899, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741392362.427059 3994099 buffer_comparator.cc:156] Difference at 453: 626.203, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741392362.427062 3994099 buffer_comparator.cc:156] Difference at 454: 803.97, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741392362.427065 3994099 buffer_comparator.cc:156] Difference at 455: 1208.29, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741392362.427068 3994099 buffer_comparator.cc:156] Difference at 456: 1052.01, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741392362.427071 3994099 buffer_comparator.cc:156] Difference at 457: 737.437, expected 1821.28</span></span>
<span class="line"><span>2025-03-08 00:06:02.427076: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.430433 3994099 buffer_comparator.cc:156] Difference at 448: 1221.14, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741392362.430460 3994099 buffer_comparator.cc:156] Difference at 449: 1061.5, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392362.430464 3994099 buffer_comparator.cc:156] Difference at 450: 743.315, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741392362.430467 3994099 buffer_comparator.cc:156] Difference at 451: 1192.79, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741392362.430470 3994099 buffer_comparator.cc:156] Difference at 452: 864.899, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741392362.430475 3994099 buffer_comparator.cc:156] Difference at 453: 626.203, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741392362.430478 3994099 buffer_comparator.cc:156] Difference at 454: 803.97, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741392362.430481 3994099 buffer_comparator.cc:156] Difference at 455: 1208.29, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741392362.430484 3994099 buffer_comparator.cc:156] Difference at 456: 1052.01, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741392362.430487 3994099 buffer_comparator.cc:156] Difference at 457: 737.437, expected 1821.28</span></span>
<span class="line"><span>2025-03-08 00:06:02.430492: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.433658 3994099 buffer_comparator.cc:156] Difference at 448: 1221.14, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741392362.433675 3994099 buffer_comparator.cc:156] Difference at 449: 1061.5, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392362.433678 3994099 buffer_comparator.cc:156] Difference at 450: 743.315, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741392362.433681 3994099 buffer_comparator.cc:156] Difference at 451: 1192.79, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741392362.433684 3994099 buffer_comparator.cc:156] Difference at 452: 864.899, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741392362.433687 3994099 buffer_comparator.cc:156] Difference at 453: 626.203, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741392362.433690 3994099 buffer_comparator.cc:156] Difference at 454: 803.97, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741392362.433693 3994099 buffer_comparator.cc:156] Difference at 455: 1208.29, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741392362.433696 3994099 buffer_comparator.cc:156] Difference at 456: 1052.01, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741392362.433699 3994099 buffer_comparator.cc:156] Difference at 457: 737.437, expected 1821.28</span></span>
<span class="line"><span>2025-03-08 00:06:02.433704: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.437023 3994099 buffer_comparator.cc:156] Difference at 896: 1204.66, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1741392362.437041 3994099 buffer_comparator.cc:156] Difference at 897: 1053.28, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741392362.437045 3994099 buffer_comparator.cc:156] Difference at 898: 740.998, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1741392362.437048 3994099 buffer_comparator.cc:156] Difference at 899: 1185.71, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1741392362.437051 3994099 buffer_comparator.cc:156] Difference at 900: 850.478, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741392362.437054 3994099 buffer_comparator.cc:156] Difference at 901: 634.712, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1741392362.437057 3994099 buffer_comparator.cc:156] Difference at 902: 799.593, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741392362.437060 3994099 buffer_comparator.cc:156] Difference at 903: 1208.15, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1741392362.437063 3994099 buffer_comparator.cc:156] Difference at 904: 1055.09, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1741392362.437066 3994099 buffer_comparator.cc:156] Difference at 905: 746.267, expected 1817.41</span></span>
<span class="line"><span>2025-03-08 00:06:02.437071: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392362.440384 3994099 buffer_comparator.cc:156] Difference at 896: 1204.66, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1741392362.440402 3994099 buffer_comparator.cc:156] Difference at 897: 1053.28, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741392362.440405 3994099 buffer_comparator.cc:156] Difference at 898: 740.998, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1741392362.440408 3994099 buffer_comparator.cc:156] Difference at 899: 1185.71, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1741392362.440411 3994099 buffer_comparator.cc:156] Difference at 900: 850.478, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741392362.440415 3994099 buffer_comparator.cc:156] Difference at 901: 634.712, expected 1796.71</span></span>
<span class="line"><span>Epoch   1	Train Loss: 14.561998	Train Acc: 23.5714%	Val Loss: 7.762640	Val Acc: 26.8000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 7.368236	Train Acc: 27.1429%	Val Loss: 4.301461	Val Acc: 32.2000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 2.985260	Train Acc: 44.2857%	Val Loss: 1.842087	Val Acc: 39.4000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 1.855432	Train Acc: 52.1429%	Val Loss: 1.850034	Val Acc: 46.0000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 1.644040	Train Acc: 62.1429%	Val Loss: 1.798013	Val Acc: 50.8000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.366259	Train Acc: 72.8571%	Val Loss: 1.688894	Val Acc: 52.2000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.166350	Train Acc: 76.4286%	Val Loss: 1.593970	Val Acc: 55.2000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 0.996649	Train Acc: 80.0000%	Val Loss: 1.548063	Val Acc: 57.2000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 0.875526	Train Acc: 77.1429%	Val Loss: 1.549064	Val Acc: 60.6000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 0.791592	Train Acc: 70.7143%	Val Loss: 1.731398	Val Acc: 55.0000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 0.880753	Train Acc: 76.4286%	Val Loss: 1.619656	Val Acc: 59.4000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 0.730879	Train Acc: 79.2857%	Val Loss: 1.635585	Val Acc: 59.6000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.663583	Train Acc: 79.2857%	Val Loss: 1.662424	Val Acc: 59.8000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 0.645980	Train Acc: 80.7143%	Val Loss: 1.669243	Val Acc: 61.2000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 0.617162	Train Acc: 83.5714%	Val Loss: 1.654438	Val Acc: 62.0000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.579956	Train Acc: 84.2857%	Val Loss: 1.626856	Val Acc: 62.6000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.542971	Train Acc: 83.5714%	Val Loss: 1.594012	Val Acc: 63.6000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.501601	Train Acc: 85.0000%	Val Loss: 1.567793	Val Acc: 65.4000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.472169	Train Acc: 84.2857%	Val Loss: 1.554716	Val Acc: 65.4000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.458515	Train Acc: 84.2857%	Val Loss: 1.552392	Val Acc: 65.8000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.446849	Train Acc: 85.7143%	Val Loss: 1.558038	Val Acc: 66.6000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.434107	Train Acc: 85.7143%	Val Loss: 1.569652	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.415245	Train Acc: 85.7143%	Val Loss: 1.587653	Val Acc: 67.2000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.395874	Train Acc: 85.7143%	Val Loss: 1.611356	Val Acc: 66.8000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.379470	Train Acc: 86.4286%	Val Loss: 1.638929	Val Acc: 65.8000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.365806	Train Acc: 86.4286%	Val Loss: 1.667264	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.354344	Train Acc: 87.1429%	Val Loss: 1.694059	Val Acc: 66.2000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.344957	Train Acc: 86.4286%	Val Loss: 1.718737	Val Acc: 65.8000%</span></span>
<span class="line"><span>Early Stopping at Epoch 28</span></span>
<span class="line"><span>2025-03-08 00:06:45.442172: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:45.508142: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:45.675523: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741392405.683316 3994099 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741392405.683385 3994099 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741392405.683393 3994099 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741392405.683401 3994099 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.683408 3994099 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741392405.683415 3994099 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741392405.683422 3994099 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392405.683429 3994099 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741392405.683436 3994099 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741392405.683442 3994099 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-08 00:06:45.683460: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.687141 3994099 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741392405.687160 3994099 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741392405.687163 3994099 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741392405.687166 3994099 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.687169 3994099 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741392405.687173 3994099 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741392405.687176 3994099 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392405.687179 3994099 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741392405.687182 3994099 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741392405.687185 3994099 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-08 00:06:45.687191: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.690601 3994099 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741392405.690619 3994099 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741392405.690623 3994099 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741392405.690626 3994099 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.690629 3994099 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741392405.690632 3994099 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741392405.690635 3994099 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392405.690638 3994099 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741392405.690643 3994099 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741392405.690646 3994099 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-08 00:06:45.690652: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.694063 3994099 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741392405.694080 3994099 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741392405.694084 3994099 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741392405.694088 3994099 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.694091 3994099 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741392405.694095 3994099 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741392405.694098 3994099 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392405.694101 3994099 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741392405.694104 3994099 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741392405.694107 3994099 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-08 00:06:45.694112: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.697507 3994099 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741392405.697522 3994099 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741392405.697525 3994099 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741392405.697528 3994099 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.697532 3994099 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741392405.697535 3994099 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741392405.697538 3994099 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392405.697541 3994099 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741392405.697544 3994099 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741392405.697547 3994099 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-08 00:06:45.697552: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.700980 3994099 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741392405.700995 3994099 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741392405.700999 3994099 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741392405.701002 3994099 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.701005 3994099 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741392405.701008 3994099 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741392405.701011 3994099 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741392405.701014 3994099 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741392405.701017 3994099 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741392405.701020 3994099 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-08 00:06:45.701027: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.704355 3994099 buffer_comparator.cc:156] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1741392405.704370 3994099 buffer_comparator.cc:156] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741392405.704374 3994099 buffer_comparator.cc:156] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741392405.704377 3994099 buffer_comparator.cc:156] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741392405.704380 3994099 buffer_comparator.cc:156] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741392405.704383 3994099 buffer_comparator.cc:156] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741392405.704386 3994099 buffer_comparator.cc:156] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741392405.704389 3994099 buffer_comparator.cc:156] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1741392405.704392 3994099 buffer_comparator.cc:156] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741392405.704395 3994099 buffer_comparator.cc:156] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-03-08 00:06:45.704400: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.707721 3994099 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741392405.707741 3994099 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741392405.707744 3994099 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741392405.707747 3994099 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741392405.707750 3994099 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741392405.707754 3994099 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741392405.707757 3994099 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741392405.707760 3994099 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1741392405.707763 3994099 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1741392405.707766 3994099 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-08 00:06:45.707771: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.711151 3994099 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741392405.711171 3994099 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741392405.711174 3994099 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741392405.711178 3994099 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741392405.711181 3994099 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741392405.711184 3994099 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741392405.711187 3994099 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741392405.711190 3994099 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1741392405.711193 3994099 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1741392405.711196 3994099 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-08 00:06:45.711201: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.714577 3994099 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392405.714594 3994099 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741392405.714597 3994099 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741392405.714600 3994099 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741392405.714603 3994099 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741392405.714606 3994099 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741392405.714609 3994099 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.714612 3994099 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.714615 3994099 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741392405.714618 3994099 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-08 00:06:45.714624: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.717921 3994099 buffer_comparator.cc:156] Difference at 7: 1058.92, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1741392405.717937 3994099 buffer_comparator.cc:156] Difference at 11: 1263.92, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1741392405.717941 3994099 buffer_comparator.cc:156] Difference at 179: 1223.75, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1741392405.717945 3994099 buffer_comparator.cc:156] Difference at 266: 1047.35, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1741392405.717948 3994099 buffer_comparator.cc:156] Difference at 270: 1246.8, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1741392405.717951 3994099 buffer_comparator.cc:156] Difference at 417: 1222.47, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1741392405.717955 3994099 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392405.717958 3994099 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741392405.717961 3994099 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741392405.717964 3994099 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>2025-03-08 00:06:45.717969: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.721317 3994099 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392405.721344 3994099 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741392405.721347 3994099 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741392405.721350 3994099 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741392405.721353 3994099 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741392405.721356 3994099 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741392405.721359 3994099 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.721362 3994099 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.721365 3994099 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741392405.721368 3994099 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-08 00:06:45.721375: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.724822 3994099 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392405.724851 3994099 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741392405.724854 3994099 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741392405.724858 3994099 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741392405.724861 3994099 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741392405.724864 3994099 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741392405.724867 3994099 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.724870 3994099 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.724873 3994099 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741392405.724876 3994099 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-08 00:06:45.724881: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.728257 3994099 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392405.728278 3994099 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741392405.728281 3994099 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741392405.728284 3994099 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741392405.728287 3994099 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741392405.728290 3994099 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741392405.728293 3994099 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.728296 3994099 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.728299 3994099 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741392405.728302 3994099 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-08 00:06:45.728307: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.731659 3994099 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741392405.731683 3994099 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741392405.731686 3994099 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741392405.731689 3994099 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741392405.731692 3994099 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741392405.731695 3994099 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741392405.731698 3994099 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.731702 3994099 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741392405.731705 3994099 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741392405.731708 3994099 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-08 00:06:45.731713: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.735186 3994099 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741392405.735209 3994099 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741392405.735212 3994099 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741392405.735218 3994099 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741392405.735221 3994099 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741392405.735224 3994099 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741392405.735227 3994099 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741392405.735230 3994099 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741392405.735233 3994099 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741392405.735236 3994099 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-08 00:06:45.735242: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.738723 3994099 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741392405.738757 3994099 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741392405.738760 3994099 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741392405.738763 3994099 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741392405.738767 3994099 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741392405.738770 3994099 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741392405.738773 3994099 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741392405.738776 3994099 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741392405.738779 3994099 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741392405.738782 3994099 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-08 00:06:45.738789: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.742283 3994099 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741392405.742315 3994099 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741392405.742318 3994099 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741392405.742321 3994099 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741392405.742324 3994099 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741392405.742327 3994099 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741392405.742330 3994099 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741392405.742333 3994099 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741392405.742336 3994099 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741392405.742339 3994099 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-08 00:06:45.742346: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.745775 3994099 buffer_comparator.cc:156] Difference at 896: 485.098, expected 958.133</span></span>
<span class="line"><span>E0000 00:00:1741392405.745798 3994099 buffer_comparator.cc:156] Difference at 897: 732.587, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741392405.745801 3994099 buffer_comparator.cc:156] Difference at 898: 635.29, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741392405.745804 3994099 buffer_comparator.cc:156] Difference at 899: 446.948, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741392405.745807 3994099 buffer_comparator.cc:156] Difference at 900: 712.745, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741392405.745812 3994099 buffer_comparator.cc:156] Difference at 901: 516.07, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741392405.745815 3994099 buffer_comparator.cc:156] Difference at 902: 373.095, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741392405.745818 3994099 buffer_comparator.cc:156] Difference at 903: 483.905, expected 941.483</span></span>
<span class="line"><span>E0000 00:00:1741392405.745821 3994099 buffer_comparator.cc:156] Difference at 904: 721.412, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741392405.745824 3994099 buffer_comparator.cc:156] Difference at 905: 633.571, expected 1817.42</span></span>
<span class="line"><span>2025-03-08 00:06:45.745830: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.749445 3994099 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741392405.749474 3994099 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741392405.749478 3994099 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741392405.749481 3994099 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.749484 3994099 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741392405.749487 3994099 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741392405.749490 3994099 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741392405.749493 3994099 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741392405.749496 3994099 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.749499 3994099 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-08 00:06:45.749506: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.753193 3994099 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741392405.753214 3994099 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741392405.753217 3994099 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741392405.753220 3994099 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.753223 3994099 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741392405.753227 3994099 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741392405.753230 3994099 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741392405.753233 3994099 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741392405.753236 3994099 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.753239 3994099 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-08 00:06:45.753244: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392405.756796 3994099 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741392405.756818 3994099 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741392405.756821 3994099 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741392405.756825 3994099 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.756828 3994099 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741392405.756831 3994099 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741392405.756834 3994099 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741392405.756839 3994099 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741392405.756842 3994099 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741392405.756845 3994099 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-08 00:06:45.756850: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-03-08 00:06:47.333191: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:47.445148: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-08 00:06:47.830795: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741392407.836189 3994099 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741392407.836244 3994099 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741392407.836248 3994099 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741392407.836251 3994099 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741392407.836254 3994099 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741392407.836257 3994099 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741392407.836260 3994099 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741392407.836263 3994099 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741392407.836266 3994099 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741392407.836269 3994099 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-08 00:06:47.836278: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.838626 3994099 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741392407.838644 3994099 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741392407.838647 3994099 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741392407.838650 3994099 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741392407.838653 3994099 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741392407.838656 3994099 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741392407.838659 3994099 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741392407.838662 3994099 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741392407.838665 3994099 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741392407.838668 3994099 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-08 00:06:47.838674: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.841008 3994099 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741392407.841025 3994099 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741392407.841028 3994099 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741392407.841031 3994099 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741392407.841036 3994099 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741392407.841039 3994099 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741392407.841042 3994099 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741392407.841045 3994099 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741392407.841048 3994099 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741392407.841051 3994099 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-08 00:06:47.841056: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.843401 3994099 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741392407.843419 3994099 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741392407.843422 3994099 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741392407.843425 3994099 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741392407.843428 3994099 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741392407.843431 3994099 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741392407.843434 3994099 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741392407.843437 3994099 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741392407.843440 3994099 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741392407.843443 3994099 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-08 00:06:47.843448: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.845806 3994099 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741392407.845824 3994099 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741392407.845827 3994099 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741392407.845830 3994099 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741392407.845833 3994099 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741392407.845836 3994099 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741392407.845839 3994099 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741392407.845842 3994099 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741392407.845845 3994099 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741392407.845848 3994099 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-08 00:06:47.845853: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.848217 3994099 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741392407.848232 3994099 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741392407.848235 3994099 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741392407.848238 3994099 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741392407.848241 3994099 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741392407.848244 3994099 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741392407.848247 3994099 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741392407.848250 3994099 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741392407.848255 3994099 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741392407.848258 3994099 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-08 00:06:47.848263: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.850567 3994099 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741392407.850583 3994099 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741392407.850587 3994099 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741392407.850590 3994099 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741392407.850593 3994099 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741392407.850595 3994099 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741392407.850598 3994099 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741392407.850601 3994099 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741392407.850604 3994099 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741392407.850607 3994099 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-08 00:06:47.850612: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.852923 3994099 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741392407.852939 3994099 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741392407.852942 3994099 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741392407.852945 3994099 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741392407.852948 3994099 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741392407.852951 3994099 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741392407.852954 3994099 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741392407.852957 3994099 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741392407.852960 3994099 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741392407.852963 3994099 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-08 00:06:47.852968: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.855299 3994099 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741392407.855319 3994099 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741392407.855323 3994099 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741392407.855326 3994099 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741392407.855329 3994099 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741392407.855331 3994099 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741392407.855334 3994099 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741392407.855337 3994099 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741392407.855340 3994099 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741392407.855343 3994099 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-08 00:06:47.855348: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.857683 3994099 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741392407.857700 3994099 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741392407.857703 3994099 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741392407.857706 3994099 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741392407.857709 3994099 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741392407.857712 3994099 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741392407.857715 3994099 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741392407.857717 3994099 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741392407.857720 3994099 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741392407.857723 3994099 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-08 00:06:47.857728: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.860034 3994099 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741392407.860053 3994099 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741392407.860056 3994099 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741392407.860059 3994099 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741392407.860062 3994099 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741392407.860065 3994099 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741392407.860068 3994099 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741392407.860071 3994099 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741392407.860073 3994099 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741392407.860076 3994099 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-08 00:06:47.860081: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.862404 3994099 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741392407.862423 3994099 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741392407.862426 3994099 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741392407.862429 3994099 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741392407.862432 3994099 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741392407.862435 3994099 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741392407.862438 3994099 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741392407.862441 3994099 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741392407.862444 3994099 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741392407.862446 3994099 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-08 00:06:47.862452: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.864792 3994099 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741392407.864811 3994099 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741392407.864815 3994099 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741392407.864820 3994099 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741392407.864823 3994099 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741392407.864826 3994099 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741392407.864829 3994099 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741392407.864832 3994099 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741392407.864834 3994099 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741392407.864837 3994099 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-08 00:06:47.864843: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.867138 3994099 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741392407.867154 3994099 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741392407.867157 3994099 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741392407.867160 3994099 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741392407.867163 3994099 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741392407.867166 3994099 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741392407.867169 3994099 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741392407.867172 3994099 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741392407.867175 3994099 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741392407.867178 3994099 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-08 00:06:47.867183: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.869486 3994099 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741392407.869505 3994099 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741392407.869508 3994099 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741392407.869511 3994099 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741392407.869514 3994099 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741392407.869517 3994099 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741392407.869520 3994099 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741392407.869523 3994099 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741392407.869526 3994099 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741392407.869529 3994099 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-08 00:06:47.869534: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.871975 3994099 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741392407.871999 3994099 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741392407.872002 3994099 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741392407.872005 3994099 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741392407.872008 3994099 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741392407.872011 3994099 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741392407.872014 3994099 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741392407.872019 3994099 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741392407.872022 3994099 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741392407.872024 3994099 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-08 00:06:47.872030: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.874453 3994099 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741392407.874478 3994099 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741392407.874481 3994099 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741392407.874484 3994099 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741392407.874487 3994099 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741392407.874490 3994099 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741392407.874493 3994099 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741392407.874496 3994099 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741392407.874499 3994099 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741392407.874502 3994099 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-08 00:06:47.874507: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.876840 3994099 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741392407.876855 3994099 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741392407.876858 3994099 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741392407.876861 3994099 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741392407.876864 3994099 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741392407.876867 3994099 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741392407.876870 3994099 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741392407.876873 3994099 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741392407.876876 3994099 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741392407.876879 3994099 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-08 00:06:47.876883: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.879195 3994099 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741392407.879210 3994099 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741392407.879213 3994099 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741392407.879216 3994099 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741392407.879219 3994099 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741392407.879222 3994099 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741392407.879225 3994099 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741392407.879228 3994099 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741392407.879231 3994099 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741392407.879233 3994099 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-08 00:06:47.879238: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.881690 3994099 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741392407.881705 3994099 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741392407.881708 3994099 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741392407.881711 3994099 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741392407.881714 3994099 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741392407.881717 3994099 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741392407.881720 3994099 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741392407.881723 3994099 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741392407.881726 3994099 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741392407.881729 3994099 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-08 00:06:47.881734: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.884182 3994099 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741392407.884197 3994099 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741392407.884200 3994099 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741392407.884203 3994099 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741392407.884206 3994099 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741392407.884209 3994099 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741392407.884212 3994099 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741392407.884215 3994099 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741392407.884218 3994099 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741392407.884221 3994099 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-08 00:06:47.884226: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741392407.886632 3994099 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741392407.886648 3994099 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741392407.886651 3994099 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741392407.886654 3994099 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741392407.886657 3994099 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741392407.886660 3994099 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741392407.886663 3994099 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741392407.886666 3994099 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741392407.886669 3994099 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741392407.886672 3994099 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-08 00:06:47.886677: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Test Loss: 1.535548	Test Acc: 68.4000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.3</span></span>
<span class="line"><span>Commit d63adeda50d (2025-01-21 19:42 UTC)</span></span>
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
