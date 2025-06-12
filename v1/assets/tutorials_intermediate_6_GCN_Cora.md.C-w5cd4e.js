import{_ as s,c as n,o as e,al as p}from"./chunks/framework.CJJb5tWv.js";const d=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),c={name:"tutorials/intermediate/6_GCN_Cora.md"};function i(t,a,r,l,f,o){return e(),n("div",null,a[0]||(a[0]=[p(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux,</span></span>
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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-05-24 12:53:20.407641: I external/xla/xla/service/service.cc:152] XLA service 0x3392c1c0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-05-24 12:53:20.407770: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1748091200.408461 3462857 se_gpu_pjrt_client.cc:1026] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1748091200.408538 3462857 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1748091200.408567 3462857 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1748091200.423272 3462857 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>2025-05-24 12:54:30.131012: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:30.131838: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 32 bytes spill stores, 32 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:30.278721: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 292 bytes spill stores, 292 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:30.406613: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 284 bytes spill stores, 284 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:30.543706: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 272 bytes spill stores, 272 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:30.929365: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 996 bytes spill stores, 968 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:31.022186: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 1212 bytes spill stores, 976 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:31.100032: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 980 bytes spill stores, 976 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:31.143690: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:31.228898: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 604 bytes spill stores, 608 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:31.286865: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 360 bytes spill stores, 356 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:31.818803: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 48 bytes spill stores, 48 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:33.556142: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:33.862603: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:34.290679: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 48 bytes spill stores, 48 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:34.777444: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:35.047181: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:35.256581: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:35.539836: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 104 bytes spill stores, 104 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:54:36.051287: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1748091276.180168 3462857 buffer_comparator.cc:145] Difference at 112: 513.993, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.180208 3462857 buffer_comparator.cc:145] Difference at 113: 357.807, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1748091276.180212 3462857 buffer_comparator.cc:145] Difference at 114: 585.471, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1748091276.180216 3462857 buffer_comparator.cc:145] Difference at 115: 420.444, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1748091276.180219 3462857 buffer_comparator.cc:145] Difference at 116: 302.398, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1748091276.180222 3462857 buffer_comparator.cc:145] Difference at 117: 386.144, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1748091276.180225 3462857 buffer_comparator.cc:145] Difference at 118: 587.071, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748091276.180228 3462857 buffer_comparator.cc:145] Difference at 119: 508.94, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1748091276.180231 3462857 buffer_comparator.cc:145] Difference at 120: 358.918, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1748091276.180234 3462857 buffer_comparator.cc:145] Difference at 121: 578.094, expected 1820.15</span></span>
<span class="line"><span>2025-05-24 12:54:36.180244: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.183656 3462857 buffer_comparator.cc:145] Difference at 112: 513.993, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.183673 3462857 buffer_comparator.cc:145] Difference at 113: 357.807, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1748091276.183676 3462857 buffer_comparator.cc:145] Difference at 114: 585.471, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1748091276.183679 3462857 buffer_comparator.cc:145] Difference at 115: 420.444, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1748091276.183682 3462857 buffer_comparator.cc:145] Difference at 116: 302.398, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1748091276.183686 3462857 buffer_comparator.cc:145] Difference at 117: 386.144, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1748091276.183689 3462857 buffer_comparator.cc:145] Difference at 118: 587.071, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748091276.183692 3462857 buffer_comparator.cc:145] Difference at 119: 508.94, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1748091276.183695 3462857 buffer_comparator.cc:145] Difference at 120: 358.918, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1748091276.183698 3462857 buffer_comparator.cc:145] Difference at 121: 578.094, expected 1820.15</span></span>
<span class="line"><span>2025-05-24 12:54:36.183703: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.186877 3462857 buffer_comparator.cc:145] Difference at 112: 513.993, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.186892 3462857 buffer_comparator.cc:145] Difference at 113: 357.807, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1748091276.186895 3462857 buffer_comparator.cc:145] Difference at 114: 585.471, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1748091276.186898 3462857 buffer_comparator.cc:145] Difference at 115: 420.444, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1748091276.186901 3462857 buffer_comparator.cc:145] Difference at 116: 302.398, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1748091276.186904 3462857 buffer_comparator.cc:145] Difference at 117: 386.144, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1748091276.186907 3462857 buffer_comparator.cc:145] Difference at 118: 587.071, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748091276.186910 3462857 buffer_comparator.cc:145] Difference at 119: 508.94, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1748091276.186914 3462857 buffer_comparator.cc:145] Difference at 120: 358.918, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1748091276.186917 3462857 buffer_comparator.cc:145] Difference at 121: 578.094, expected 1820.15</span></span>
<span class="line"><span>2025-05-24 12:54:36.186922: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.190030 3462857 buffer_comparator.cc:145] Difference at 0: 1084.56, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1748091276.190044 3462857 buffer_comparator.cc:145] Difference at 1: 1350.61, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1748091276.190047 3462857 buffer_comparator.cc:145] Difference at 2: 2009.8, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1748091276.190050 3462857 buffer_comparator.cc:145] Difference at 3: 1768.04, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1748091276.190053 3462857 buffer_comparator.cc:145] Difference at 4: 1240.61, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.190056 3462857 buffer_comparator.cc:145] Difference at 6: 1407.03, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091276.190059 3462857 buffer_comparator.cc:145] Difference at 7: 1138.83, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.190062 3462857 buffer_comparator.cc:145] Difference at 8: 1417.44, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091276.190065 3462857 buffer_comparator.cc:145] Difference at 9: 2084.44, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1748091276.190068 3462857 buffer_comparator.cc:145] Difference at 10: 1844.73, expected 1592.37</span></span>
<span class="line"><span>2025-05-24 12:54:36.190073: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.193247 3462857 buffer_comparator.cc:145] Difference at 0: 1084.56, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1748091276.193264 3462857 buffer_comparator.cc:145] Difference at 1: 1350.61, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1748091276.193267 3462857 buffer_comparator.cc:145] Difference at 2: 2009.8, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1748091276.193270 3462857 buffer_comparator.cc:145] Difference at 3: 1768.04, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1748091276.193273 3462857 buffer_comparator.cc:145] Difference at 4: 1240.61, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.193276 3462857 buffer_comparator.cc:145] Difference at 6: 1407.03, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091276.193279 3462857 buffer_comparator.cc:145] Difference at 7: 1138.83, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.193282 3462857 buffer_comparator.cc:145] Difference at 8: 1417.44, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091276.193285 3462857 buffer_comparator.cc:145] Difference at 9: 2084.44, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1748091276.193288 3462857 buffer_comparator.cc:145] Difference at 10: 1844.73, expected 1592.37</span></span>
<span class="line"><span>2025-05-24 12:54:36.193293: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.196415 3462857 buffer_comparator.cc:145] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1748091276.196429 3462857 buffer_comparator.cc:145] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1748091276.196432 3462857 buffer_comparator.cc:145] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1748091276.196435 3462857 buffer_comparator.cc:145] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1748091276.196439 3462857 buffer_comparator.cc:145] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1748091276.196442 3462857 buffer_comparator.cc:145] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1748091276.196445 3462857 buffer_comparator.cc:145] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748091276.196448 3462857 buffer_comparator.cc:145] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1748091276.196451 3462857 buffer_comparator.cc:145] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1748091276.196454 3462857 buffer_comparator.cc:145] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-05-24 12:54:36.196459: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.199614 3462857 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.199632 3462857 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.199636 3462857 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.8</span></span>
<span class="line"><span>E0000 00:00:1748091276.199640 3462857 buffer_comparator.cc:145] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1748091276.199643 3462857 buffer_comparator.cc:145] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1748091276.199646 3462857 buffer_comparator.cc:145] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1748091276.199649 3462857 buffer_comparator.cc:145] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1748091276.199652 3462857 buffer_comparator.cc:145] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1748091276.199655 3462857 buffer_comparator.cc:145] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1748091276.199658 3462857 buffer_comparator.cc:145] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>2025-05-24 12:54:36.199662: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.202738 3462857 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1748091276.202751 3462857 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1748091276.202754 3462857 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1748091276.202757 3462857 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1748091276.202760 3462857 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.202763 3462857 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.202766 3462857 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091276.202769 3462857 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.202772 3462857 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091276.202775 3462857 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.76</span></span>
<span class="line"><span>2025-05-24 12:54:36.202780: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.205994 3462857 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1748091276.206008 3462857 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091276.206011 3462857 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.206014 3462857 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1748091276.206017 3462857 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091276.206020 3462857 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1748091276.206023 3462857 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1748091276.206026 3462857 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1748091276.206029 3462857 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091276.206032 3462857 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-24 12:54:36.206037: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.209343 3462857 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1748091276.209357 3462857 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091276.209360 3462857 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.209363 3462857 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1748091276.209366 3462857 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091276.209369 3462857 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1748091276.209372 3462857 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1748091276.209375 3462857 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1748091276.209378 3462857 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091276.209381 3462857 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-24 12:54:36.209386: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.212501 3462857 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1748091276.212517 3462857 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091276.212521 3462857 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.212523 3462857 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1748091276.212527 3462857 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091276.212530 3462857 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1748091276.212533 3462857 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1748091276.212536 3462857 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1748091276.212538 3462857 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091276.212541 3462857 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-24 12:54:36.212546: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.215652 3462857 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1748091276.215666 3462857 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091276.215669 3462857 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.215672 3462857 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1748091276.215675 3462857 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091276.215678 3462857 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1748091276.215681 3462857 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1748091276.215684 3462857 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1748091276.215687 3462857 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091276.215690 3462857 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-24 12:54:36.215695: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.218811 3462857 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1748091276.218825 3462857 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091276.218828 3462857 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.218831 3462857 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1748091276.218834 3462857 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091276.218837 3462857 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1748091276.218840 3462857 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1748091276.218843 3462857 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1748091276.218846 3462857 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091276.218849 3462857 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-24 12:54:36.218854: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.221968 3462857 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1748091276.221982 3462857 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091276.221985 3462857 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.221988 3462857 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1748091276.221991 3462857 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091276.221994 3462857 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1748091276.221997 3462857 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1748091276.222000 3462857 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1748091276.222003 3462857 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091276.222006 3462857 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-24 12:54:36.222011: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.225127 3462857 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1748091276.225142 3462857 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091276.225145 3462857 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.225148 3462857 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1748091276.225151 3462857 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091276.225154 3462857 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1748091276.225157 3462857 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1748091276.225160 3462857 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1748091276.225163 3462857 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091276.225166 3462857 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-24 12:54:36.225171: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.228271 3462857 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1748091276.228287 3462857 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091276.228290 3462857 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.228294 3462857 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1748091276.228297 3462857 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091276.228300 3462857 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1748091276.228303 3462857 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1748091276.228306 3462857 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1748091276.228309 3462857 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091276.228311 3462857 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-24 12:54:36.228316: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.231408 3462857 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1748091276.231422 3462857 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1748091276.231425 3462857 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1748091276.231428 3462857 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1748091276.231431 3462857 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.231434 3462857 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.231437 3462857 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091276.231440 3462857 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.231443 3462857 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091276.231447 3462857 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.76</span></span>
<span class="line"><span>2025-05-24 12:54:36.231451: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.234566 3462857 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1748091276.234579 3462857 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1748091276.234583 3462857 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1748091276.234586 3462857 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1748091276.234589 3462857 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.234592 3462857 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.234595 3462857 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091276.234598 3462857 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.234601 3462857 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091276.234604 3462857 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.76</span></span>
<span class="line"><span>2025-05-24 12:54:36.234608: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.237672 3462857 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1748091276.237689 3462857 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1748091276.237692 3462857 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1748091276.237695 3462857 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1748091276.237698 3462857 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.237702 3462857 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.237705 3462857 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091276.237708 3462857 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.237711 3462857 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091276.237714 3462857 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.76</span></span>
<span class="line"><span>2025-05-24 12:54:36.237718: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.240849 3462857 buffer_comparator.cc:145] Difference at 0: 1506.95, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1748091276.240862 3462857 buffer_comparator.cc:145] Difference at 1: 1786.05, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1748091276.240866 3462857 buffer_comparator.cc:145] Difference at 2: 2699.73, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1748091276.240869 3462857 buffer_comparator.cc:145] Difference at 3: 2437.04, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1748091276.240872 3462857 buffer_comparator.cc:145] Difference at 4: 1660.22, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.240875 3462857 buffer_comparator.cc:145] Difference at 5: 2636.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.240878 3462857 buffer_comparator.cc:145] Difference at 6: 1898.51, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091276.240880 3462857 buffer_comparator.cc:145] Difference at 7: 1482.97, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.240884 3462857 buffer_comparator.cc:145] Difference at 8: 1801.1, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091276.240887 3462857 buffer_comparator.cc:145] Difference at 9: 2662.69, expected 1833.76</span></span>
<span class="line"><span>2025-05-24 12:54:36.240891: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.244065 3462857 buffer_comparator.cc:145] Difference at 0: 1506.95, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1748091276.244079 3462857 buffer_comparator.cc:145] Difference at 1: 1786.05, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1748091276.244083 3462857 buffer_comparator.cc:145] Difference at 2: 2699.73, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1748091276.244086 3462857 buffer_comparator.cc:145] Difference at 3: 2437.04, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1748091276.244089 3462857 buffer_comparator.cc:145] Difference at 4: 1660.22, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.244092 3462857 buffer_comparator.cc:145] Difference at 5: 2636.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.244094 3462857 buffer_comparator.cc:145] Difference at 6: 1898.51, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091276.244097 3462857 buffer_comparator.cc:145] Difference at 7: 1482.97, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.244100 3462857 buffer_comparator.cc:145] Difference at 8: 1801.1, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091276.244103 3462857 buffer_comparator.cc:145] Difference at 9: 2662.69, expected 1833.76</span></span>
<span class="line"><span>2025-05-24 12:54:36.244108: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.247298 3462857 buffer_comparator.cc:145] Difference at 0: 1506.95, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1748091276.247314 3462857 buffer_comparator.cc:145] Difference at 1: 1786.05, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1748091276.247317 3462857 buffer_comparator.cc:145] Difference at 2: 2699.73, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1748091276.247320 3462857 buffer_comparator.cc:145] Difference at 3: 2437.04, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1748091276.247323 3462857 buffer_comparator.cc:145] Difference at 4: 1660.22, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.247326 3462857 buffer_comparator.cc:145] Difference at 5: 2636.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.247329 3462857 buffer_comparator.cc:145] Difference at 6: 1898.51, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091276.247332 3462857 buffer_comparator.cc:145] Difference at 7: 1482.97, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.247336 3462857 buffer_comparator.cc:145] Difference at 8: 1801.1, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091276.247339 3462857 buffer_comparator.cc:145] Difference at 9: 2662.69, expected 1833.76</span></span>
<span class="line"><span>2025-05-24 12:54:36.247344: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.250571 3462857 buffer_comparator.cc:145] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1748091276.250585 3462857 buffer_comparator.cc:145] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1748091276.250588 3462857 buffer_comparator.cc:145] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.250592 3462857 buffer_comparator.cc:145] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1748091276.250595 3462857 buffer_comparator.cc:145] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1748091276.250598 3462857 buffer_comparator.cc:145] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1748091276.250601 3462857 buffer_comparator.cc:145] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1748091276.250604 3462857 buffer_comparator.cc:145] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1748091276.250607 3462857 buffer_comparator.cc:145] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1748091276.250610 3462857 buffer_comparator.cc:145] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-05-24 12:54:36.250614: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.253723 3462857 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.253739 3462857 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.253744 3462857 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.8</span></span>
<span class="line"><span>E0000 00:00:1748091276.253747 3462857 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.413</span></span>
<span class="line"><span>E0000 00:00:1748091276.253750 3462857 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.54</span></span>
<span class="line"><span>E0000 00:00:1748091276.253754 3462857 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1748091276.253758 3462857 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1748091276.253761 3462857 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.76</span></span>
<span class="line"><span>E0000 00:00:1748091276.253764 3462857 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.08</span></span>
<span class="line"><span>E0000 00:00:1748091276.253768 3462857 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.61</span></span>
<span class="line"><span>2025-05-24 12:54:36.253772: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.256864 3462857 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.256878 3462857 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.256882 3462857 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.8</span></span>
<span class="line"><span>E0000 00:00:1748091276.256886 3462857 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.413</span></span>
<span class="line"><span>E0000 00:00:1748091276.256889 3462857 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.54</span></span>
<span class="line"><span>E0000 00:00:1748091276.256892 3462857 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1748091276.256896 3462857 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1748091276.256899 3462857 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.76</span></span>
<span class="line"><span>E0000 00:00:1748091276.256902 3462857 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.08</span></span>
<span class="line"><span>E0000 00:00:1748091276.256906 3462857 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.61</span></span>
<span class="line"><span>2025-05-24 12:54:36.256911: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.260125 3462857 buffer_comparator.cc:145] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1748091276.260139 3462857 buffer_comparator.cc:145] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1748091276.260142 3462857 buffer_comparator.cc:145] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.260145 3462857 buffer_comparator.cc:145] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1748091276.260148 3462857 buffer_comparator.cc:145] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1748091276.260151 3462857 buffer_comparator.cc:145] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1748091276.260154 3462857 buffer_comparator.cc:145] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1748091276.260157 3462857 buffer_comparator.cc:145] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1748091276.260160 3462857 buffer_comparator.cc:145] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1748091276.260163 3462857 buffer_comparator.cc:145] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-05-24 12:54:36.260168: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.263251 3462857 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1748091276.263267 3462857 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1748091276.263270 3462857 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1748091276.263273 3462857 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1748091276.263276 3462857 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091276.263279 3462857 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1748091276.263282 3462857 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091276.263285 3462857 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1748091276.263288 3462857 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091276.263291 3462857 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.76</span></span>
<span class="line"><span>2025-05-24 12:54:36.263296: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.267192 3462857 buffer_comparator.cc:145] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1748091276.267206 3462857 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1748091276.267209 3462857 buffer_comparator.cc:145] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1748091276.267212 3462857 buffer_comparator.cc:145] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1748091276.267215 3462857 buffer_comparator.cc:145] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1748091276.267217 3462857 buffer_comparator.cc:145] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1748091276.267220 3462857 buffer_comparator.cc:145] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1748091276.267223 3462857 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1748091276.267226 3462857 buffer_comparator.cc:145] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1748091276.267228 3462857 buffer_comparator.cc:145] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-05-24 12:54:36.267233: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.269699 3462857 buffer_comparator.cc:145] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1748091276.269713 3462857 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1748091276.269717 3462857 buffer_comparator.cc:145] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1748091276.269719 3462857 buffer_comparator.cc:145] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1748091276.269722 3462857 buffer_comparator.cc:145] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1748091276.269725 3462857 buffer_comparator.cc:145] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1748091276.269727 3462857 buffer_comparator.cc:145] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1748091276.269730 3462857 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1748091276.269733 3462857 buffer_comparator.cc:145] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1748091276.269736 3462857 buffer_comparator.cc:145] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-05-24 12:54:36.269740: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.272125 3462857 buffer_comparator.cc:145] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1748091276.272139 3462857 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1748091276.272142 3462857 buffer_comparator.cc:145] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1748091276.272144 3462857 buffer_comparator.cc:145] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1748091276.272147 3462857 buffer_comparator.cc:145] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1748091276.272150 3462857 buffer_comparator.cc:145] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1748091276.272153 3462857 buffer_comparator.cc:145] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1748091276.272155 3462857 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1748091276.272158 3462857 buffer_comparator.cc:145] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1748091276.272161 3462857 buffer_comparator.cc:145] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-05-24 12:54:36.272165: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.274433 3462857 buffer_comparator.cc:145] Difference at 16: -nan, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1748091276.274451 3462857 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1748091276.274455 3462857 buffer_comparator.cc:145] Difference at 18: -nan, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1748091276.274458 3462857 buffer_comparator.cc:145] Difference at 19: -nan, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1748091276.274462 3462857 buffer_comparator.cc:145] Difference at 20: -nan, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1748091276.274464 3462857 buffer_comparator.cc:145] Difference at 21: -nan, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1748091276.274467 3462857 buffer_comparator.cc:145] Difference at 22: -nan, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1748091276.274470 3462857 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1748091276.274473 3462857 buffer_comparator.cc:145] Difference at 24: -nan, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1748091276.274475 3462857 buffer_comparator.cc:145] Difference at 25: -nan, expected 18.5767</span></span>
<span class="line"><span>2025-05-24 12:54:36.274480: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.276736 3462857 buffer_comparator.cc:145] Difference at 32: -nan, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1748091276.276750 3462857 buffer_comparator.cc:145] Difference at 33: -nan, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1748091276.276753 3462857 buffer_comparator.cc:145] Difference at 34: -nan, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1748091276.276756 3462857 buffer_comparator.cc:145] Difference at 35: -nan, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1748091276.276759 3462857 buffer_comparator.cc:145] Difference at 36: -nan, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1748091276.276762 3462857 buffer_comparator.cc:145] Difference at 37: -nan, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1748091276.276765 3462857 buffer_comparator.cc:145] Difference at 38: -nan, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1748091276.276768 3462857 buffer_comparator.cc:145] Difference at 39: -nan, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1748091276.276770 3462857 buffer_comparator.cc:145] Difference at 40: -nan, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1748091276.276773 3462857 buffer_comparator.cc:145] Difference at 41: -nan, expected 20.3484</span></span>
<span class="line"><span>2025-05-24 12:54:36.276777: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.279041 3462857 buffer_comparator.cc:145] Difference at 32: -nan, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1748091276.279055 3462857 buffer_comparator.cc:145] Difference at 33: -nan, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1748091276.279058 3462857 buffer_comparator.cc:145] Difference at 34: -nan, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1748091276.279061 3462857 buffer_comparator.cc:145] Difference at 35: -nan, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1748091276.279064 3462857 buffer_comparator.cc:145] Difference at 36: -nan, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1748091276.279067 3462857 buffer_comparator.cc:145] Difference at 37: -nan, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1748091276.279069 3462857 buffer_comparator.cc:145] Difference at 38: -nan, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1748091276.279072 3462857 buffer_comparator.cc:145] Difference at 39: -nan, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1748091276.279075 3462857 buffer_comparator.cc:145] Difference at 40: -nan, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1748091276.279077 3462857 buffer_comparator.cc:145] Difference at 41: -nan, expected 20.3484</span></span>
<span class="line"><span>2025-05-24 12:54:36.279082: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.281382 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.281397 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.281400 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.281403 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.281406 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.281408 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.281411 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.281414 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.281417 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.281419 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.281424: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.283681 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.283698 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.283701 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.283704 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.283707 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.283710 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.283712 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.283716 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.283718 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.283721 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.283726: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.285994 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.286008 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.286011 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.286014 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.286017 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.286019 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.286022 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.286025 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.286028 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.286030 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.286035: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.288298 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.288312 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.288315 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.288318 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.288321 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.288324 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.288326 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.288329 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.288332 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.288334 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.288339: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.290606 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.290621 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.290624 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.290631 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.290634 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.290637 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.290639 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.290642 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.290645 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.290648 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.290653: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.292937 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.292953 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.292956 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.292959 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.292962 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.292964 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.292967 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.292970 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.292973 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.292975 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.292980: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.295254 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.295267 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.295270 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.295273 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.295276 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.295279 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.295281 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.295284 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.295287 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.295290 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.295294: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.297570 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.297583 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.297586 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.297589 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.297592 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.297595 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.297597 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.297600 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.297603 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.297606 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.297610: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.299879 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.299896 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.299899 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.299902 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.299905 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.299908 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.299910 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.299913 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.299916 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.299918 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.299923: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.302179 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.302195 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.302198 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.302201 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.302204 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.302207 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.302209 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.302212 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.302215 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.302217 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.302222: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.304500 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.304514 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.304517 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.304520 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.304522 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.304525 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.304528 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.304531 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.304533 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.304536 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.304540: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.306821 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.307034 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.307037 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.307039 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.307043 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.307046 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.307048 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.307051 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.307054 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.307056 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.307061: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.309423 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.309437 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.309440 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.309443 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.309446 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.309448 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.309451 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.309454 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.309456 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.309459 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.309464: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.311813 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.311828 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.311831 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.311834 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.311837 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.311839 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748091276.311842 3462857 buffer_comparator.cc:145] Difference at 70: -nan, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748091276.311845 3462857 buffer_comparator.cc:145] Difference at 71: -nan, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748091276.311847 3462857 buffer_comparator.cc:145] Difference at 72: -nan, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748091276.311850 3462857 buffer_comparator.cc:145] Difference at 73: -nan, expected 17.8359</span></span>
<span class="line"><span>2025-05-24 12:54:36.311855: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091276.314219 3462857 buffer_comparator.cc:145] Difference at 64: -nan, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748091276.314235 3462857 buffer_comparator.cc:145] Difference at 65: -nan, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748091276.314238 3462857 buffer_comparator.cc:145] Difference at 66: -nan, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748091276.314241 3462857 buffer_comparator.cc:145] Difference at 67: -nan, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748091276.314243 3462857 buffer_comparator.cc:145] Difference at 68: -nan, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748091276.314246 3462857 buffer_comparator.cc:145] Difference at 69: -nan, expected 19.1597</span></span>
<span class="line"><span>Epoch   1	Train Loss: 16.164062	Train Acc: 14.2857%	Val Loss: 13.462575	Val Acc: 6.6000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 15.994289	Train Acc: 14.2857%	Val Loss: 14.477214	Val Acc: 8.2000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 16.341110	Train Acc: 12.8571%	Val Loss: 15.896417	Val Acc: 10.4000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 19.424477	Train Acc: 12.1429%	Val Loss: 17.228157	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 15.998941	Train Acc: 13.5714%	Val Loss: 18.332340	Val Acc: 12.0000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 19.519047	Train Acc: 12.8571%	Val Loss: 19.271805	Val Acc: 12.4000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 25.523829	Train Acc: 11.4286%	Val Loss: 20.074450	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 24.791389	Train Acc: 11.4286%	Val Loss: 20.636673	Val Acc: 10.8000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 26.397612	Train Acc: 11.4286%	Val Loss: 21.064434	Val Acc: 11.0000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 28.971565	Train Acc: 12.8571%	Val Loss: 21.507832	Val Acc: 10.4000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 27.521463	Train Acc: 10.7143%	Val Loss: 22.238356	Val Acc: 11.8000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 28.513298	Train Acc: 8.5714%	Val Loss: 25.221201	Val Acc: 11.8000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 34.258091	Train Acc: 12.1429%	Val Loss: 29.793264	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 40.258194	Train Acc: 14.2857%	Val Loss: 34.941338	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 48.994648	Train Acc: 14.2857%	Val Loss: 40.441193	Val Acc: 11.0000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 48.451763	Train Acc: 14.2857%	Val Loss: 46.297611	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 56.309483	Train Acc: 14.2857%	Val Loss: 51.959545	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 61.755219	Train Acc: 14.2857%	Val Loss: 57.675751	Val Acc: 11.0000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 69.987755	Train Acc: 14.2857%	Val Loss: 63.371517	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 78.888634	Train Acc: 14.2857%	Val Loss: 68.976616	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 64.435921	Train Acc: 14.2857%	Val Loss: 74.517609	Val Acc: 11.4000%</span></span>
<span class="line"><span>Early Stopping at Epoch 21</span></span>
<span class="line"><span>2025-05-24 12:55:50.178280: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:55:50.705840: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:55:50.947236: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:55:51.119186: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1748091351.126013 3462857 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748091351.126058 3462857 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748091351.126067 3462857 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748091351.126071 3462857 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.126074 3462857 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748091351.126078 3462857 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748091351.126081 3462857 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748091351.126085 3462857 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748091351.126088 3462857 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748091351.126091 3462857 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-24 12:55:51.126103: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.129464 3462857 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748091351.129476 3462857 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748091351.129480 3462857 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748091351.129483 3462857 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.129486 3462857 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748091351.129489 3462857 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748091351.129492 3462857 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748091351.129495 3462857 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748091351.129498 3462857 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748091351.129501 3462857 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-24 12:55:51.129507: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.132863 3462857 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748091351.132875 3462857 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748091351.132879 3462857 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748091351.132882 3462857 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.132885 3462857 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748091351.132888 3462857 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748091351.132893 3462857 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748091351.132897 3462857 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748091351.132900 3462857 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748091351.132903 3462857 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-24 12:55:51.132908: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.136165 3462857 buffer_comparator.cc:145] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748091351.136177 3462857 buffer_comparator.cc:145] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748091351.136181 3462857 buffer_comparator.cc:145] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748091351.136184 3462857 buffer_comparator.cc:145] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.136187 3462857 buffer_comparator.cc:145] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748091351.136190 3462857 buffer_comparator.cc:145] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748091351.136193 3462857 buffer_comparator.cc:145] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748091351.136197 3462857 buffer_comparator.cc:145] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748091351.136200 3462857 buffer_comparator.cc:145] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748091351.136203 3462857 buffer_comparator.cc:145] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-05-24 12:55:51.136207: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.139552 3462857 buffer_comparator.cc:145] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748091351.139564 3462857 buffer_comparator.cc:145] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748091351.139567 3462857 buffer_comparator.cc:145] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748091351.139571 3462857 buffer_comparator.cc:145] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.139574 3462857 buffer_comparator.cc:145] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748091351.139577 3462857 buffer_comparator.cc:145] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748091351.139580 3462857 buffer_comparator.cc:145] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748091351.139583 3462857 buffer_comparator.cc:145] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748091351.139586 3462857 buffer_comparator.cc:145] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748091351.139589 3462857 buffer_comparator.cc:145] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-05-24 12:55:51.139594: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.142884 3462857 buffer_comparator.cc:145] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1748091351.142895 3462857 buffer_comparator.cc:145] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1748091351.142899 3462857 buffer_comparator.cc:145] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1748091351.142902 3462857 buffer_comparator.cc:145] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1748091351.142905 3462857 buffer_comparator.cc:145] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1748091351.142908 3462857 buffer_comparator.cc:145] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.142911 3462857 buffer_comparator.cc:145] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1748091351.142915 3462857 buffer_comparator.cc:145] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1748091351.142920 3462857 buffer_comparator.cc:145] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1748091351.142923 3462857 buffer_comparator.cc:145] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-05-24 12:55:51.142927: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.146302 3462857 buffer_comparator.cc:145] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1748091351.146314 3462857 buffer_comparator.cc:145] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1748091351.146317 3462857 buffer_comparator.cc:145] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1748091351.146321 3462857 buffer_comparator.cc:145] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1748091351.146324 3462857 buffer_comparator.cc:145] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.146327 3462857 buffer_comparator.cc:145] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1748091351.146330 3462857 buffer_comparator.cc:145] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1748091351.146333 3462857 buffer_comparator.cc:145] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1748091351.146336 3462857 buffer_comparator.cc:145] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1748091351.146339 3462857 buffer_comparator.cc:145] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-05-24 12:55:51.146344: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.149583 3462857 buffer_comparator.cc:145] Difference at 0: 1084.56, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748091351.149595 3462857 buffer_comparator.cc:145] Difference at 1: 1350.61, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748091351.149598 3462857 buffer_comparator.cc:145] Difference at 2: 2009.8, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748091351.149602 3462857 buffer_comparator.cc:145] Difference at 3: 1768.04, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748091351.149605 3462857 buffer_comparator.cc:145] Difference at 4: 1240.61, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091351.149608 3462857 buffer_comparator.cc:145] Difference at 6: 1407.03, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091351.149611 3462857 buffer_comparator.cc:145] Difference at 7: 1138.83, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748091351.149614 3462857 buffer_comparator.cc:145] Difference at 8: 1417.44, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.149617 3462857 buffer_comparator.cc:145] Difference at 9: 2084.44, expected 1833.77</span></span>
<span class="line"><span>E0000 00:00:1748091351.149620 3462857 buffer_comparator.cc:145] Difference at 10: 1844.73, expected 1592.38</span></span>
<span class="line"><span>2025-05-24 12:55:51.149625: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.153038 3462857 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091351.153050 3462857 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748091351.153054 3462857 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.153057 3462857 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091351.153060 3462857 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748091351.153063 3462857 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748091351.153066 3462857 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091351.153069 3462857 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.153072 3462857 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748091351.153075 3462857 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-24 12:55:51.153081: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.156418 3462857 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091351.156430 3462857 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748091351.156433 3462857 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.156437 3462857 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091351.156440 3462857 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748091351.156443 3462857 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748091351.156446 3462857 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091351.156449 3462857 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.156452 3462857 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748091351.156455 3462857 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-24 12:55:51.156460: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.159770 3462857 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091351.159782 3462857 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748091351.159786 3462857 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.159789 3462857 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091351.159792 3462857 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748091351.159795 3462857 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748091351.159798 3462857 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091351.159801 3462857 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.159804 3462857 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748091351.159807 3462857 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-24 12:55:51.159812: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.163065 3462857 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091351.163076 3462857 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748091351.163080 3462857 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.163083 3462857 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091351.163086 3462857 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748091351.163089 3462857 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748091351.163092 3462857 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091351.163095 3462857 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.163098 3462857 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748091351.163101 3462857 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-24 12:55:51.163106: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.166404 3462857 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091351.166415 3462857 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748091351.166419 3462857 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.166422 3462857 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091351.166425 3462857 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748091351.166428 3462857 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748091351.166431 3462857 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091351.166435 3462857 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.166438 3462857 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748091351.166441 3462857 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-24 12:55:51.166445: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.169720 3462857 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091351.169731 3462857 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748091351.169735 3462857 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.169738 3462857 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091351.169741 3462857 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748091351.169744 3462857 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748091351.169747 3462857 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091351.169750 3462857 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.169753 3462857 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748091351.169756 3462857 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-24 12:55:51.169761: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.173063 3462857 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091351.173074 3462857 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748091351.173078 3462857 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.173081 3462857 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091351.173084 3462857 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748091351.173087 3462857 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748091351.173090 3462857 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091351.173093 3462857 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.173096 3462857 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748091351.173099 3462857 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-24 12:55:51.173104: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.176370 3462857 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748091351.176383 3462857 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748091351.176386 3462857 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.176390 3462857 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748091351.176393 3462857 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748091351.176396 3462857 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748091351.176399 3462857 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748091351.176402 3462857 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748091351.176405 3462857 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748091351.176408 3462857 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-24 12:55:51.176413: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.179686 3462857 buffer_comparator.cc:145] Difference at 0: 1100.47, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748091351.179698 3462857 buffer_comparator.cc:145] Difference at 1: 1361.33, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748091351.179703 3462857 buffer_comparator.cc:145] Difference at 2: 2059.82, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748091351.179707 3462857 buffer_comparator.cc:145] Difference at 3: 1808.05, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748091351.179710 3462857 buffer_comparator.cc:145] Difference at 4: 1265.06, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091351.179713 3462857 buffer_comparator.cc:145] Difference at 5: 1986, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.179717 3462857 buffer_comparator.cc:145] Difference at 6: 1409.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091351.179721 3462857 buffer_comparator.cc:145] Difference at 7: 1173.38, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748091351.179725 3462857 buffer_comparator.cc:145] Difference at 8: 1420.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.179728 3462857 buffer_comparator.cc:145] Difference at 9: 2114.57, expected 1833.77</span></span>
<span class="line"><span>2025-05-24 12:55:51.179733: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.183010 3462857 buffer_comparator.cc:145] Difference at 0: 1100.47, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748091351.183022 3462857 buffer_comparator.cc:145] Difference at 1: 1361.33, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748091351.183025 3462857 buffer_comparator.cc:145] Difference at 2: 2059.82, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748091351.183028 3462857 buffer_comparator.cc:145] Difference at 3: 1808.05, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748091351.183032 3462857 buffer_comparator.cc:145] Difference at 4: 1265.06, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091351.183035 3462857 buffer_comparator.cc:145] Difference at 5: 1986, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.183038 3462857 buffer_comparator.cc:145] Difference at 6: 1409.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091351.183041 3462857 buffer_comparator.cc:145] Difference at 7: 1173.38, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748091351.183044 3462857 buffer_comparator.cc:145] Difference at 8: 1420.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.183047 3462857 buffer_comparator.cc:145] Difference at 9: 2114.57, expected 1833.77</span></span>
<span class="line"><span>2025-05-24 12:55:51.183052: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.186258 3462857 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748091351.186269 3462857 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748091351.186273 3462857 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748091351.186277 3462857 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748091351.186281 3462857 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091351.186284 3462857 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.186287 3462857 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091351.186290 3462857 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748091351.186293 3462857 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.186296 3462857 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-24 12:55:51.186301: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.189591 3462857 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748091351.189602 3462857 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748091351.189606 3462857 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748091351.189609 3462857 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748091351.189613 3462857 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091351.189616 3462857 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.189619 3462857 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091351.189622 3462857 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748091351.189625 3462857 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.189632 3462857 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-24 12:55:51.189637: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.192957 3462857 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748091351.192968 3462857 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748091351.192972 3462857 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748091351.192975 3462857 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748091351.192978 3462857 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091351.192981 3462857 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.192984 3462857 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091351.192987 3462857 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748091351.192990 3462857 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.192993 3462857 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-24 12:55:51.192998: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.196322 3462857 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748091351.196333 3462857 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748091351.196337 3462857 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748091351.196340 3462857 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748091351.196343 3462857 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091351.196346 3462857 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.196350 3462857 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091351.196354 3462857 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748091351.196357 3462857 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.196360 3462857 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-24 12:55:51.196365: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.199685 3462857 buffer_comparator.cc:145] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1748091351.199696 3462857 buffer_comparator.cc:145] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.199700 3462857 buffer_comparator.cc:145] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1748091351.199703 3462857 buffer_comparator.cc:145] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1748091351.199707 3462857 buffer_comparator.cc:145] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1748091351.199710 3462857 buffer_comparator.cc:145] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1748091351.199713 3462857 buffer_comparator.cc:145] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1748091351.199716 3462857 buffer_comparator.cc:145] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1748091351.199719 3462857 buffer_comparator.cc:145] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1748091351.199722 3462857 buffer_comparator.cc:145] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-05-24 12:55:51.199727: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.202984 3462857 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748091351.202995 3462857 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1748091351.203000 3462857 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1748091351.203003 3462857 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1748091351.203007 3462857 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1748091351.203011 3462857 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1748091351.203014 3462857 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1748091351.203017 3462857 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.77</span></span>
<span class="line"><span>E0000 00:00:1748091351.203021 3462857 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.09</span></span>
<span class="line"><span>E0000 00:00:1748091351.203025 3462857 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.62</span></span>
<span class="line"><span>2025-05-24 12:55:51.203029: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.206276 3462857 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748091351.206288 3462857 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1748091351.206292 3462857 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1748091351.206296 3462857 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1748091351.206299 3462857 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1748091351.206303 3462857 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1748091351.206306 3462857 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1748091351.206310 3462857 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.77</span></span>
<span class="line"><span>E0000 00:00:1748091351.206314 3462857 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.09</span></span>
<span class="line"><span>E0000 00:00:1748091351.206318 3462857 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.62</span></span>
<span class="line"><span>2025-05-24 12:55:51.206323: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.209634 3462857 buffer_comparator.cc:145] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1748091351.209646 3462857 buffer_comparator.cc:145] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.209649 3462857 buffer_comparator.cc:145] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1748091351.209653 3462857 buffer_comparator.cc:145] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1748091351.209656 3462857 buffer_comparator.cc:145] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1748091351.209659 3462857 buffer_comparator.cc:145] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1748091351.209662 3462857 buffer_comparator.cc:145] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1748091351.209665 3462857 buffer_comparator.cc:145] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1748091351.209668 3462857 buffer_comparator.cc:145] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1748091351.209671 3462857 buffer_comparator.cc:145] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-05-24 12:55:51.209676: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091351.212891 3462857 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748091351.212903 3462857 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748091351.212907 3462857 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748091351.212910 3462857 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748091351.212913 3462857 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748091351.212916 3462857 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748091351.212919 3462857 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748091351.212922 3462857 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748091351.212925 3462857 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748091351.212928 3462857 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.77</span></span>
<span class="line"><span>2025-05-24 12:55:51.212933: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-05-24 12:55:54.207826: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:55:54.393059: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:55:54.622369: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-24 12:55:54.768205: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1748091354.775458 3462857 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1748091354.775519 3462857 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1748091354.775526 3462857 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1748091354.775529 3462857 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1748091354.775532 3462857 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1748091354.775535 3462857 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1748091354.775538 3462857 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1748091354.775541 3462857 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1748091354.775544 3462857 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1748091354.775547 3462857 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-05-24 12:55:54.775559: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.779183 3462857 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1748091354.779222 3462857 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1748091354.779225 3462857 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1748091354.779228 3462857 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1748091354.779231 3462857 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1748091354.779234 3462857 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1748091354.779237 3462857 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1748091354.779240 3462857 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1748091354.779243 3462857 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1748091354.779245 3462857 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-05-24 12:55:54.779255: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.782778 3462857 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1748091354.782810 3462857 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1748091354.782813 3462857 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1748091354.782816 3462857 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1748091354.782819 3462857 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1748091354.782822 3462857 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1748091354.782825 3462857 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1748091354.782828 3462857 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1748091354.782831 3462857 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1748091354.782834 3462857 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-05-24 12:55:54.782841: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.786257 3462857 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1748091354.786285 3462857 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1748091354.786289 3462857 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1748091354.786292 3462857 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1748091354.786295 3462857 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1748091354.786298 3462857 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1748091354.786302 3462857 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1748091354.786305 3462857 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1748091354.786308 3462857 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1748091354.786311 3462857 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-05-24 12:55:54.786318: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.789807 3462857 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1748091354.789836 3462857 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1748091354.789839 3462857 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1748091354.789842 3462857 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1748091354.789845 3462857 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1748091354.789848 3462857 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1748091354.789851 3462857 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1748091354.789854 3462857 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1748091354.789857 3462857 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1748091354.789860 3462857 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-05-24 12:55:54.789867: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.793248 3462857 buffer_comparator.cc:145] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1748091354.793277 3462857 buffer_comparator.cc:145] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1748091354.793280 3462857 buffer_comparator.cc:145] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1748091354.793283 3462857 buffer_comparator.cc:145] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1748091354.793286 3462857 buffer_comparator.cc:145] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1748091354.793289 3462857 buffer_comparator.cc:145] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1748091354.793292 3462857 buffer_comparator.cc:145] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1748091354.793295 3462857 buffer_comparator.cc:145] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1748091354.793298 3462857 buffer_comparator.cc:145] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1748091354.793301 3462857 buffer_comparator.cc:145] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-05-24 12:55:54.793308: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.796788 3462857 buffer_comparator.cc:145] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1748091354.796806 3462857 buffer_comparator.cc:145] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1748091354.796809 3462857 buffer_comparator.cc:145] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1748091354.796812 3462857 buffer_comparator.cc:145] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1748091354.796815 3462857 buffer_comparator.cc:145] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1748091354.796818 3462857 buffer_comparator.cc:145] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1748091354.796821 3462857 buffer_comparator.cc:145] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1748091354.796824 3462857 buffer_comparator.cc:145] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1748091354.796827 3462857 buffer_comparator.cc:145] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1748091354.796830 3462857 buffer_comparator.cc:145] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-05-24 12:55:54.796836: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.800142 3462857 buffer_comparator.cc:145] Difference at 0: 903.336, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748091354.800159 3462857 buffer_comparator.cc:145] Difference at 1: 1271.45, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748091354.800162 3462857 buffer_comparator.cc:145] Difference at 2: 1218.72, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748091354.800165 3462857 buffer_comparator.cc:145] Difference at 3: 1830.29, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748091354.800168 3462857 buffer_comparator.cc:145] Difference at 4: 1832.52, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748091354.800171 3462857 buffer_comparator.cc:145] Difference at 5: 1505.57, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748091354.800174 3462857 buffer_comparator.cc:145] Difference at 6: 1003.78, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748091354.800178 3462857 buffer_comparator.cc:145] Difference at 7: 895.724, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1748091354.800181 3462857 buffer_comparator.cc:145] Difference at 8: 1254.14, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748091354.800184 3462857 buffer_comparator.cc:145] Difference at 9: 1207.96, expected 1052.46</span></span>
<span class="line"><span>2025-05-24 12:55:54.800189: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.803648 3462857 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748091354.803666 3462857 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748091354.803669 3462857 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748091354.803672 3462857 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748091354.803675 3462857 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748091354.803678 3462857 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748091354.803681 3462857 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748091354.803684 3462857 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748091354.803687 3462857 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748091354.803690 3462857 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-24 12:55:54.803695: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.807062 3462857 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748091354.807076 3462857 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748091354.807080 3462857 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748091354.807082 3462857 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748091354.807086 3462857 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748091354.807088 3462857 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748091354.807091 3462857 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748091354.807094 3462857 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748091354.807097 3462857 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748091354.807100 3462857 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-24 12:55:54.807105: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.810436 3462857 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748091354.810450 3462857 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748091354.810454 3462857 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748091354.810457 3462857 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748091354.810460 3462857 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748091354.810463 3462857 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748091354.810466 3462857 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748091354.810469 3462857 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748091354.810472 3462857 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748091354.810475 3462857 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-24 12:55:54.810480: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.813753 3462857 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748091354.813767 3462857 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748091354.813770 3462857 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748091354.813773 3462857 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748091354.813776 3462857 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748091354.813779 3462857 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748091354.813782 3462857 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748091354.813785 3462857 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748091354.813788 3462857 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748091354.813790 3462857 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-24 12:55:54.813796: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.817099 3462857 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748091354.817113 3462857 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748091354.817116 3462857 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748091354.817119 3462857 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748091354.817122 3462857 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748091354.817125 3462857 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748091354.817128 3462857 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748091354.817131 3462857 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748091354.817133 3462857 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748091354.817136 3462857 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-24 12:55:54.817141: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.820449 3462857 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748091354.820463 3462857 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748091354.820467 3462857 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748091354.820470 3462857 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748091354.820473 3462857 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748091354.820475 3462857 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748091354.820480 3462857 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748091354.820483 3462857 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748091354.820486 3462857 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748091354.820488 3462857 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-24 12:55:54.820493: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.823862 3462857 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748091354.823903 3462857 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748091354.823906 3462857 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748091354.823909 3462857 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748091354.823912 3462857 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748091354.823915 3462857 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748091354.823918 3462857 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748091354.823921 3462857 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748091354.823924 3462857 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748091354.823927 3462857 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-24 12:55:54.823935: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.827405 3462857 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748091354.827419 3462857 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748091354.827422 3462857 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748091354.827425 3462857 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748091354.827428 3462857 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748091354.827431 3462857 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748091354.827434 3462857 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748091354.827437 3462857 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748091354.827440 3462857 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748091354.827443 3462857 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-24 12:55:54.827447: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.830735 3462857 buffer_comparator.cc:145] Difference at 0: 876.475, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748091354.830749 3462857 buffer_comparator.cc:145] Difference at 1: 1292.4, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748091354.830753 3462857 buffer_comparator.cc:145] Difference at 2: 1239.8, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748091354.830756 3462857 buffer_comparator.cc:145] Difference at 3: 1830.71, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748091354.830759 3462857 buffer_comparator.cc:145] Difference at 4: 1857.47, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748091354.830762 3462857 buffer_comparator.cc:145] Difference at 5: 1551.94, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748091354.830765 3462857 buffer_comparator.cc:145] Difference at 6: 1022.45, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748091354.830768 3462857 buffer_comparator.cc:145] Difference at 8: 1214.29, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748091354.830771 3462857 buffer_comparator.cc:145] Difference at 9: 1173.34, expected 1052.46</span></span>
<span class="line"><span>E0000 00:00:1748091354.830776 3462857 buffer_comparator.cc:145] Difference at 10: 1732.94, expected 1556.04</span></span>
<span class="line"><span>2025-05-24 12:55:54.830781: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.834122 3462857 buffer_comparator.cc:145] Difference at 0: 876.475, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748091354.834175 3462857 buffer_comparator.cc:145] Difference at 1: 1292.4, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748091354.834179 3462857 buffer_comparator.cc:145] Difference at 2: 1239.8, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748091354.834182 3462857 buffer_comparator.cc:145] Difference at 3: 1830.71, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748091354.834185 3462857 buffer_comparator.cc:145] Difference at 4: 1857.47, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748091354.834188 3462857 buffer_comparator.cc:145] Difference at 5: 1551.94, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748091354.834191 3462857 buffer_comparator.cc:145] Difference at 6: 1022.45, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748091354.834194 3462857 buffer_comparator.cc:145] Difference at 8: 1214.29, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748091354.834197 3462857 buffer_comparator.cc:145] Difference at 9: 1173.34, expected 1052.46</span></span>
<span class="line"><span>E0000 00:00:1748091354.834200 3462857 buffer_comparator.cc:145] Difference at 10: 1732.94, expected 1556.04</span></span>
<span class="line"><span>2025-05-24 12:55:54.834210: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.837758 3462857 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748091354.837815 3462857 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748091354.837819 3462857 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748091354.837822 3462857 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748091354.837825 3462857 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748091354.837828 3462857 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748091354.837831 3462857 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748091354.837834 3462857 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1748091354.837837 3462857 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748091354.837840 3462857 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-24 12:55:54.837852: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.841502 3462857 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748091354.841558 3462857 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748091354.841562 3462857 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748091354.841565 3462857 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748091354.841568 3462857 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748091354.841571 3462857 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748091354.841574 3462857 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748091354.841577 3462857 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1748091354.841580 3462857 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748091354.841583 3462857 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-24 12:55:54.841595: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.845298 3462857 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748091354.845356 3462857 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748091354.845359 3462857 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748091354.845363 3462857 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748091354.845366 3462857 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748091354.845369 3462857 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748091354.845372 3462857 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748091354.845375 3462857 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1748091354.845378 3462857 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748091354.845381 3462857 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-24 12:55:54.845393: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.849148 3462857 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748091354.849189 3462857 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748091354.849192 3462857 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748091354.849196 3462857 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748091354.849199 3462857 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748091354.849202 3462857 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748091354.849205 3462857 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748091354.849208 3462857 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1748091354.849211 3462857 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748091354.849214 3462857 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-24 12:55:54.849224: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.852621 3462857 buffer_comparator.cc:145] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1748091354.852637 3462857 buffer_comparator.cc:145] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1748091354.852641 3462857 buffer_comparator.cc:145] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1748091354.852644 3462857 buffer_comparator.cc:145] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1748091354.852647 3462857 buffer_comparator.cc:145] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1748091354.852650 3462857 buffer_comparator.cc:145] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1748091354.852653 3462857 buffer_comparator.cc:145] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1748091354.852656 3462857 buffer_comparator.cc:145] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1748091354.852659 3462857 buffer_comparator.cc:145] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1748091354.852661 3462857 buffer_comparator.cc:145] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-05-24 12:55:54.852666: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748091354.855973 3462857 buffer_comparator.cc:145] Difference at 540: 1203.79, expected 1064.85</span></span>
<span class="line"><span>Test Loss: 71.188759	Test Acc: 10.3000%</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,21)]))}const E=s(c,[["render",i]]);export{d as __pageData,E as default};
