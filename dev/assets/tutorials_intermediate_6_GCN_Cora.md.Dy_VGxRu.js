import{_ as s,c as n,o as e,al as p}from"./chunks/framework.BCN3FD2k.js";const d=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),c={name:"tutorials/intermediate/6_GCN_Cora.md"};function i(t,a,r,l,f,o){return e(),n("div",null,a[0]||(a[0]=[p(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux,</span></span>
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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-05-08 13:58:34.695375: I external/xla/xla/service/service.cc:152] XLA service 0x434d9ec0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-05-08 13:58:34.695788: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1746712714.696550 3243293 se_gpu_pjrt_client.cc:1026] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1746712714.696621 3243293 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1746712714.696652 3243293 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1746712714.709312 3243293 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-9/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-9/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>2025-05-08 13:59:39.056375: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 32 bytes spill stores, 32 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:39.159323: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 48 bytes spill stores, 48 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:39.779883: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 604 bytes spill stores, 608 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:40.165093: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:40.233748: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 1212 bytes spill stores, 976 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:40.472376: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 980 bytes spill stores, 976 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:40.796846: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 104 bytes spill stores, 104 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:41.183877: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 272 bytes spill stores, 272 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:42.061677: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 284 bytes spill stores, 284 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:42.348165: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 292 bytes spill stores, 292 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:42.425377: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:42.472126: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 360 bytes spill stores, 356 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:42.684147: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:42.699837: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:42.871941: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 48 bytes spill stores, 48 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:42.960439: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 996 bytes spill stores, 968 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:43.972938: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:44.011232: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:44.377430: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 13:59:44.668475: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1746712785.076892 3243293 buffer_comparator.cc:145] Difference at 112: 513.993, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.076933 3243293 buffer_comparator.cc:145] Difference at 113: 357.807, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1746712785.076936 3243293 buffer_comparator.cc:145] Difference at 114: 585.471, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1746712785.076940 3243293 buffer_comparator.cc:145] Difference at 115: 420.444, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1746712785.076943 3243293 buffer_comparator.cc:145] Difference at 116: 302.398, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1746712785.076946 3243293 buffer_comparator.cc:145] Difference at 117: 386.144, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1746712785.076949 3243293 buffer_comparator.cc:145] Difference at 118: 587.071, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1746712785.076952 3243293 buffer_comparator.cc:145] Difference at 119: 508.94, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1746712785.076955 3243293 buffer_comparator.cc:145] Difference at 120: 358.918, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1746712785.076958 3243293 buffer_comparator.cc:145] Difference at 121: 578.094, expected 1820.15</span></span>
<span class="line"><span>2025-05-08 13:59:45.076968: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.080363 3243293 buffer_comparator.cc:145] Difference at 112: 513.993, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.080377 3243293 buffer_comparator.cc:145] Difference at 113: 357.807, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1746712785.080381 3243293 buffer_comparator.cc:145] Difference at 114: 585.471, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1746712785.080384 3243293 buffer_comparator.cc:145] Difference at 115: 420.444, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1746712785.080387 3243293 buffer_comparator.cc:145] Difference at 116: 302.398, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1746712785.080390 3243293 buffer_comparator.cc:145] Difference at 117: 386.144, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1746712785.080393 3243293 buffer_comparator.cc:145] Difference at 118: 587.071, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1746712785.080396 3243293 buffer_comparator.cc:145] Difference at 119: 508.94, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1746712785.080399 3243293 buffer_comparator.cc:145] Difference at 120: 358.918, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1746712785.080402 3243293 buffer_comparator.cc:145] Difference at 121: 578.094, expected 1820.15</span></span>
<span class="line"><span>2025-05-08 13:59:45.080407: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.083512 3243293 buffer_comparator.cc:145] Difference at 112: 513.993, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.083529 3243293 buffer_comparator.cc:145] Difference at 113: 357.807, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1746712785.083532 3243293 buffer_comparator.cc:145] Difference at 114: 585.471, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1746712785.083535 3243293 buffer_comparator.cc:145] Difference at 115: 420.444, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1746712785.083538 3243293 buffer_comparator.cc:145] Difference at 116: 302.398, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1746712785.083541 3243293 buffer_comparator.cc:145] Difference at 117: 386.144, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1746712785.083544 3243293 buffer_comparator.cc:145] Difference at 118: 587.071, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1746712785.083547 3243293 buffer_comparator.cc:145] Difference at 119: 508.94, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1746712785.083551 3243293 buffer_comparator.cc:145] Difference at 120: 358.918, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1746712785.083554 3243293 buffer_comparator.cc:145] Difference at 121: 578.094, expected 1820.15</span></span>
<span class="line"><span>2025-05-08 13:59:45.083559: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.086592 3243293 buffer_comparator.cc:145] Difference at 0: 1084.56, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1746712785.086607 3243293 buffer_comparator.cc:145] Difference at 1: 1350.61, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1746712785.086610 3243293 buffer_comparator.cc:145] Difference at 2: 2009.8, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1746712785.086613 3243293 buffer_comparator.cc:145] Difference at 3: 1768.04, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.086616 3243293 buffer_comparator.cc:145] Difference at 4: 1240.61, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.086619 3243293 buffer_comparator.cc:145] Difference at 6: 1407.03, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712785.086622 3243293 buffer_comparator.cc:145] Difference at 7: 1138.83, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.086626 3243293 buffer_comparator.cc:145] Difference at 8: 1417.44, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712785.086628 3243293 buffer_comparator.cc:145] Difference at 9: 2084.44, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1746712785.086631 3243293 buffer_comparator.cc:145] Difference at 10: 1844.73, expected 1592.37</span></span>
<span class="line"><span>2025-05-08 13:59:45.086636: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.089740 3243293 buffer_comparator.cc:145] Difference at 0: 1084.56, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1746712785.089755 3243293 buffer_comparator.cc:145] Difference at 1: 1350.61, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1746712785.089758 3243293 buffer_comparator.cc:145] Difference at 2: 2009.8, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1746712785.089761 3243293 buffer_comparator.cc:145] Difference at 3: 1768.04, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.089764 3243293 buffer_comparator.cc:145] Difference at 4: 1240.61, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.089767 3243293 buffer_comparator.cc:145] Difference at 6: 1407.03, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712785.089770 3243293 buffer_comparator.cc:145] Difference at 7: 1138.83, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.089773 3243293 buffer_comparator.cc:145] Difference at 8: 1417.44, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712785.089776 3243293 buffer_comparator.cc:145] Difference at 9: 2084.44, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1746712785.089779 3243293 buffer_comparator.cc:145] Difference at 10: 1844.73, expected 1592.37</span></span>
<span class="line"><span>2025-05-08 13:59:45.089784: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.092838 3243293 buffer_comparator.cc:145] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1746712785.092852 3243293 buffer_comparator.cc:145] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1746712785.092855 3243293 buffer_comparator.cc:145] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1746712785.092858 3243293 buffer_comparator.cc:145] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1746712785.092861 3243293 buffer_comparator.cc:145] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1746712785.092864 3243293 buffer_comparator.cc:145] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1746712785.092867 3243293 buffer_comparator.cc:145] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1746712785.092870 3243293 buffer_comparator.cc:145] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1746712785.092873 3243293 buffer_comparator.cc:145] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1746712785.092876 3243293 buffer_comparator.cc:145] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-05-08 13:59:45.092882: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.095972 3243293 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.095987 3243293 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.095992 3243293 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.8</span></span>
<span class="line"><span>E0000 00:00:1746712785.095995 3243293 buffer_comparator.cc:145] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1746712785.095998 3243293 buffer_comparator.cc:145] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1746712785.096001 3243293 buffer_comparator.cc:145] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1746712785.096004 3243293 buffer_comparator.cc:145] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1746712785.096007 3243293 buffer_comparator.cc:145] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1746712785.096010 3243293 buffer_comparator.cc:145] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1746712785.096013 3243293 buffer_comparator.cc:145] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>2025-05-08 13:59:45.096018: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.099037 3243293 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1746712785.099052 3243293 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1746712785.099055 3243293 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1746712785.099058 3243293 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.099061 3243293 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.099064 3243293 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.099067 3243293 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712785.099070 3243293 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.099073 3243293 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712785.099076 3243293 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.76</span></span>
<span class="line"><span>2025-05-08 13:59:45.099081: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.102233 3243293 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1746712785.102250 3243293 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.102253 3243293 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.102256 3243293 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1746712785.102259 3243293 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712785.102263 3243293 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1746712785.102266 3243293 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1746712785.102269 3243293 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1746712785.102272 3243293 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712785.102275 3243293 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-08 13:59:45.102279: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.105374 3243293 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1746712785.105389 3243293 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.105392 3243293 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.105395 3243293 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1746712785.105398 3243293 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712785.105402 3243293 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1746712785.105405 3243293 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1746712785.105408 3243293 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1746712785.105410 3243293 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712785.105414 3243293 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-08 13:59:45.105418: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.108480 3243293 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1746712785.108495 3243293 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.108499 3243293 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.108502 3243293 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1746712785.108505 3243293 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712785.108508 3243293 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1746712785.108511 3243293 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1746712785.108514 3243293 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1746712785.108517 3243293 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712785.108520 3243293 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-08 13:59:45.108527: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.111584 3243293 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1746712785.111599 3243293 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.111602 3243293 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.111605 3243293 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1746712785.111608 3243293 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712785.111611 3243293 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1746712785.111614 3243293 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1746712785.111617 3243293 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1746712785.111620 3243293 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712785.111623 3243293 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-08 13:59:45.111628: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.114712 3243293 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1746712785.114730 3243293 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.114733 3243293 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.114736 3243293 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1746712785.114739 3243293 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712785.114742 3243293 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1746712785.114745 3243293 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1746712785.114748 3243293 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1746712785.114751 3243293 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712785.114754 3243293 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-08 13:59:45.114759: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.117818 3243293 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1746712785.117832 3243293 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.117835 3243293 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.117838 3243293 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1746712785.117841 3243293 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712785.117844 3243293 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1746712785.117847 3243293 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1746712785.117850 3243293 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1746712785.117853 3243293 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712785.117856 3243293 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-08 13:59:45.117861: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.120925 3243293 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1746712785.120941 3243293 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.120945 3243293 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.120948 3243293 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1746712785.120951 3243293 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712785.120954 3243293 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1746712785.120957 3243293 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1746712785.120960 3243293 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1746712785.120963 3243293 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712785.120966 3243293 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-08 13:59:45.120971: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.124014 3243293 buffer_comparator.cc:145] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1746712785.124028 3243293 buffer_comparator.cc:145] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.124031 3243293 buffer_comparator.cc:145] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.124035 3243293 buffer_comparator.cc:145] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1746712785.124038 3243293 buffer_comparator.cc:145] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712785.124041 3243293 buffer_comparator.cc:145] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1746712785.124044 3243293 buffer_comparator.cc:145] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1746712785.124047 3243293 buffer_comparator.cc:145] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1746712785.124050 3243293 buffer_comparator.cc:145] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712785.124053 3243293 buffer_comparator.cc:145] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-05-08 13:59:45.124058: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.127110 3243293 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1746712785.127124 3243293 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1746712785.127127 3243293 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1746712785.127130 3243293 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.127133 3243293 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.127136 3243293 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.127139 3243293 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712785.127142 3243293 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.127145 3243293 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712785.127148 3243293 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.76</span></span>
<span class="line"><span>2025-05-08 13:59:45.127153: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.130208 3243293 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1746712785.130226 3243293 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1746712785.130230 3243293 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1746712785.130233 3243293 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.130236 3243293 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.130239 3243293 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.130242 3243293 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712785.130245 3243293 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.130248 3243293 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712785.130251 3243293 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.76</span></span>
<span class="line"><span>2025-05-08 13:59:45.130256: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.133253 3243293 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1746712785.133267 3243293 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1746712785.133270 3243293 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1746712785.133273 3243293 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.133276 3243293 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.133280 3243293 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.133283 3243293 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712785.133291 3243293 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.133295 3243293 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712785.133298 3243293 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.76</span></span>
<span class="line"><span>2025-05-08 13:59:45.133302: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.136389 3243293 buffer_comparator.cc:145] Difference at 0: 1506.95, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1746712785.136403 3243293 buffer_comparator.cc:145] Difference at 1: 1786.05, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1746712785.136407 3243293 buffer_comparator.cc:145] Difference at 2: 2699.73, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1746712785.136410 3243293 buffer_comparator.cc:145] Difference at 3: 2437.04, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.136413 3243293 buffer_comparator.cc:145] Difference at 4: 1660.22, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.136416 3243293 buffer_comparator.cc:145] Difference at 5: 2636.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.136419 3243293 buffer_comparator.cc:145] Difference at 6: 1898.51, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712785.136422 3243293 buffer_comparator.cc:145] Difference at 7: 1482.97, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.136425 3243293 buffer_comparator.cc:145] Difference at 8: 1801.1, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712785.136428 3243293 buffer_comparator.cc:145] Difference at 9: 2662.69, expected 1833.76</span></span>
<span class="line"><span>2025-05-08 13:59:45.136432: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.139563 3243293 buffer_comparator.cc:145] Difference at 0: 1506.95, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1746712785.139577 3243293 buffer_comparator.cc:145] Difference at 1: 1786.05, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1746712785.139580 3243293 buffer_comparator.cc:145] Difference at 2: 2699.73, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1746712785.139583 3243293 buffer_comparator.cc:145] Difference at 3: 2437.04, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.139586 3243293 buffer_comparator.cc:145] Difference at 4: 1660.22, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.139589 3243293 buffer_comparator.cc:145] Difference at 5: 2636.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.139592 3243293 buffer_comparator.cc:145] Difference at 6: 1898.51, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712785.139595 3243293 buffer_comparator.cc:145] Difference at 7: 1482.97, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.139598 3243293 buffer_comparator.cc:145] Difference at 8: 1801.1, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712785.139601 3243293 buffer_comparator.cc:145] Difference at 9: 2662.69, expected 1833.76</span></span>
<span class="line"><span>2025-05-08 13:59:45.139606: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.142735 3243293 buffer_comparator.cc:145] Difference at 0: 1506.95, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1746712785.142752 3243293 buffer_comparator.cc:145] Difference at 1: 1786.05, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1746712785.142755 3243293 buffer_comparator.cc:145] Difference at 2: 2699.73, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1746712785.142758 3243293 buffer_comparator.cc:145] Difference at 3: 2437.04, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.142761 3243293 buffer_comparator.cc:145] Difference at 4: 1660.22, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.142764 3243293 buffer_comparator.cc:145] Difference at 5: 2636.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.142767 3243293 buffer_comparator.cc:145] Difference at 6: 1898.51, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712785.142770 3243293 buffer_comparator.cc:145] Difference at 7: 1482.97, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.142774 3243293 buffer_comparator.cc:145] Difference at 8: 1801.1, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712785.142777 3243293 buffer_comparator.cc:145] Difference at 9: 2662.69, expected 1833.76</span></span>
<span class="line"><span>2025-05-08 13:59:45.142782: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.146099 3243293 buffer_comparator.cc:145] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1746712785.146113 3243293 buffer_comparator.cc:145] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1746712785.146117 3243293 buffer_comparator.cc:145] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.146120 3243293 buffer_comparator.cc:145] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1746712785.146123 3243293 buffer_comparator.cc:145] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1746712785.146126 3243293 buffer_comparator.cc:145] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1746712785.146129 3243293 buffer_comparator.cc:145] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1746712785.146132 3243293 buffer_comparator.cc:145] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1746712785.146135 3243293 buffer_comparator.cc:145] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1746712785.146138 3243293 buffer_comparator.cc:145] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-05-08 13:59:45.146143: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.149194 3243293 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.149211 3243293 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.149215 3243293 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.8</span></span>
<span class="line"><span>E0000 00:00:1746712785.149218 3243293 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.413</span></span>
<span class="line"><span>E0000 00:00:1746712785.149222 3243293 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.54</span></span>
<span class="line"><span>E0000 00:00:1746712785.149225 3243293 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1746712785.149229 3243293 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1746712785.149232 3243293 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.76</span></span>
<span class="line"><span>E0000 00:00:1746712785.149236 3243293 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.08</span></span>
<span class="line"><span>E0000 00:00:1746712785.149239 3243293 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.61</span></span>
<span class="line"><span>2025-05-08 13:59:45.149244: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.152294 3243293 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.152308 3243293 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.152312 3243293 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.8</span></span>
<span class="line"><span>E0000 00:00:1746712785.152316 3243293 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.413</span></span>
<span class="line"><span>E0000 00:00:1746712785.152319 3243293 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.54</span></span>
<span class="line"><span>E0000 00:00:1746712785.152323 3243293 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1746712785.152326 3243293 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1746712785.152330 3243293 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.76</span></span>
<span class="line"><span>E0000 00:00:1746712785.152333 3243293 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.08</span></span>
<span class="line"><span>E0000 00:00:1746712785.152336 3243293 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.61</span></span>
<span class="line"><span>2025-05-08 13:59:45.152342: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.155496 3243293 buffer_comparator.cc:145] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1746712785.155510 3243293 buffer_comparator.cc:145] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1746712785.155514 3243293 buffer_comparator.cc:145] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.155517 3243293 buffer_comparator.cc:145] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1746712785.155520 3243293 buffer_comparator.cc:145] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1746712785.155523 3243293 buffer_comparator.cc:145] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1746712785.155526 3243293 buffer_comparator.cc:145] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1746712785.155529 3243293 buffer_comparator.cc:145] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1746712785.155532 3243293 buffer_comparator.cc:145] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1746712785.155535 3243293 buffer_comparator.cc:145] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-05-08 13:59:45.155539: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.158758 3243293 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1746712785.158772 3243293 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1746712785.158775 3243293 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1746712785.158778 3243293 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.158781 3243293 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.158784 3243293 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1746712785.158787 3243293 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712785.158790 3243293 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1746712785.158793 3243293 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712785.158796 3243293 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.76</span></span>
<span class="line"><span>2025-05-08 13:59:45.158801: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.162463 3243293 buffer_comparator.cc:145] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1746712785.162477 3243293 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1746712785.162481 3243293 buffer_comparator.cc:145] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1746712785.162483 3243293 buffer_comparator.cc:145] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1746712785.162486 3243293 buffer_comparator.cc:145] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1746712785.162489 3243293 buffer_comparator.cc:145] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1746712785.162492 3243293 buffer_comparator.cc:145] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1746712785.162494 3243293 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1746712785.162497 3243293 buffer_comparator.cc:145] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1746712785.162500 3243293 buffer_comparator.cc:145] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-05-08 13:59:45.162505: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.164613 3243293 buffer_comparator.cc:145] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1746712785.164629 3243293 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1746712785.164633 3243293 buffer_comparator.cc:145] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1746712785.164635 3243293 buffer_comparator.cc:145] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1746712785.164638 3243293 buffer_comparator.cc:145] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1746712785.164641 3243293 buffer_comparator.cc:145] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1746712785.164644 3243293 buffer_comparator.cc:145] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1746712785.164646 3243293 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1746712785.164649 3243293 buffer_comparator.cc:145] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1746712785.164652 3243293 buffer_comparator.cc:145] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-05-08 13:59:45.164656: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.166769 3243293 buffer_comparator.cc:145] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1746712785.166783 3243293 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1746712785.166786 3243293 buffer_comparator.cc:145] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1746712785.166789 3243293 buffer_comparator.cc:145] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1746712785.166792 3243293 buffer_comparator.cc:145] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1746712785.166795 3243293 buffer_comparator.cc:145] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1746712785.166797 3243293 buffer_comparator.cc:145] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1746712785.166800 3243293 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1746712785.166803 3243293 buffer_comparator.cc:145] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1746712785.166805 3243293 buffer_comparator.cc:145] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-05-08 13:59:45.166810: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.168955 3243293 buffer_comparator.cc:145] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1746712785.168969 3243293 buffer_comparator.cc:145] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1746712785.168972 3243293 buffer_comparator.cc:145] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1746712785.168975 3243293 buffer_comparator.cc:145] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1746712785.168978 3243293 buffer_comparator.cc:145] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1746712785.168981 3243293 buffer_comparator.cc:145] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1746712785.168983 3243293 buffer_comparator.cc:145] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1746712785.168986 3243293 buffer_comparator.cc:145] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1746712785.168989 3243293 buffer_comparator.cc:145] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1746712785.168992 3243293 buffer_comparator.cc:145] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-05-08 13:59:45.168996: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.171121 3243293 buffer_comparator.cc:145] Difference at 32: -nan, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1746712785.171135 3243293 buffer_comparator.cc:145] Difference at 33: -nan, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1746712785.171138 3243293 buffer_comparator.cc:145] Difference at 34: -nan, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1746712785.171141 3243293 buffer_comparator.cc:145] Difference at 35: -nan, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1746712785.171144 3243293 buffer_comparator.cc:145] Difference at 36: -nan, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1746712785.171147 3243293 buffer_comparator.cc:145] Difference at 37: -nan, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1746712785.171150 3243293 buffer_comparator.cc:145] Difference at 38: -nan, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1746712785.171153 3243293 buffer_comparator.cc:145] Difference at 39: -nan, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1746712785.171156 3243293 buffer_comparator.cc:145] Difference at 40: -nan, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1746712785.171158 3243293 buffer_comparator.cc:145] Difference at 41: -nan, expected 13.7427</span></span>
<span class="line"><span>2025-05-08 13:59:45.171163: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.173280 3243293 buffer_comparator.cc:145] Difference at 32: -nan, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1746712785.173299 3243293 buffer_comparator.cc:145] Difference at 33: -nan, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1746712785.173302 3243293 buffer_comparator.cc:145] Difference at 34: -nan, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1746712785.173305 3243293 buffer_comparator.cc:145] Difference at 35: -nan, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1746712785.173308 3243293 buffer_comparator.cc:145] Difference at 36: -nan, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1746712785.173311 3243293 buffer_comparator.cc:145] Difference at 37: -nan, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1746712785.173313 3243293 buffer_comparator.cc:145] Difference at 38: -nan, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1746712785.173316 3243293 buffer_comparator.cc:145] Difference at 39: -nan, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1746712785.173319 3243293 buffer_comparator.cc:145] Difference at 40: -nan, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1746712785.173322 3243293 buffer_comparator.cc:145] Difference at 41: -nan, expected 13.7427</span></span>
<span class="line"><span>2025-05-08 13:59:45.173326: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.175438 3243293 buffer_comparator.cc:145] Difference at 32: -nan, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1746712785.175455 3243293 buffer_comparator.cc:145] Difference at 33: -nan, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1746712785.175458 3243293 buffer_comparator.cc:145] Difference at 34: -nan, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1746712785.175460 3243293 buffer_comparator.cc:145] Difference at 35: -nan, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1746712785.175463 3243293 buffer_comparator.cc:145] Difference at 36: -nan, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1746712785.175466 3243293 buffer_comparator.cc:145] Difference at 37: -nan, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1746712785.175469 3243293 buffer_comparator.cc:145] Difference at 38: -nan, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1746712785.175471 3243293 buffer_comparator.cc:145] Difference at 39: -nan, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1746712785.175474 3243293 buffer_comparator.cc:145] Difference at 40: -nan, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1746712785.175477 3243293 buffer_comparator.cc:145] Difference at 41: -nan, expected 13.7427</span></span>
<span class="line"><span>2025-05-08 13:59:45.175481: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.177602 3243293 buffer_comparator.cc:145] Difference at 64: -nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1746712785.177615 3243293 buffer_comparator.cc:145] Difference at 65: -nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1746712785.177618 3243293 buffer_comparator.cc:145] Difference at 66: -nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1746712785.177621 3243293 buffer_comparator.cc:145] Difference at 67: -nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1746712785.177624 3243293 buffer_comparator.cc:145] Difference at 68: -nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1746712785.177627 3243293 buffer_comparator.cc:145] Difference at 69: -nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1746712785.177629 3243293 buffer_comparator.cc:145] Difference at 70: -nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1746712785.177633 3243293 buffer_comparator.cc:145] Difference at 71: -nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1746712785.177636 3243293 buffer_comparator.cc:145] Difference at 72: -nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1746712785.177638 3243293 buffer_comparator.cc:145] Difference at 73: -nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-08 13:59:45.177643: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.179777 3243293 buffer_comparator.cc:145] Difference at 64: -nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1746712785.179796 3243293 buffer_comparator.cc:145] Difference at 65: -nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1746712785.179800 3243293 buffer_comparator.cc:145] Difference at 66: -nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1746712785.179802 3243293 buffer_comparator.cc:145] Difference at 67: -nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1746712785.179805 3243293 buffer_comparator.cc:145] Difference at 68: -nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1746712785.179809 3243293 buffer_comparator.cc:145] Difference at 69: -nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1746712785.179812 3243293 buffer_comparator.cc:145] Difference at 70: -nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1746712785.179815 3243293 buffer_comparator.cc:145] Difference at 71: -nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1746712785.179819 3243293 buffer_comparator.cc:145] Difference at 72: -nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1746712785.179821 3243293 buffer_comparator.cc:145] Difference at 73: -nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-08 13:59:45.179826: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.181953 3243293 buffer_comparator.cc:145] Difference at 64: -nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1746712785.181971 3243293 buffer_comparator.cc:145] Difference at 65: -nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1746712785.181974 3243293 buffer_comparator.cc:145] Difference at 66: -nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1746712785.181976 3243293 buffer_comparator.cc:145] Difference at 67: -nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1746712785.181979 3243293 buffer_comparator.cc:145] Difference at 68: -nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1746712785.181982 3243293 buffer_comparator.cc:145] Difference at 69: -nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1746712785.181985 3243293 buffer_comparator.cc:145] Difference at 70: -nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1746712785.181988 3243293 buffer_comparator.cc:145] Difference at 71: -nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1746712785.181990 3243293 buffer_comparator.cc:145] Difference at 72: -nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1746712785.181993 3243293 buffer_comparator.cc:145] Difference at 73: -nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-08 13:59:45.181997: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.184251 3243293 buffer_comparator.cc:145] Difference at 0: 16.5369, expected 14.4011</span></span>
<span class="line"><span>E0000 00:00:1746712785.184265 3243293 buffer_comparator.cc:145] Difference at 1: 19.4176, expected 15.9904</span></span>
<span class="line"><span>E0000 00:00:1746712785.184268 3243293 buffer_comparator.cc:145] Difference at 2: 16.204, expected 13.4103</span></span>
<span class="line"><span>E0000 00:00:1746712785.184272 3243293 buffer_comparator.cc:145] Difference at 6: 13.1759, expected 11.4953</span></span>
<span class="line"><span>E0000 00:00:1746712785.184275 3243293 buffer_comparator.cc:145] Difference at 9: 16.3002, expected 14.2452</span></span>
<span class="line"><span>E0000 00:00:1746712785.184278 3243293 buffer_comparator.cc:145] Difference at 11: 15.6508, expected 13.739</span></span>
<span class="line"><span>E0000 00:00:1746712785.184281 3243293 buffer_comparator.cc:145] Difference at 12: 20.6885, expected 16.297</span></span>
<span class="line"><span>E0000 00:00:1746712785.184290 3243293 buffer_comparator.cc:145] Difference at 13: 17.247, expected 14.372</span></span>
<span class="line"><span>E0000 00:00:1746712785.184293 3243293 buffer_comparator.cc:145] Difference at 14: 14.7694, expected 12.4213</span></span>
<span class="line"><span>E0000 00:00:1746712785.184296 3243293 buffer_comparator.cc:145] Difference at 16: 17.2743, expected 15.1227</span></span>
<span class="line"><span>2025-05-08 13:59:45.184302: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.186412 3243293 buffer_comparator.cc:145] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1746712785.186426 3243293 buffer_comparator.cc:145] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1746712785.186429 3243293 buffer_comparator.cc:145] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1746712785.186432 3243293 buffer_comparator.cc:145] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1746712785.186435 3243293 buffer_comparator.cc:145] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1746712785.186437 3243293 buffer_comparator.cc:145] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1746712785.186440 3243293 buffer_comparator.cc:145] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1746712785.186443 3243293 buffer_comparator.cc:145] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1746712785.186446 3243293 buffer_comparator.cc:145] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1746712785.186448 3243293 buffer_comparator.cc:145] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-08 13:59:45.186453: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.188579 3243293 buffer_comparator.cc:145] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1746712785.188594 3243293 buffer_comparator.cc:145] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1746712785.188597 3243293 buffer_comparator.cc:145] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1746712785.188599 3243293 buffer_comparator.cc:145] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1746712785.188602 3243293 buffer_comparator.cc:145] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1746712785.188605 3243293 buffer_comparator.cc:145] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1746712785.188608 3243293 buffer_comparator.cc:145] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1746712785.188610 3243293 buffer_comparator.cc:145] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1746712785.188613 3243293 buffer_comparator.cc:145] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1746712785.188616 3243293 buffer_comparator.cc:145] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-08 13:59:45.188620: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.190716 3243293 buffer_comparator.cc:145] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1746712785.190730 3243293 buffer_comparator.cc:145] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1746712785.190733 3243293 buffer_comparator.cc:145] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1746712785.190736 3243293 buffer_comparator.cc:145] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1746712785.190739 3243293 buffer_comparator.cc:145] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1746712785.190742 3243293 buffer_comparator.cc:145] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1746712785.190744 3243293 buffer_comparator.cc:145] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1746712785.190747 3243293 buffer_comparator.cc:145] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1746712785.190750 3243293 buffer_comparator.cc:145] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1746712785.190752 3243293 buffer_comparator.cc:145] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-08 13:59:45.190757: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.192741 3243293 buffer_comparator.cc:145] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1746712785.192755 3243293 buffer_comparator.cc:145] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1746712785.192758 3243293 buffer_comparator.cc:145] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1746712785.192761 3243293 buffer_comparator.cc:145] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1746712785.192764 3243293 buffer_comparator.cc:145] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1746712785.192767 3243293 buffer_comparator.cc:145] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1746712785.192769 3243293 buffer_comparator.cc:145] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1746712785.192772 3243293 buffer_comparator.cc:145] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1746712785.192775 3243293 buffer_comparator.cc:145] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1746712785.192777 3243293 buffer_comparator.cc:145] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-08 13:59:45.192782: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.194582 3243293 buffer_comparator.cc:145] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1746712785.194595 3243293 buffer_comparator.cc:145] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1746712785.194598 3243293 buffer_comparator.cc:145] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1746712785.194601 3243293 buffer_comparator.cc:145] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1746712785.194604 3243293 buffer_comparator.cc:145] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1746712785.194607 3243293 buffer_comparator.cc:145] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1746712785.194609 3243293 buffer_comparator.cc:145] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1746712785.194612 3243293 buffer_comparator.cc:145] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1746712785.194615 3243293 buffer_comparator.cc:145] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1746712785.194617 3243293 buffer_comparator.cc:145] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-05-08 13:59:45.194622: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.196619 3243293 buffer_comparator.cc:145] Difference at 128: nan, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1746712785.196635 3243293 buffer_comparator.cc:145] Difference at 129: nan, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1746712785.196638 3243293 buffer_comparator.cc:145] Difference at 130: nan, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1746712785.196641 3243293 buffer_comparator.cc:145] Difference at 131: nan, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1746712785.196644 3243293 buffer_comparator.cc:145] Difference at 132: nan, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1746712785.196647 3243293 buffer_comparator.cc:145] Difference at 133: nan, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1746712785.196649 3243293 buffer_comparator.cc:145] Difference at 134: nan, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1746712785.196652 3243293 buffer_comparator.cc:145] Difference at 135: nan, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1746712785.196655 3243293 buffer_comparator.cc:145] Difference at 136: nan, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1746712785.196657 3243293 buffer_comparator.cc:145] Difference at 137: nan, expected 12.9584</span></span>
<span class="line"><span>2025-05-08 13:59:45.196662: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.198454 3243293 buffer_comparator.cc:145] Difference at 128: nan, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1746712785.198468 3243293 buffer_comparator.cc:145] Difference at 129: nan, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1746712785.198471 3243293 buffer_comparator.cc:145] Difference at 130: nan, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1746712785.198474 3243293 buffer_comparator.cc:145] Difference at 131: nan, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1746712785.198477 3243293 buffer_comparator.cc:145] Difference at 132: nan, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1746712785.198480 3243293 buffer_comparator.cc:145] Difference at 133: nan, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1746712785.198483 3243293 buffer_comparator.cc:145] Difference at 134: nan, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1746712785.198485 3243293 buffer_comparator.cc:145] Difference at 135: nan, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1746712785.198488 3243293 buffer_comparator.cc:145] Difference at 136: nan, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1746712785.198491 3243293 buffer_comparator.cc:145] Difference at 137: nan, expected 12.9584</span></span>
<span class="line"><span>2025-05-08 13:59:45.198495: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.207467 3243293 buffer_comparator.cc:145] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1746712785.207482 3243293 buffer_comparator.cc:145] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.207485 3243293 buffer_comparator.cc:145] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1746712785.207488 3243293 buffer_comparator.cc:145] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1746712785.207491 3243293 buffer_comparator.cc:145] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.207494 3243293 buffer_comparator.cc:145] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1746712785.207497 3243293 buffer_comparator.cc:145] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.207500 3243293 buffer_comparator.cc:145] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1746712785.207503 3243293 buffer_comparator.cc:145] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1746712785.207506 3243293 buffer_comparator.cc:145] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-05-08 13:59:45.207510: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.212661 3243293 buffer_comparator.cc:145] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1746712785.212676 3243293 buffer_comparator.cc:145] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.212679 3243293 buffer_comparator.cc:145] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1746712785.212682 3243293 buffer_comparator.cc:145] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1746712785.212685 3243293 buffer_comparator.cc:145] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.212687 3243293 buffer_comparator.cc:145] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1746712785.212690 3243293 buffer_comparator.cc:145] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.212693 3243293 buffer_comparator.cc:145] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1746712785.212696 3243293 buffer_comparator.cc:145] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1746712785.212699 3243293 buffer_comparator.cc:145] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-05-08 13:59:45.212704: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712785.217910 3243293 buffer_comparator.cc:145] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1746712785.217927 3243293 buffer_comparator.cc:145] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1746712785.217930 3243293 buffer_comparator.cc:145] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1746712785.217933 3243293 buffer_comparator.cc:145] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1746712785.217936 3243293 buffer_comparator.cc:145] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1746712785.217939 3243293 buffer_comparator.cc:145] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1746712785.217942 3243293 buffer_comparator.cc:145] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1746712785.217945 3243293 buffer_comparator.cc:145] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>Epoch   1	Train Loss: 15.839269	Train Acc: 12.8571%	Val Loss: 13.304101	Val Acc: 6.4000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 14.910770	Train Acc: 15.0000%	Val Loss: 14.105294	Val Acc: 8.2000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 16.832224	Train Acc: 15.7143%	Val Loss: 15.776460	Val Acc: 11.8000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 17.614296	Train Acc: 15.7143%	Val Loss: 17.323206	Val Acc: 12.8000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 18.043964	Train Acc: 14.2857%	Val Loss: 18.669127	Val Acc: 12.8000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 22.216852	Train Acc: 13.5714%	Val Loss: 19.769588	Val Acc: 13.0000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 25.257898	Train Acc: 14.2857%	Val Loss: 20.870613	Val Acc: 13.0000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 26.200783	Train Acc: 15.7143%	Val Loss: 21.920601	Val Acc: 12.8000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 25.289383	Train Acc: 14.2857%	Val Loss: 23.100597	Val Acc: 12.8000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 29.291113	Train Acc: 15.0000%	Val Loss: 24.597458	Val Acc: 12.6000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 26.882961	Train Acc: 15.7143%	Val Loss: 26.585041	Val Acc: 12.6000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 35.986889	Train Acc: 13.5714%	Val Loss: 29.549456	Val Acc: 12.6000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 38.400036	Train Acc: 12.8571%	Val Loss: 34.047798	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 42.058643	Train Acc: 14.2857%	Val Loss: 39.204159	Val Acc: 10.8000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 51.282295	Train Acc: 14.2857%	Val Loss: 44.169037	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 56.163136	Train Acc: 14.2857%	Val Loss: 48.999454	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 60.577183	Train Acc: 14.2857%	Val Loss: 53.985180	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 60.782681	Train Acc: 14.2857%	Val Loss: 59.159279	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 80.903244	Train Acc: 14.2857%	Val Loss: 64.460785	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 77.494797	Train Acc: 14.2857%	Val Loss: 69.730896	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 72.299515	Train Acc: 14.2857%	Val Loss: 75.099144	Val Acc: 11.0000%</span></span>
<span class="line"><span>Early Stopping at Epoch 21</span></span>
<span class="line"><span>2025-05-08 14:00:54.638651: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 14:00:55.180983: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 14:00:55.397146: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 14:00:56.515683: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1746712856.523360 3243293 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1746712856.523428 3243293 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1746712856.523449 3243293 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1746712856.523452 3243293 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.523456 3243293 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1746712856.523459 3243293 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1746712856.523463 3243293 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1746712856.523466 3243293 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1746712856.523470 3243293 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1746712856.523473 3243293 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-08 14:00:56.523486: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.527299 3243293 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1746712856.527363 3243293 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1746712856.527370 3243293 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1746712856.527373 3243293 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.527376 3243293 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1746712856.527379 3243293 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1746712856.527382 3243293 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1746712856.527385 3243293 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1746712856.527389 3243293 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1746712856.527392 3243293 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-08 14:00:56.527404: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.531153 3243293 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1746712856.531215 3243293 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1746712856.531221 3243293 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1746712856.531224 3243293 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.531227 3243293 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1746712856.531231 3243293 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1746712856.531241 3243293 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1746712856.531244 3243293 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1746712856.531247 3243293 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1746712856.531250 3243293 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-08 14:00:56.531262: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.534893 3243293 buffer_comparator.cc:145] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1746712856.534955 3243293 buffer_comparator.cc:145] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1746712856.534961 3243293 buffer_comparator.cc:145] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1746712856.534965 3243293 buffer_comparator.cc:145] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.534968 3243293 buffer_comparator.cc:145] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1746712856.534971 3243293 buffer_comparator.cc:145] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1746712856.534975 3243293 buffer_comparator.cc:145] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1746712856.534978 3243293 buffer_comparator.cc:145] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1746712856.534981 3243293 buffer_comparator.cc:145] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1746712856.534984 3243293 buffer_comparator.cc:145] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-05-08 14:00:56.534997: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.538746 3243293 buffer_comparator.cc:145] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1746712856.538807 3243293 buffer_comparator.cc:145] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1746712856.538814 3243293 buffer_comparator.cc:145] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1746712856.538817 3243293 buffer_comparator.cc:145] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.538820 3243293 buffer_comparator.cc:145] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1746712856.538823 3243293 buffer_comparator.cc:145] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1746712856.538826 3243293 buffer_comparator.cc:145] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1746712856.538830 3243293 buffer_comparator.cc:145] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1746712856.538833 3243293 buffer_comparator.cc:145] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1746712856.538836 3243293 buffer_comparator.cc:145] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-05-08 14:00:56.538848: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.542484 3243293 buffer_comparator.cc:145] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1746712856.542545 3243293 buffer_comparator.cc:145] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1746712856.542551 3243293 buffer_comparator.cc:145] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1746712856.542555 3243293 buffer_comparator.cc:145] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1746712856.542558 3243293 buffer_comparator.cc:145] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1746712856.542561 3243293 buffer_comparator.cc:145] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.542564 3243293 buffer_comparator.cc:145] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1746712856.542567 3243293 buffer_comparator.cc:145] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1746712856.542577 3243293 buffer_comparator.cc:145] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1746712856.542580 3243293 buffer_comparator.cc:145] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-05-08 14:00:56.542592: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.546324 3243293 buffer_comparator.cc:145] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1746712856.546382 3243293 buffer_comparator.cc:145] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1746712856.546388 3243293 buffer_comparator.cc:145] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1746712856.546391 3243293 buffer_comparator.cc:145] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1746712856.546394 3243293 buffer_comparator.cc:145] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.546397 3243293 buffer_comparator.cc:145] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1746712856.546401 3243293 buffer_comparator.cc:145] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1746712856.546404 3243293 buffer_comparator.cc:145] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1746712856.546407 3243293 buffer_comparator.cc:145] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1746712856.546410 3243293 buffer_comparator.cc:145] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-05-08 14:00:56.546423: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.550049 3243293 buffer_comparator.cc:145] Difference at 0: 1084.56, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1746712856.550110 3243293 buffer_comparator.cc:145] Difference at 1: 1350.61, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1746712856.550114 3243293 buffer_comparator.cc:145] Difference at 2: 2009.8, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1746712856.550117 3243293 buffer_comparator.cc:145] Difference at 3: 1768.04, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1746712856.550120 3243293 buffer_comparator.cc:145] Difference at 4: 1240.61, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712856.550123 3243293 buffer_comparator.cc:145] Difference at 6: 1407.03, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712856.550127 3243293 buffer_comparator.cc:145] Difference at 7: 1138.83, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1746712856.550130 3243293 buffer_comparator.cc:145] Difference at 8: 1417.44, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.550132 3243293 buffer_comparator.cc:145] Difference at 9: 2084.44, expected 1833.77</span></span>
<span class="line"><span>E0000 00:00:1746712856.550135 3243293 buffer_comparator.cc:145] Difference at 10: 1844.73, expected 1592.38</span></span>
<span class="line"><span>2025-05-08 14:00:56.550148: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.553915 3243293 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712856.553977 3243293 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1746712856.553981 3243293 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.553984 3243293 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712856.553987 3243293 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1746712856.553990 3243293 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1746712856.553993 3243293 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712856.553996 3243293 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.553999 3243293 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1746712856.554002 3243293 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-08 14:00:56.554021: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.557688 3243293 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712856.557750 3243293 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1746712856.557753 3243293 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.557756 3243293 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712856.557759 3243293 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1746712856.557763 3243293 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1746712856.557766 3243293 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712856.557769 3243293 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.557772 3243293 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1746712856.557775 3243293 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-08 14:00:56.557786: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.561443 3243293 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712856.561503 3243293 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1746712856.561507 3243293 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.561510 3243293 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712856.561513 3243293 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1746712856.561516 3243293 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1746712856.561519 3243293 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712856.561522 3243293 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.561525 3243293 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1746712856.561528 3243293 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-08 14:00:56.561540: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.565181 3243293 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712856.565245 3243293 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1746712856.565249 3243293 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.565252 3243293 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712856.565255 3243293 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1746712856.565258 3243293 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1746712856.565261 3243293 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712856.565264 3243293 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.565267 3243293 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1746712856.565270 3243293 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-08 14:00:56.565282: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.568971 3243293 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712856.569034 3243293 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1746712856.569037 3243293 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.569040 3243293 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712856.569043 3243293 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1746712856.569046 3243293 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1746712856.569050 3243293 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712856.569053 3243293 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.569056 3243293 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1746712856.569059 3243293 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-08 14:00:56.569070: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.572698 3243293 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712856.572759 3243293 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1746712856.572763 3243293 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.572766 3243293 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712856.572769 3243293 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1746712856.572772 3243293 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1746712856.572775 3243293 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712856.572778 3243293 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.572781 3243293 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1746712856.572784 3243293 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-08 14:00:56.572796: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.576460 3243293 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712856.576523 3243293 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1746712856.576526 3243293 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.576529 3243293 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712856.576532 3243293 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1746712856.576535 3243293 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1746712856.576538 3243293 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712856.576541 3243293 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.576544 3243293 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1746712856.576547 3243293 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-08 14:00:56.576559: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.580167 3243293 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1746712856.580235 3243293 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1746712856.580239 3243293 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.580242 3243293 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1746712856.580245 3243293 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1746712856.580248 3243293 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1746712856.580251 3243293 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1746712856.580254 3243293 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1746712856.580257 3243293 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1746712856.580260 3243293 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-08 14:00:56.580272: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.583915 3243293 buffer_comparator.cc:145] Difference at 0: 1100.47, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1746712856.583973 3243293 buffer_comparator.cc:145] Difference at 1: 1361.33, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1746712856.583977 3243293 buffer_comparator.cc:145] Difference at 2: 2059.82, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1746712856.583980 3243293 buffer_comparator.cc:145] Difference at 3: 1808.05, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1746712856.583984 3243293 buffer_comparator.cc:145] Difference at 4: 1265.06, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712856.583987 3243293 buffer_comparator.cc:145] Difference at 5: 1986, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.583990 3243293 buffer_comparator.cc:145] Difference at 6: 1409.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712856.583993 3243293 buffer_comparator.cc:145] Difference at 7: 1173.38, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1746712856.583997 3243293 buffer_comparator.cc:145] Difference at 8: 1420.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.584000 3243293 buffer_comparator.cc:145] Difference at 9: 2114.57, expected 1833.77</span></span>
<span class="line"><span>2025-05-08 14:00:56.584012: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.587631 3243293 buffer_comparator.cc:145] Difference at 0: 1100.47, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1746712856.587691 3243293 buffer_comparator.cc:145] Difference at 1: 1361.33, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1746712856.587695 3243293 buffer_comparator.cc:145] Difference at 2: 2059.82, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1746712856.587698 3243293 buffer_comparator.cc:145] Difference at 3: 1808.05, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1746712856.587701 3243293 buffer_comparator.cc:145] Difference at 4: 1265.06, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712856.587704 3243293 buffer_comparator.cc:145] Difference at 5: 1986, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.587707 3243293 buffer_comparator.cc:145] Difference at 6: 1409.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712856.587710 3243293 buffer_comparator.cc:145] Difference at 7: 1173.38, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1746712856.587713 3243293 buffer_comparator.cc:145] Difference at 8: 1420.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.587716 3243293 buffer_comparator.cc:145] Difference at 9: 2114.57, expected 1833.77</span></span>
<span class="line"><span>2025-05-08 14:00:56.587729: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.591315 3243293 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1746712856.591384 3243293 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1746712856.591388 3243293 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1746712856.591397 3243293 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1746712856.591400 3243293 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712856.591403 3243293 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.591406 3243293 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712856.591409 3243293 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1746712856.591412 3243293 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.591415 3243293 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-08 14:00:56.591427: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.595086 3243293 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1746712856.595149 3243293 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1746712856.595152 3243293 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1746712856.595155 3243293 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1746712856.595158 3243293 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712856.595161 3243293 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.595164 3243293 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712856.595167 3243293 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1746712856.595170 3243293 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.595173 3243293 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-08 14:00:56.595185: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.598880 3243293 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1746712856.598943 3243293 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1746712856.598947 3243293 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1746712856.598950 3243293 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1746712856.598953 3243293 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712856.598956 3243293 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.598959 3243293 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712856.598962 3243293 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1746712856.598965 3243293 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.598968 3243293 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-08 14:00:56.598980: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.602690 3243293 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1746712856.602754 3243293 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1746712856.602761 3243293 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1746712856.602764 3243293 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1746712856.602767 3243293 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712856.602770 3243293 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.602779 3243293 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712856.602782 3243293 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1746712856.602785 3243293 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.602788 3243293 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-08 14:00:56.602799: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.606511 3243293 buffer_comparator.cc:145] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1746712856.606573 3243293 buffer_comparator.cc:145] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.606579 3243293 buffer_comparator.cc:145] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1746712856.606582 3243293 buffer_comparator.cc:145] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1746712856.606585 3243293 buffer_comparator.cc:145] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1746712856.606589 3243293 buffer_comparator.cc:145] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1746712856.606592 3243293 buffer_comparator.cc:145] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1746712856.606595 3243293 buffer_comparator.cc:145] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1746712856.606598 3243293 buffer_comparator.cc:145] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1746712856.606601 3243293 buffer_comparator.cc:145] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-05-08 14:00:56.606613: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.610264 3243293 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1746712856.610336 3243293 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1746712856.610341 3243293 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1746712856.610344 3243293 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1746712856.610348 3243293 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1746712856.610351 3243293 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1746712856.610355 3243293 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1746712856.610358 3243293 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.77</span></span>
<span class="line"><span>E0000 00:00:1746712856.610362 3243293 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.09</span></span>
<span class="line"><span>E0000 00:00:1746712856.610365 3243293 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.62</span></span>
<span class="line"><span>2025-05-08 14:00:56.610377: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.614041 3243293 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1746712856.614102 3243293 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1746712856.614106 3243293 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1746712856.614110 3243293 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1746712856.614113 3243293 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1746712856.614117 3243293 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1746712856.614120 3243293 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1746712856.614124 3243293 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.77</span></span>
<span class="line"><span>E0000 00:00:1746712856.614134 3243293 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.09</span></span>
<span class="line"><span>E0000 00:00:1746712856.614138 3243293 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.62</span></span>
<span class="line"><span>2025-05-08 14:00:56.614150: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.618021 3243293 buffer_comparator.cc:145] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1746712856.618089 3243293 buffer_comparator.cc:145] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.618092 3243293 buffer_comparator.cc:145] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1746712856.618096 3243293 buffer_comparator.cc:145] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1746712856.618099 3243293 buffer_comparator.cc:145] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1746712856.618102 3243293 buffer_comparator.cc:145] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1746712856.618105 3243293 buffer_comparator.cc:145] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1746712856.618108 3243293 buffer_comparator.cc:145] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1746712856.618111 3243293 buffer_comparator.cc:145] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1746712856.618114 3243293 buffer_comparator.cc:145] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-05-08 14:00:56.618127: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712856.621759 3243293 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1746712856.621833 3243293 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1746712856.621836 3243293 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1746712856.621839 3243293 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1746712856.621843 3243293 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1746712856.621846 3243293 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1746712856.621849 3243293 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1746712856.621852 3243293 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1746712856.621855 3243293 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1746712856.621858 3243293 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.77</span></span>
<span class="line"><span>2025-05-08 14:00:56.621870: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-05-08 14:00:59.150216: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 14:00:59.748309: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 14:00:59.854483: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-08 14:01:00.338949: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1746712860.344410 3243293 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1746712860.344452 3243293 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1746712860.344457 3243293 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1746712860.344461 3243293 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1746712860.344464 3243293 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1746712860.344467 3243293 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1746712860.344470 3243293 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1746712860.344473 3243293 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1746712860.344476 3243293 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1746712860.344478 3243293 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-05-08 14:01:00.344489: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.346798 3243293 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1746712860.346813 3243293 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1746712860.346816 3243293 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1746712860.346819 3243293 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1746712860.346822 3243293 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1746712860.346825 3243293 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1746712860.346828 3243293 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1746712860.346831 3243293 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1746712860.346834 3243293 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1746712860.346837 3243293 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-05-08 14:01:00.346842: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.349143 3243293 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1746712860.349158 3243293 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1746712860.349161 3243293 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1746712860.349164 3243293 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1746712860.349167 3243293 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1746712860.349170 3243293 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1746712860.349173 3243293 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1746712860.349176 3243293 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1746712860.349179 3243293 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1746712860.349182 3243293 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-05-08 14:01:00.349186: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.351626 3243293 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1746712860.351641 3243293 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1746712860.351644 3243293 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1746712860.351647 3243293 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1746712860.351650 3243293 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1746712860.351653 3243293 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1746712860.351657 3243293 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1746712860.351660 3243293 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1746712860.351663 3243293 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1746712860.351666 3243293 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-05-08 14:01:00.351671: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.354010 3243293 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1746712860.354025 3243293 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1746712860.354029 3243293 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1746712860.354032 3243293 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1746712860.354035 3243293 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1746712860.354038 3243293 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1746712860.354040 3243293 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1746712860.354043 3243293 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1746712860.354046 3243293 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1746712860.354049 3243293 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-05-08 14:01:00.354054: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.356294 3243293 buffer_comparator.cc:145] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1746712860.356308 3243293 buffer_comparator.cc:145] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1746712860.356311 3243293 buffer_comparator.cc:145] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1746712860.356314 3243293 buffer_comparator.cc:145] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1746712860.356317 3243293 buffer_comparator.cc:145] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1746712860.356320 3243293 buffer_comparator.cc:145] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1746712860.356323 3243293 buffer_comparator.cc:145] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1746712860.356326 3243293 buffer_comparator.cc:145] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1746712860.356329 3243293 buffer_comparator.cc:145] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1746712860.356332 3243293 buffer_comparator.cc:145] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-05-08 14:01:00.356337: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.358684 3243293 buffer_comparator.cc:145] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1746712860.358698 3243293 buffer_comparator.cc:145] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1746712860.358701 3243293 buffer_comparator.cc:145] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1746712860.358704 3243293 buffer_comparator.cc:145] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1746712860.358707 3243293 buffer_comparator.cc:145] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1746712860.358710 3243293 buffer_comparator.cc:145] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1746712860.358713 3243293 buffer_comparator.cc:145] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1746712860.358716 3243293 buffer_comparator.cc:145] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1746712860.358719 3243293 buffer_comparator.cc:145] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1746712860.358722 3243293 buffer_comparator.cc:145] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-05-08 14:01:00.358727: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.360971 3243293 buffer_comparator.cc:145] Difference at 0: 903.336, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1746712860.360986 3243293 buffer_comparator.cc:145] Difference at 1: 1271.45, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1746712860.360989 3243293 buffer_comparator.cc:145] Difference at 2: 1218.72, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1746712860.360992 3243293 buffer_comparator.cc:145] Difference at 3: 1830.29, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1746712860.360995 3243293 buffer_comparator.cc:145] Difference at 4: 1832.52, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1746712860.360998 3243293 buffer_comparator.cc:145] Difference at 5: 1505.57, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1746712860.361001 3243293 buffer_comparator.cc:145] Difference at 6: 1003.78, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1746712860.361005 3243293 buffer_comparator.cc:145] Difference at 7: 895.724, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1746712860.361008 3243293 buffer_comparator.cc:145] Difference at 8: 1254.14, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1746712860.361011 3243293 buffer_comparator.cc:145] Difference at 9: 1207.96, expected 1052.46</span></span>
<span class="line"><span>2025-05-08 14:01:00.361015: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.363395 3243293 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1746712860.363410 3243293 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1746712860.363413 3243293 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1746712860.363416 3243293 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1746712860.363419 3243293 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1746712860.363422 3243293 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1746712860.363425 3243293 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1746712860.363428 3243293 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1746712860.363431 3243293 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1746712860.363434 3243293 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-08 14:01:00.363439: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.365708 3243293 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1746712860.365723 3243293 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1746712860.365726 3243293 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1746712860.365729 3243293 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1746712860.365732 3243293 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1746712860.365735 3243293 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1746712860.365738 3243293 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1746712860.365741 3243293 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1746712860.365744 3243293 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1746712860.365747 3243293 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-08 14:01:00.365751: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.367997 3243293 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1746712860.368012 3243293 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1746712860.368016 3243293 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1746712860.368019 3243293 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1746712860.368022 3243293 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1746712860.368025 3243293 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1746712860.368028 3243293 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1746712860.368031 3243293 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1746712860.368034 3243293 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1746712860.368037 3243293 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-08 14:01:00.368041: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.370264 3243293 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1746712860.370279 3243293 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1746712860.370282 3243293 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1746712860.370293 3243293 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1746712860.370296 3243293 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1746712860.370299 3243293 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1746712860.370302 3243293 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1746712860.370305 3243293 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1746712860.370308 3243293 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1746712860.370310 3243293 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-08 14:01:00.370315: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.372558 3243293 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1746712860.372573 3243293 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1746712860.372576 3243293 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1746712860.372579 3243293 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1746712860.372582 3243293 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1746712860.372585 3243293 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1746712860.372588 3243293 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1746712860.372591 3243293 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1746712860.372594 3243293 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1746712860.372597 3243293 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-08 14:01:00.372602: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.374871 3243293 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1746712860.374887 3243293 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1746712860.374890 3243293 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1746712860.374893 3243293 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1746712860.374896 3243293 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1746712860.374899 3243293 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1746712860.374903 3243293 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1746712860.374906 3243293 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1746712860.374909 3243293 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1746712860.374912 3243293 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-08 14:01:00.374917: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.377188 3243293 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1746712860.377203 3243293 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1746712860.377206 3243293 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1746712860.377209 3243293 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1746712860.377212 3243293 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1746712860.377215 3243293 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1746712860.377218 3243293 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1746712860.377221 3243293 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1746712860.377224 3243293 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1746712860.377227 3243293 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-08 14:01:00.377231: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.379500 3243293 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1746712860.379514 3243293 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1746712860.379517 3243293 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1746712860.379520 3243293 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1746712860.379523 3243293 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1746712860.379526 3243293 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1746712860.379529 3243293 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1746712860.379532 3243293 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1746712860.379535 3243293 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1746712860.379538 3243293 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-05-08 14:01:00.379542: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.381818 3243293 buffer_comparator.cc:145] Difference at 0: 876.475, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1746712860.381832 3243293 buffer_comparator.cc:145] Difference at 1: 1292.4, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1746712860.381835 3243293 buffer_comparator.cc:145] Difference at 2: 1239.8, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1746712860.381838 3243293 buffer_comparator.cc:145] Difference at 3: 1830.71, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1746712860.381841 3243293 buffer_comparator.cc:145] Difference at 4: 1857.47, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1746712860.381844 3243293 buffer_comparator.cc:145] Difference at 5: 1551.94, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1746712860.381847 3243293 buffer_comparator.cc:145] Difference at 6: 1022.45, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1746712860.381850 3243293 buffer_comparator.cc:145] Difference at 8: 1214.29, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1746712860.381853 3243293 buffer_comparator.cc:145] Difference at 9: 1173.34, expected 1052.46</span></span>
<span class="line"><span>E0000 00:00:1746712860.381857 3243293 buffer_comparator.cc:145] Difference at 10: 1732.94, expected 1556.04</span></span>
<span class="line"><span>2025-05-08 14:01:00.381862: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.384120 3243293 buffer_comparator.cc:145] Difference at 0: 876.475, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1746712860.384135 3243293 buffer_comparator.cc:145] Difference at 1: 1292.4, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1746712860.384138 3243293 buffer_comparator.cc:145] Difference at 2: 1239.8, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1746712860.384141 3243293 buffer_comparator.cc:145] Difference at 3: 1830.71, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1746712860.384144 3243293 buffer_comparator.cc:145] Difference at 4: 1857.47, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1746712860.384148 3243293 buffer_comparator.cc:145] Difference at 5: 1551.94, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1746712860.384151 3243293 buffer_comparator.cc:145] Difference at 6: 1022.45, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1746712860.384154 3243293 buffer_comparator.cc:145] Difference at 8: 1214.29, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1746712860.384157 3243293 buffer_comparator.cc:145] Difference at 9: 1173.34, expected 1052.46</span></span>
<span class="line"><span>E0000 00:00:1746712860.384160 3243293 buffer_comparator.cc:145] Difference at 10: 1732.94, expected 1556.04</span></span>
<span class="line"><span>2025-05-08 14:01:00.384164: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.386388 3243293 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1746712860.386402 3243293 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1746712860.386406 3243293 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1746712860.386410 3243293 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1746712860.386413 3243293 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1746712860.386416 3243293 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1746712860.386419 3243293 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1746712860.386422 3243293 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1746712860.386426 3243293 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1746712860.386429 3243293 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-08 14:01:00.386434: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.388713 3243293 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1746712860.388727 3243293 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1746712860.388730 3243293 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1746712860.388733 3243293 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1746712860.388736 3243293 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1746712860.388739 3243293 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1746712860.388742 3243293 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1746712860.388745 3243293 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1746712860.388748 3243293 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1746712860.388751 3243293 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-08 14:01:00.388756: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.391032 3243293 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1746712860.391046 3243293 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1746712860.391050 3243293 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1746712860.391053 3243293 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1746712860.391056 3243293 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1746712860.391059 3243293 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1746712860.391062 3243293 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1746712860.391065 3243293 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1746712860.391068 3243293 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1746712860.391071 3243293 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-08 14:01:00.391076: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.393395 3243293 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1746712860.393409 3243293 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1746712860.393412 3243293 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1746712860.393415 3243293 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1746712860.393418 3243293 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1746712860.393421 3243293 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1746712860.393424 3243293 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1746712860.393427 3243293 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1746712860.393430 3243293 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1746712860.393433 3243293 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-08 14:01:00.393438: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.395724 3243293 buffer_comparator.cc:145] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1746712860.395738 3243293 buffer_comparator.cc:145] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1746712860.395741 3243293 buffer_comparator.cc:145] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1746712860.395744 3243293 buffer_comparator.cc:145] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1746712860.395747 3243293 buffer_comparator.cc:145] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1746712860.395750 3243293 buffer_comparator.cc:145] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1746712860.395753 3243293 buffer_comparator.cc:145] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1746712860.395756 3243293 buffer_comparator.cc:145] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1746712860.395758 3243293 buffer_comparator.cc:145] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1746712860.395761 3243293 buffer_comparator.cc:145] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-05-08 14:01:00.395766: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746712860.398032 3243293 buffer_comparator.cc:145] Difference at 540: 1203.79, expected 1064.85</span></span>
<span class="line"><span>Test Loss: 71.846130	Test Acc: 10.5000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
