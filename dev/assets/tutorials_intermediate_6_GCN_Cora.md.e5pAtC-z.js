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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-04-25 01:54:23.658841: I external/xla/xla/service/service.cc:152] XLA service 0x5b94360 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-04-25 01:54:23.659201: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1745546063.660716 1595206 se_gpu_pjrt_client.cc:999] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1745546063.661146 1595206 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1745546063.661362 1595206 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1745546063.675144 1595206 cuda_dnn.cc:527] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-13/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-13/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>2025-04-25 01:55:32.672810: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 104 bytes spill stores, 104 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:32.784507: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:32.854139: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 48 bytes spill stores, 48 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:33.052790: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 996 bytes spill stores, 968 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:33.587316: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 272 bytes spill stores, 272 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:33.684726: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 284 bytes spill stores, 284 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:33.816331: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 1212 bytes spill stores, 948 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:33.823722: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:34.770425: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 52 bytes spill stores, 52 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:34.910466: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 980 bytes spill stores, 976 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:34.957538: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 612 bytes spill stores, 616 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:35.099383: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:36.059675: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 32 bytes spill stores, 32 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:36.095683: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 360 bytes spill stores, 356 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:36.383653: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:36.890175: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:37.015900: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 292 bytes spill stores, 292 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:37.266141: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:37.725495: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:55:37.769923: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1745546137.924094 1595206 buffer_comparator.cc:145] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1745546137.924893 1595206 buffer_comparator.cc:145] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1745546137.924904 1595206 buffer_comparator.cc:145] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1745546137.924911 1595206 buffer_comparator.cc:145] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1745546137.924918 1595206 buffer_comparator.cc:145] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1745546137.924924 1595206 buffer_comparator.cc:145] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1745546137.924931 1595206 buffer_comparator.cc:145] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1745546137.924937 1595206 buffer_comparator.cc:145] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1745546137.924944 1595206 buffer_comparator.cc:145] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1745546137.924950 1595206 buffer_comparator.cc:145] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-04-25 01:55:37.924965: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.928160 1595206 buffer_comparator.cc:145] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1745546137.928186 1595206 buffer_comparator.cc:145] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1745546137.928194 1595206 buffer_comparator.cc:145] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1745546137.928201 1595206 buffer_comparator.cc:145] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1745546137.928208 1595206 buffer_comparator.cc:145] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1745546137.928214 1595206 buffer_comparator.cc:145] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1745546137.928221 1595206 buffer_comparator.cc:145] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1745546137.928227 1595206 buffer_comparator.cc:145] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1745546137.928234 1595206 buffer_comparator.cc:145] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1745546137.928241 1595206 buffer_comparator.cc:145] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-04-25 01:55:37.928251: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.931130 1595206 buffer_comparator.cc:145] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1745546137.931155 1595206 buffer_comparator.cc:145] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1745546137.931163 1595206 buffer_comparator.cc:145] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1745546137.931170 1595206 buffer_comparator.cc:145] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1745546137.931176 1595206 buffer_comparator.cc:145] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1745546137.931183 1595206 buffer_comparator.cc:145] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1745546137.931190 1595206 buffer_comparator.cc:145] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1745546137.931196 1595206 buffer_comparator.cc:145] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1745546137.931203 1595206 buffer_comparator.cc:145] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1745546137.931209 1595206 buffer_comparator.cc:145] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-04-25 01:55:37.931223: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.934116 1595206 buffer_comparator.cc:145] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1745546137.934145 1595206 buffer_comparator.cc:145] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1745546137.934153 1595206 buffer_comparator.cc:145] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1745546137.934160 1595206 buffer_comparator.cc:145] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1745546137.934166 1595206 buffer_comparator.cc:145] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1745546137.934173 1595206 buffer_comparator.cc:145] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1745546137.934179 1595206 buffer_comparator.cc:145] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1745546137.934186 1595206 buffer_comparator.cc:145] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1745546137.934193 1595206 buffer_comparator.cc:145] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1745546137.934199 1595206 buffer_comparator.cc:145] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-04-25 01:55:37.934209: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.936788 1595206 buffer_comparator.cc:145] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1745546137.936799 1595206 buffer_comparator.cc:145] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1745546137.936803 1595206 buffer_comparator.cc:145] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1745546137.936806 1595206 buffer_comparator.cc:145] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1745546137.936809 1595206 buffer_comparator.cc:145] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1745546137.936812 1595206 buffer_comparator.cc:145] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1745546137.936814 1595206 buffer_comparator.cc:145] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1745546137.936817 1595206 buffer_comparator.cc:145] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1745546137.936820 1595206 buffer_comparator.cc:145] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1745546137.936823 1595206 buffer_comparator.cc:145] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-04-25 01:55:37.936828: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.939306 1595206 buffer_comparator.cc:145] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1745546137.939318 1595206 buffer_comparator.cc:145] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1745546137.939321 1595206 buffer_comparator.cc:145] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1745546137.939324 1595206 buffer_comparator.cc:145] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1745546137.939327 1595206 buffer_comparator.cc:145] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1745546137.939330 1595206 buffer_comparator.cc:145] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1745546137.939333 1595206 buffer_comparator.cc:145] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1745546137.939336 1595206 buffer_comparator.cc:145] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1745546137.939339 1595206 buffer_comparator.cc:145] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1745546137.939342 1595206 buffer_comparator.cc:145] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-04-25 01:55:37.939347: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.941860 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.941871 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.941876 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.941879 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.941882 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.941885 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.941888 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.941891 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.941894 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.941897 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.941901: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.944385 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.944396 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.944400 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.944403 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.944406 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.944409 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.944412 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.944414 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.944417 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.944420 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.944425: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.946897 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.946908 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.946911 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.946915 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.946918 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.946920 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.946923 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.946926 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.946929 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.946932 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.946937: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.949422 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.949433 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.949436 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.949440 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.949442 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.949445 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.949463 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.949466 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.949468 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.949471 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.949476: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.951954 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.951965 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.951968 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.951972 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.951974 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.951977 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.951980 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.951983 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.951986 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.951989 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.951994: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.954471 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.954483 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.954486 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.954489 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.954492 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.954495 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.954498 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.954501 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.954504 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.954507 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.954512: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.956975 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.956987 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.956990 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.956993 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.956996 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.956999 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.957002 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.957005 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.957008 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.957011 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.957018: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.959488 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.959500 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.959503 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.959506 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.959509 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.959512 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.959515 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.959518 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.959521 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.959524 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.959528: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.961987 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.961998 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.962001 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.962004 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.962007 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.962010 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.962013 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.962016 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.962019 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.962022 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.962027: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.964503 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.964514 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.964518 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.964521 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.964524 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.964527 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.964530 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.964533 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.964536 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.964539 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.964544: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.967032 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.967043 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.967049 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.967052 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.967055 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.967058 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.967061 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.967063 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.967066 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.967069 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.967074: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.969544 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.969555 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.969559 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.969562 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.969565 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.969568 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.969571 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.969574 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.969577 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.969580 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.969584: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.972046 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.972058 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.972061 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.972064 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.972067 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.972070 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.972073 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.972076 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.972079 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.972082 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.972087: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.974549 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.974560 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.974564 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.974567 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.974570 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.974573 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.974578 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.974581 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.974584 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.974586 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.974591: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.977062 1595206 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1745546137.977073 1595206 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1745546137.977076 1595206 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1745546137.977079 1595206 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1745546137.977082 1595206 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1745546137.977085 1595206 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1745546137.977088 1595206 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1745546137.977091 1595206 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1745546137.977094 1595206 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1745546137.977097 1595206 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-25 01:55:37.977101: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.979611 1595206 buffer_comparator.cc:145] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1745546137.979623 1595206 buffer_comparator.cc:145] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1745546137.979626 1595206 buffer_comparator.cc:145] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1745546137.979629 1595206 buffer_comparator.cc:145] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1745546137.979632 1595206 buffer_comparator.cc:145] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1745546137.979635 1595206 buffer_comparator.cc:145] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1745546137.979638 1595206 buffer_comparator.cc:145] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1745546137.979641 1595206 buffer_comparator.cc:145] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1745546137.979644 1595206 buffer_comparator.cc:145] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1745546137.979647 1595206 buffer_comparator.cc:145] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-04-25 01:55:37.979652: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.982136 1595206 buffer_comparator.cc:145] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1745546137.982148 1595206 buffer_comparator.cc:145] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1745546137.982151 1595206 buffer_comparator.cc:145] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1745546137.982154 1595206 buffer_comparator.cc:145] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1745546137.982157 1595206 buffer_comparator.cc:145] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1745546137.982160 1595206 buffer_comparator.cc:145] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1745546137.982163 1595206 buffer_comparator.cc:145] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1745546137.982166 1595206 buffer_comparator.cc:145] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1745546137.982169 1595206 buffer_comparator.cc:145] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1745546137.982172 1595206 buffer_comparator.cc:145] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-04-25 01:55:37.982179: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.984671 1595206 buffer_comparator.cc:145] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1745546137.984682 1595206 buffer_comparator.cc:145] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1745546137.984685 1595206 buffer_comparator.cc:145] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1745546137.984688 1595206 buffer_comparator.cc:145] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1745546137.984691 1595206 buffer_comparator.cc:145] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1745546137.984694 1595206 buffer_comparator.cc:145] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1745546137.984697 1595206 buffer_comparator.cc:145] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1745546137.984700 1595206 buffer_comparator.cc:145] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1745546137.984703 1595206 buffer_comparator.cc:145] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1745546137.984706 1595206 buffer_comparator.cc:145] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-04-25 01:55:37.984711: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.987201 1595206 buffer_comparator.cc:145] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1745546137.987213 1595206 buffer_comparator.cc:145] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1745546137.987216 1595206 buffer_comparator.cc:145] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1745546137.987219 1595206 buffer_comparator.cc:145] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1745546137.987222 1595206 buffer_comparator.cc:145] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1745546137.987225 1595206 buffer_comparator.cc:145] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1745546137.987228 1595206 buffer_comparator.cc:145] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1745546137.987231 1595206 buffer_comparator.cc:145] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1745546137.987234 1595206 buffer_comparator.cc:145] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1745546137.987237 1595206 buffer_comparator.cc:145] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-04-25 01:55:37.987242: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.989739 1595206 buffer_comparator.cc:145] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1745546137.989750 1595206 buffer_comparator.cc:145] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1745546137.989754 1595206 buffer_comparator.cc:145] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1745546137.989757 1595206 buffer_comparator.cc:145] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1745546137.989760 1595206 buffer_comparator.cc:145] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1745546137.989762 1595206 buffer_comparator.cc:145] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1745546137.989765 1595206 buffer_comparator.cc:145] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1745546137.989768 1595206 buffer_comparator.cc:145] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1745546137.989771 1595206 buffer_comparator.cc:145] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1745546137.989774 1595206 buffer_comparator.cc:145] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-04-25 01:55:37.989779: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.992307 1595206 buffer_comparator.cc:145] Difference at 0: 21.7575, expected 19.3855</span></span>
<span class="line"><span>E0000 00:00:1745546137.992319 1595206 buffer_comparator.cc:145] Difference at 3: 14.319, expected 17.5973</span></span>
<span class="line"><span>E0000 00:00:1745546137.992325 1595206 buffer_comparator.cc:145] Difference at 9: 14.8402, expected 16.6531</span></span>
<span class="line"><span>E0000 00:00:1745546137.992328 1595206 buffer_comparator.cc:145] Difference at 20: 13.7726, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1745546137.992331 1595206 buffer_comparator.cc:145] Difference at 26: 15.2226, expected 17.2903</span></span>
<span class="line"><span>E0000 00:00:1745546137.992334 1595206 buffer_comparator.cc:145] Difference at 27: 18.7304, expected 16.5311</span></span>
<span class="line"><span>E0000 00:00:1745546137.992338 1595206 buffer_comparator.cc:145] Difference at 31: 14.8392, expected 16.8073</span></span>
<span class="line"><span>E0000 00:00:1745546137.992341 1595206 buffer_comparator.cc:145] Difference at 33: 12.4405, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1745546137.992344 1595206 buffer_comparator.cc:145] Difference at 39: 19.1851, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1745546137.992347 1595206 buffer_comparator.cc:145] Difference at 41: 17.4688, expected 20.3484</span></span>
<span class="line"><span>2025-04-25 01:55:37.992351: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546137.994834 1595206 buffer_comparator.cc:145] Difference at 128: 0.278558, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1745546137.994846 1595206 buffer_comparator.cc:145] Difference at 129: 0.766922, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1745546137.994849 1595206 buffer_comparator.cc:145] Difference at 130: 0.824242, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1745546137.994852 1595206 buffer_comparator.cc:145] Difference at 131: 0.934478, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1745546137.994855 1595206 buffer_comparator.cc:145] Difference at 132: 0.683298, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1745546137.994858 1595206 buffer_comparator.cc:145] Difference at 133: 0.107889, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1745546137.994862 1595206 buffer_comparator.cc:145] Difference at 134: 0.716831, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1745546137.994865 1595206 buffer_comparator.cc:145] Difference at 135: 0.182228, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1745546137.994868 1595206 buffer_comparator.cc:145] Difference at 136: 0.780881, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1745546137.994871 1595206 buffer_comparator.cc:145] Difference at 137: 0.0990953, expected 18.5916</span></span>
<span class="line"><span>2025-04-25 01:55:37.994876: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.002597 1595206 buffer_comparator.cc:145] Difference at 16: 0, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1745546138.002609 1595206 buffer_comparator.cc:145] Difference at 17: 0, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1745546138.002613 1595206 buffer_comparator.cc:145] Difference at 18: 0, expected 358.37</span></span>
<span class="line"><span>E0000 00:00:1745546138.002616 1595206 buffer_comparator.cc:145] Difference at 19: 0, expected 346.727</span></span>
<span class="line"><span>E0000 00:00:1745546138.002619 1595206 buffer_comparator.cc:145] Difference at 20: 0, expected 356.216</span></span>
<span class="line"><span>E0000 00:00:1745546138.002622 1595206 buffer_comparator.cc:145] Difference at 21: 0, expected 358.962</span></span>
<span class="line"><span>E0000 00:00:1745546138.002625 1595206 buffer_comparator.cc:145] Difference at 22: 0, expected 359.155</span></span>
<span class="line"><span>E0000 00:00:1745546138.002628 1595206 buffer_comparator.cc:145] Difference at 23: 0, expected 360.559</span></span>
<span class="line"><span>E0000 00:00:1745546138.002631 1595206 buffer_comparator.cc:145] Difference at 24: 0, expected 371.461</span></span>
<span class="line"><span>E0000 00:00:1745546138.002634 1595206 buffer_comparator.cc:145] Difference at 25: 0, expected 357.082</span></span>
<span class="line"><span>2025-04-25 01:55:38.002639: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.006253 1595206 buffer_comparator.cc:145] Difference at 16: 0, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1745546138.006265 1595206 buffer_comparator.cc:145] Difference at 17: 0, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1745546138.006269 1595206 buffer_comparator.cc:145] Difference at 18: 0, expected 358.37</span></span>
<span class="line"><span>E0000 00:00:1745546138.006272 1595206 buffer_comparator.cc:145] Difference at 19: 0, expected 346.727</span></span>
<span class="line"><span>E0000 00:00:1745546138.006275 1595206 buffer_comparator.cc:145] Difference at 20: 0, expected 356.216</span></span>
<span class="line"><span>E0000 00:00:1745546138.006280 1595206 buffer_comparator.cc:145] Difference at 21: 0, expected 358.962</span></span>
<span class="line"><span>E0000 00:00:1745546138.006283 1595206 buffer_comparator.cc:145] Difference at 22: 0, expected 359.155</span></span>
<span class="line"><span>E0000 00:00:1745546138.006286 1595206 buffer_comparator.cc:145] Difference at 23: 0, expected 360.559</span></span>
<span class="line"><span>E0000 00:00:1745546138.006289 1595206 buffer_comparator.cc:145] Difference at 24: 0, expected 371.461</span></span>
<span class="line"><span>E0000 00:00:1745546138.006292 1595206 buffer_comparator.cc:145] Difference at 25: 0, expected 357.082</span></span>
<span class="line"><span>2025-04-25 01:55:38.006297: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.010668 1595206 buffer_comparator.cc:145] Difference at 16: 0, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1745546138.010680 1595206 buffer_comparator.cc:145] Difference at 17: 0, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1745546138.010683 1595206 buffer_comparator.cc:145] Difference at 18: 0, expected 358.37</span></span>
<span class="line"><span>E0000 00:00:1745546138.010686 1595206 buffer_comparator.cc:145] Difference at 19: 0, expected 346.727</span></span>
<span class="line"><span>E0000 00:00:1745546138.010689 1595206 buffer_comparator.cc:145] Difference at 20: 0, expected 356.216</span></span>
<span class="line"><span>E0000 00:00:1745546138.010692 1595206 buffer_comparator.cc:145] Difference at 21: 0, expected 358.962</span></span>
<span class="line"><span>E0000 00:00:1745546138.010695 1595206 buffer_comparator.cc:145] Difference at 22: 0, expected 359.155</span></span>
<span class="line"><span>E0000 00:00:1745546138.010698 1595206 buffer_comparator.cc:145] Difference at 23: 0, expected 360.559</span></span>
<span class="line"><span>E0000 00:00:1745546138.010701 1595206 buffer_comparator.cc:145] Difference at 24: 0, expected 371.461</span></span>
<span class="line"><span>E0000 00:00:1745546138.010704 1595206 buffer_comparator.cc:145] Difference at 25: 0, expected 357.082</span></span>
<span class="line"><span>2025-04-25 01:55:38.010709: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.017309 1595206 buffer_comparator.cc:145] Difference at 16: 0, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1745546138.017321 1595206 buffer_comparator.cc:145] Difference at 17: 0, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1745546138.017324 1595206 buffer_comparator.cc:145] Difference at 18: 0, expected 358.37</span></span>
<span class="line"><span>E0000 00:00:1745546138.017327 1595206 buffer_comparator.cc:145] Difference at 19: 0, expected 346.727</span></span>
<span class="line"><span>E0000 00:00:1745546138.017330 1595206 buffer_comparator.cc:145] Difference at 20: 0, expected 356.216</span></span>
<span class="line"><span>E0000 00:00:1745546138.017333 1595206 buffer_comparator.cc:145] Difference at 21: 0, expected 358.962</span></span>
<span class="line"><span>E0000 00:00:1745546138.017336 1595206 buffer_comparator.cc:145] Difference at 22: 0, expected 359.155</span></span>
<span class="line"><span>E0000 00:00:1745546138.017339 1595206 buffer_comparator.cc:145] Difference at 23: 0, expected 360.559</span></span>
<span class="line"><span>E0000 00:00:1745546138.017342 1595206 buffer_comparator.cc:145] Difference at 24: 0, expected 371.461</span></span>
<span class="line"><span>E0000 00:00:1745546138.017345 1595206 buffer_comparator.cc:145] Difference at 25: 0, expected 357.082</span></span>
<span class="line"><span>2025-04-25 01:55:38.017350: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.020737 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.020749 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.020752 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.020755 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.020758 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.020761 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.020764 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.020767 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.020770 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.020775 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.020780: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.024029 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.024041 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.024044 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.024047 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.024050 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.024053 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.024056 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.024059 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.024062 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.024065 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.024070: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.031192 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.031203 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.031207 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.031210 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.031213 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.031216 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.031219 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.031222 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.031225 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.031228 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.031232: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.038575 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.038587 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.038590 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.038593 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.038596 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.038599 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.038602 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.038605 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.038608 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.038611 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.038616: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.042746 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.042760 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.042763 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.042766 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.042769 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.042772 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.042775 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.042778 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.042781 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.042783 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.042788: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.046071 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.046082 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.046085 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.046088 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.046091 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.046094 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.046097 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.046100 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.046103 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.046106 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.046111: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.050104 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.050118 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.050121 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.050124 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.050127 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.050130 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.050133 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.050136 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.050139 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.050142 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.050147: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.054397 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.054412 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.054415 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.054418 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.054421 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.054426 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.054429 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.054432 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.054435 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.054438 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.054443: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.058598 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.058610 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.058613 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.058616 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.058619 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.058622 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.058625 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.058628 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.058631 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.058634 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.058639: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.063571 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.063583 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.063586 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.063589 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.063592 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.063595 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.063598 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.063601 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.063604 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.063607 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.063612: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.068571 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.068583 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.068586 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.068589 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.068593 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.068596 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.068600 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.068603 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.068606 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.068612 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.068618: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.073632 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.073644 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.073647 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.073650 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.073653 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.073656 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.073659 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.073662 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.073665 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.073668 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.073673: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.078778 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.078790 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.078793 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.078796 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.078799 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.078802 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.078805 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.078808 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.078811 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.078814 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.078819: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.082171 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.082183 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.082186 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.082189 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.082192 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.082195 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.082198 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.082201 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.082204 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.082207 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.082212: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.087484 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.087499 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.087502 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.087505 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.087508 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.087511 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.087514 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.087517 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.087520 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.087523 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.087528: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.092540 1595206 buffer_comparator.cc:145] Difference at 256: 0, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1745546138.092554 1595206 buffer_comparator.cc:145] Difference at 257: 0, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1745546138.092557 1595206 buffer_comparator.cc:145] Difference at 258: 0, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1745546138.092560 1595206 buffer_comparator.cc:145] Difference at 259: 0, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1745546138.092563 1595206 buffer_comparator.cc:145] Difference at 260: 0, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1745546138.092566 1595206 buffer_comparator.cc:145] Difference at 261: 0, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1745546138.092569 1595206 buffer_comparator.cc:145] Difference at 262: 0, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1745546138.092572 1595206 buffer_comparator.cc:145] Difference at 263: 0, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1745546138.092575 1595206 buffer_comparator.cc:145] Difference at 264: 0, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1745546138.092578 1595206 buffer_comparator.cc:145] Difference at 265: 0, expected 350.299</span></span>
<span class="line"><span>2025-04-25 01:55:38.092582: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.095808 1595206 buffer_comparator.cc:145] Difference at 16: 0.446302, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1745546138.095820 1595206 buffer_comparator.cc:145] Difference at 17: 0.0661601, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1745546138.095824 1595206 buffer_comparator.cc:145] Difference at 18: 0.607784, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1745546138.095827 1595206 buffer_comparator.cc:145] Difference at 19: 0.275431, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1745546138.095830 1595206 buffer_comparator.cc:145] Difference at 20: 0.0248372, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1745546138.095834 1595206 buffer_comparator.cc:145] Difference at 21: 0.28182, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1745546138.095837 1595206 buffer_comparator.cc:145] Difference at 22: 0.98984, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1745546138.095840 1595206 buffer_comparator.cc:145] Difference at 23: 0.880066, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1745546138.095843 1595206 buffer_comparator.cc:145] Difference at 24: 0.869889, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1745546138.095846 1595206 buffer_comparator.cc:145] Difference at 25: 0.44433, expected 13.4166</span></span>
<span class="line"><span>2025-04-25 01:55:38.095850: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546138.098039 1595206 buffer_comparator.cc:145] Difference at 16: 0.446302, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1745546138.098056 1595206 buffer_comparator.cc:145] Difference at 17: 0.0661601, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1745546138.098059 1595206 buffer_comparator.cc:145] Difference at 18: 0.607784, expected 15.6849</span></span>
<span class="line"><span>Epoch   1	Train Loss: 13.319801	Train Acc: 10.0000%	Val Loss: 13.132221	Val Acc: 6.8000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 15.799801	Train Acc: 9.2857%	Val Loss: 13.924333	Val Acc: 7.0000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 15.925514	Train Acc: 7.8571%	Val Loss: 15.036303	Val Acc: 9.6000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 18.923717	Train Acc: 7.1429%	Val Loss: 16.116011	Val Acc: 11.8000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 19.468706	Train Acc: 5.0000%	Val Loss: 17.008490	Val Acc: 13.2000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 21.814241	Train Acc: 4.2857%	Val Loss: 17.985186	Val Acc: 15.0000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 20.494919	Train Acc: 5.7143%	Val Loss: 18.983805	Val Acc: 14.8000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 27.182863	Train Acc: 8.5714%	Val Loss: 20.116503	Val Acc: 16.6000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 27.248310	Train Acc: 8.5714%	Val Loss: 22.032333	Val Acc: 14.8000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 28.839548	Train Acc: 12.1429%	Val Loss: 25.521946	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 31.351496	Train Acc: 12.8571%	Val Loss: 30.753056	Val Acc: 12.2000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 32.994808	Train Acc: 14.2857%	Val Loss: 36.492493	Val Acc: 12.2000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 49.721786	Train Acc: 14.2857%	Val Loss: 42.526810	Val Acc: 11.6000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 54.435287	Train Acc: 14.2857%	Val Loss: 48.551044	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 55.899689	Train Acc: 14.2857%	Val Loss: 54.252762	Val Acc: 11.6000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 59.852539	Train Acc: 14.2857%	Val Loss: 59.727749	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 72.923004	Train Acc: 14.2857%	Val Loss: 65.039841	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 65.585579	Train Acc: 14.2857%	Val Loss: 70.252174	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 91.514122	Train Acc: 14.2857%	Val Loss: 75.379662	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 97.295891	Train Acc: 14.2857%	Val Loss: 80.630722	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 74.843208	Train Acc: 14.2857%	Val Loss: 85.822235	Val Acc: 11.4000%</span></span>
<span class="line"><span>Early Stopping at Epoch 21</span></span>
<span class="line"><span>2025-04-25 01:56:51.626476: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:56:51.939807: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:56:52.091749: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:56:52.133575: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1745546212.141010 1595206 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1745546212.141057 1595206 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1745546212.141067 1595206 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1745546212.141074 1595206 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.141082 1595206 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1745546212.141089 1595206 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1745546212.141096 1595206 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1745546212.141103 1595206 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1745546212.141110 1595206 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1745546212.141117 1595206 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-04-25 01:56:52.141128: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.144835 1595206 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1745546212.144861 1595206 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1745546212.144868 1595206 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1745546212.144876 1595206 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.144883 1595206 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1745546212.144890 1595206 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1745546212.144897 1595206 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1745546212.144904 1595206 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1745546212.144911 1595206 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1745546212.144917 1595206 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-04-25 01:56:52.144928: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.148674 1595206 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1745546212.148699 1595206 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1745546212.148707 1595206 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1745546212.148714 1595206 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.148722 1595206 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1745546212.148729 1595206 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1745546212.148738 1595206 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1745546212.148745 1595206 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1745546212.148752 1595206 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1745546212.148759 1595206 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-04-25 01:56:52.148770: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.152367 1595206 buffer_comparator.cc:145] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1745546212.152379 1595206 buffer_comparator.cc:145] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1745546212.152382 1595206 buffer_comparator.cc:145] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1745546212.152385 1595206 buffer_comparator.cc:145] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.152389 1595206 buffer_comparator.cc:145] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1745546212.152392 1595206 buffer_comparator.cc:145] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1745546212.152395 1595206 buffer_comparator.cc:145] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1745546212.152398 1595206 buffer_comparator.cc:145] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1745546212.152401 1595206 buffer_comparator.cc:145] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1745546212.152404 1595206 buffer_comparator.cc:145] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-25 01:56:52.152409: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.155781 1595206 buffer_comparator.cc:145] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1745546212.155792 1595206 buffer_comparator.cc:145] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1745546212.155796 1595206 buffer_comparator.cc:145] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1745546212.155799 1595206 buffer_comparator.cc:145] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.155802 1595206 buffer_comparator.cc:145] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1745546212.155805 1595206 buffer_comparator.cc:145] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1745546212.155808 1595206 buffer_comparator.cc:145] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1745546212.155811 1595206 buffer_comparator.cc:145] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1745546212.155814 1595206 buffer_comparator.cc:145] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1745546212.155817 1595206 buffer_comparator.cc:145] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-25 01:56:52.155822: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.159130 1595206 buffer_comparator.cc:145] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1745546212.159142 1595206 buffer_comparator.cc:145] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1745546212.159145 1595206 buffer_comparator.cc:145] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1745546212.159149 1595206 buffer_comparator.cc:145] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1745546212.159152 1595206 buffer_comparator.cc:145] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1745546212.159155 1595206 buffer_comparator.cc:145] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.159158 1595206 buffer_comparator.cc:145] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1745546212.159161 1595206 buffer_comparator.cc:145] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1745546212.159165 1595206 buffer_comparator.cc:145] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1745546212.159168 1595206 buffer_comparator.cc:145] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-04-25 01:56:52.159173: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.162763 1595206 buffer_comparator.cc:145] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1745546212.162774 1595206 buffer_comparator.cc:145] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1745546212.162778 1595206 buffer_comparator.cc:145] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1745546212.162781 1595206 buffer_comparator.cc:145] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1745546212.162784 1595206 buffer_comparator.cc:145] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.162787 1595206 buffer_comparator.cc:145] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1745546212.162790 1595206 buffer_comparator.cc:145] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1745546212.162793 1595206 buffer_comparator.cc:145] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1745546212.162796 1595206 buffer_comparator.cc:145] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1745546212.162799 1595206 buffer_comparator.cc:145] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-04-25 01:56:52.162804: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.166078 1595206 buffer_comparator.cc:145] Difference at 0: 1084.56, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1745546212.166090 1595206 buffer_comparator.cc:145] Difference at 1: 1350.61, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1745546212.166094 1595206 buffer_comparator.cc:145] Difference at 2: 2009.8, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1745546212.166097 1595206 buffer_comparator.cc:145] Difference at 3: 1768.04, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1745546212.166100 1595206 buffer_comparator.cc:145] Difference at 4: 1240.61, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1745546212.166103 1595206 buffer_comparator.cc:145] Difference at 6: 1407.03, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1745546212.166106 1595206 buffer_comparator.cc:145] Difference at 7: 1138.83, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1745546212.166109 1595206 buffer_comparator.cc:145] Difference at 8: 1417.44, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.166112 1595206 buffer_comparator.cc:145] Difference at 9: 2084.44, expected 1833.77</span></span>
<span class="line"><span>E0000 00:00:1745546212.166115 1595206 buffer_comparator.cc:145] Difference at 10: 1844.73, expected 1592.38</span></span>
<span class="line"><span>2025-04-25 01:56:52.166120: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.169564 1595206 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1745546212.169575 1595206 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1745546212.169579 1595206 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.169582 1595206 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1745546212.169585 1595206 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1745546212.169588 1595206 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1745546212.169591 1595206 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1745546212.169594 1595206 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.169597 1595206 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1745546212.169600 1595206 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-25 01:56:52.169607: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.172964 1595206 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1745546212.172976 1595206 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1745546212.172979 1595206 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.172982 1595206 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1745546212.172985 1595206 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1745546212.172988 1595206 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1745546212.172992 1595206 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1745546212.172995 1595206 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.172998 1595206 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1745546212.173001 1595206 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-25 01:56:52.173005: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.176359 1595206 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1745546212.176370 1595206 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1745546212.176373 1595206 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.176377 1595206 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1745546212.176380 1595206 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1745546212.176383 1595206 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1745546212.176386 1595206 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1745546212.176389 1595206 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.176392 1595206 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1745546212.176395 1595206 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-25 01:56:52.176400: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.179672 1595206 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1745546212.179683 1595206 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1745546212.179687 1595206 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.179690 1595206 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1745546212.179693 1595206 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1745546212.179696 1595206 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1745546212.179699 1595206 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1745546212.179702 1595206 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.179705 1595206 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1745546212.179708 1595206 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-25 01:56:52.179713: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.183028 1595206 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1745546212.183040 1595206 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1745546212.183043 1595206 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.183046 1595206 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1745546212.183049 1595206 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1745546212.183052 1595206 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1745546212.183055 1595206 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1745546212.183058 1595206 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.183061 1595206 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1745546212.183065 1595206 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-25 01:56:52.183069: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.186387 1595206 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1745546212.186399 1595206 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1745546212.186402 1595206 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.186405 1595206 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1745546212.186409 1595206 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1745546212.186412 1595206 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1745546212.186415 1595206 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1745546212.186418 1595206 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.186421 1595206 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1745546212.186424 1595206 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-25 01:56:52.186428: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.189741 1595206 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1745546212.189752 1595206 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1745546212.189755 1595206 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.189758 1595206 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1745546212.189762 1595206 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1745546212.189765 1595206 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1745546212.189768 1595206 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1745546212.189771 1595206 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.189774 1595206 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1745546212.189777 1595206 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-25 01:56:52.189781: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.193072 1595206 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1745546212.193086 1595206 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1745546212.193089 1595206 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.193092 1595206 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1745546212.193095 1595206 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1745546212.193098 1595206 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1745546212.193101 1595206 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1745546212.193105 1595206 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1745546212.193108 1595206 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1745546212.193111 1595206 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-25 01:56:52.193115: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.196416 1595206 buffer_comparator.cc:145] Difference at 0: 1100.47, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1745546212.196428 1595206 buffer_comparator.cc:145] Difference at 1: 1361.33, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1745546212.196431 1595206 buffer_comparator.cc:145] Difference at 2: 2059.82, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1745546212.196434 1595206 buffer_comparator.cc:145] Difference at 3: 1808.05, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1745546212.196438 1595206 buffer_comparator.cc:145] Difference at 4: 1265.06, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1745546212.196441 1595206 buffer_comparator.cc:145] Difference at 5: 1986, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.196444 1595206 buffer_comparator.cc:145] Difference at 6: 1409.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1745546212.196447 1595206 buffer_comparator.cc:145] Difference at 7: 1173.38, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1745546212.196450 1595206 buffer_comparator.cc:145] Difference at 8: 1420.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.196453 1595206 buffer_comparator.cc:145] Difference at 9: 2114.57, expected 1833.77</span></span>
<span class="line"><span>2025-04-25 01:56:52.196458: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.199767 1595206 buffer_comparator.cc:145] Difference at 0: 1100.47, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1745546212.199778 1595206 buffer_comparator.cc:145] Difference at 1: 1361.33, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1745546212.199782 1595206 buffer_comparator.cc:145] Difference at 2: 2059.82, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1745546212.199785 1595206 buffer_comparator.cc:145] Difference at 3: 1808.05, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1745546212.199788 1595206 buffer_comparator.cc:145] Difference at 4: 1265.06, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1745546212.199791 1595206 buffer_comparator.cc:145] Difference at 5: 1986, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.199794 1595206 buffer_comparator.cc:145] Difference at 6: 1409.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1745546212.199797 1595206 buffer_comparator.cc:145] Difference at 7: 1173.38, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1745546212.199800 1595206 buffer_comparator.cc:145] Difference at 8: 1420.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.199803 1595206 buffer_comparator.cc:145] Difference at 9: 2114.57, expected 1833.77</span></span>
<span class="line"><span>2025-04-25 01:56:52.199808: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.203047 1595206 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1745546212.203058 1595206 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1745546212.203062 1595206 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1745546212.203066 1595206 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1745546212.203070 1595206 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1745546212.203073 1595206 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.203076 1595206 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1745546212.203079 1595206 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1745546212.203082 1595206 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.203085 1595206 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-04-25 01:56:52.203090: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.206403 1595206 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1745546212.206415 1595206 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1745546212.206418 1595206 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1745546212.206421 1595206 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1745546212.206424 1595206 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1745546212.206428 1595206 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.206431 1595206 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1745546212.206434 1595206 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1745546212.206437 1595206 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.206440 1595206 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-04-25 01:56:52.206444: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.209784 1595206 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1745546212.209795 1595206 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1745546212.209799 1595206 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1745546212.209802 1595206 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1745546212.209805 1595206 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1745546212.209808 1595206 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.209811 1595206 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1745546212.209814 1595206 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1745546212.209817 1595206 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.209820 1595206 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-04-25 01:56:52.209825: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.213375 1595206 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1745546212.213386 1595206 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1745546212.213390 1595206 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1745546212.213393 1595206 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1745546212.213396 1595206 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1745546212.213399 1595206 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.213404 1595206 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1745546212.213407 1595206 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1745546212.213410 1595206 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.213413 1595206 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-04-25 01:56:52.213418: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.216781 1595206 buffer_comparator.cc:145] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1745546212.216792 1595206 buffer_comparator.cc:145] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.216796 1595206 buffer_comparator.cc:145] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1745546212.216799 1595206 buffer_comparator.cc:145] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1745546212.216802 1595206 buffer_comparator.cc:145] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1745546212.216805 1595206 buffer_comparator.cc:145] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1745546212.216808 1595206 buffer_comparator.cc:145] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1745546212.216811 1595206 buffer_comparator.cc:145] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1745546212.216814 1595206 buffer_comparator.cc:145] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1745546212.216817 1595206 buffer_comparator.cc:145] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-25 01:56:52.216822: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.220130 1595206 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1745546212.220142 1595206 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1745546212.220146 1595206 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1745546212.220150 1595206 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1745546212.220153 1595206 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1745546212.220157 1595206 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1745546212.220161 1595206 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1745546212.220164 1595206 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.77</span></span>
<span class="line"><span>E0000 00:00:1745546212.220167 1595206 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.09</span></span>
<span class="line"><span>E0000 00:00:1745546212.220171 1595206 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.62</span></span>
<span class="line"><span>2025-04-25 01:56:52.220176: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.223459 1595206 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1745546212.223471 1595206 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1745546212.223475 1595206 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1745546212.223479 1595206 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1745546212.223482 1595206 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1745546212.223486 1595206 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1745546212.223490 1595206 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1745546212.223493 1595206 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.77</span></span>
<span class="line"><span>E0000 00:00:1745546212.223498 1595206 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.09</span></span>
<span class="line"><span>E0000 00:00:1745546212.223502 1595206 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.62</span></span>
<span class="line"><span>2025-04-25 01:56:52.223506: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.226841 1595206 buffer_comparator.cc:145] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1745546212.226853 1595206 buffer_comparator.cc:145] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.226856 1595206 buffer_comparator.cc:145] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1745546212.226860 1595206 buffer_comparator.cc:145] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1745546212.226863 1595206 buffer_comparator.cc:145] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1745546212.226866 1595206 buffer_comparator.cc:145] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1745546212.226869 1595206 buffer_comparator.cc:145] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1745546212.226872 1595206 buffer_comparator.cc:145] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1745546212.226875 1595206 buffer_comparator.cc:145] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1745546212.226878 1595206 buffer_comparator.cc:145] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-25 01:56:52.226883: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546212.230146 1595206 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1745546212.230157 1595206 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1745546212.230161 1595206 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1745546212.230164 1595206 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1745546212.230167 1595206 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1745546212.230170 1595206 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1745546212.230173 1595206 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1745546212.230176 1595206 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1745546212.230179 1595206 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1745546212.230182 1595206 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.77</span></span>
<span class="line"><span>2025-04-25 01:56:52.230187: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-04-25 01:56:54.410441: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:56:54.618878: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:56:54.739663: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-25 01:56:54.835468: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1745546214.843190 1595206 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1745546214.843239 1595206 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1745546214.843250 1595206 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1745546214.843257 1595206 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1745546214.843264 1595206 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1745546214.843271 1595206 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1745546214.843277 1595206 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1745546214.843284 1595206 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1745546214.843290 1595206 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1745546214.843297 1595206 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-25 01:56:54.843311: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.846999 1595206 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1745546214.847014 1595206 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1745546214.847017 1595206 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1745546214.847020 1595206 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1745546214.847023 1595206 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1745546214.847026 1595206 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1745546214.847029 1595206 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1745546214.847032 1595206 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1745546214.847035 1595206 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1745546214.847037 1595206 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-25 01:56:54.847042: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.850432 1595206 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1745546214.850443 1595206 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1745546214.850447 1595206 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1745546214.850450 1595206 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1745546214.850453 1595206 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1745546214.850456 1595206 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1745546214.850459 1595206 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1745546214.850461 1595206 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1745546214.850464 1595206 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1745546214.850467 1595206 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-25 01:56:54.850472: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.853752 1595206 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1745546214.853764 1595206 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1745546214.853767 1595206 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1745546214.853770 1595206 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1745546214.853773 1595206 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1745546214.853776 1595206 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1745546214.853780 1595206 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1745546214.853783 1595206 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1745546214.853786 1595206 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1745546214.853789 1595206 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-25 01:56:54.853794: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.857180 1595206 buffer_comparator.cc:145] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1745546214.857191 1595206 buffer_comparator.cc:145] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1745546214.857207 1595206 buffer_comparator.cc:145] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1745546214.857210 1595206 buffer_comparator.cc:145] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1745546214.857214 1595206 buffer_comparator.cc:145] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1745546214.857217 1595206 buffer_comparator.cc:145] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1745546214.857221 1595206 buffer_comparator.cc:145] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1745546214.857224 1595206 buffer_comparator.cc:145] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1745546214.857227 1595206 buffer_comparator.cc:145] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1745546214.857230 1595206 buffer_comparator.cc:145] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-25 01:56:54.857235: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.860568 1595206 buffer_comparator.cc:145] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1745546214.860580 1595206 buffer_comparator.cc:145] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1745546214.860584 1595206 buffer_comparator.cc:145] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1745546214.860588 1595206 buffer_comparator.cc:145] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1745546214.860591 1595206 buffer_comparator.cc:145] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1745546214.860594 1595206 buffer_comparator.cc:145] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1745546214.860597 1595206 buffer_comparator.cc:145] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1745546214.860600 1595206 buffer_comparator.cc:145] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1745546214.860604 1595206 buffer_comparator.cc:145] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1745546214.860607 1595206 buffer_comparator.cc:145] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-04-25 01:56:54.860612: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.864021 1595206 buffer_comparator.cc:145] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1745546214.864032 1595206 buffer_comparator.cc:145] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1745546214.864037 1595206 buffer_comparator.cc:145] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1745546214.864040 1595206 buffer_comparator.cc:145] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1745546214.864043 1595206 buffer_comparator.cc:145] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1745546214.864046 1595206 buffer_comparator.cc:145] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1745546214.864049 1595206 buffer_comparator.cc:145] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1745546214.864052 1595206 buffer_comparator.cc:145] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1745546214.864056 1595206 buffer_comparator.cc:145] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1745546214.864059 1595206 buffer_comparator.cc:145] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-04-25 01:56:54.864065: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.867385 1595206 buffer_comparator.cc:145] Difference at 0: 903.336, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1745546214.867396 1595206 buffer_comparator.cc:145] Difference at 1: 1271.45, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1745546214.867400 1595206 buffer_comparator.cc:145] Difference at 2: 1218.72, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1745546214.867403 1595206 buffer_comparator.cc:145] Difference at 3: 1830.29, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1745546214.867406 1595206 buffer_comparator.cc:145] Difference at 4: 1832.52, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1745546214.867409 1595206 buffer_comparator.cc:145] Difference at 5: 1505.57, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1745546214.867412 1595206 buffer_comparator.cc:145] Difference at 6: 1003.78, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1745546214.867415 1595206 buffer_comparator.cc:145] Difference at 7: 895.724, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1745546214.867418 1595206 buffer_comparator.cc:145] Difference at 8: 1254.14, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1745546214.867421 1595206 buffer_comparator.cc:145] Difference at 9: 1207.96, expected 1052.46</span></span>
<span class="line"><span>2025-04-25 01:56:54.867426: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.870849 1595206 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1745546214.870860 1595206 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1745546214.870864 1595206 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1745546214.870867 1595206 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1745546214.870870 1595206 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1745546214.870873 1595206 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1745546214.870875 1595206 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1745546214.870878 1595206 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1745546214.870881 1595206 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1745546214.870884 1595206 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-25 01:56:54.870889: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.874245 1595206 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1745546214.874257 1595206 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1745546214.874260 1595206 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1745546214.874263 1595206 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1745546214.874266 1595206 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1745546214.874269 1595206 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1745546214.874272 1595206 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1745546214.874275 1595206 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1745546214.874278 1595206 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1745546214.874280 1595206 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-25 01:56:54.874285: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.877634 1595206 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1745546214.877645 1595206 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1745546214.877652 1595206 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1745546214.877656 1595206 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1745546214.877659 1595206 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1745546214.877662 1595206 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1745546214.877666 1595206 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1745546214.877669 1595206 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1745546214.877672 1595206 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1745546214.877675 1595206 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-25 01:56:54.877680: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.880949 1595206 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1745546214.880961 1595206 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1745546214.880965 1595206 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1745546214.880968 1595206 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1745546214.880972 1595206 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1745546214.880975 1595206 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1745546214.880978 1595206 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1745546214.880981 1595206 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1745546214.880984 1595206 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1745546214.880987 1595206 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-25 01:56:54.880992: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.884323 1595206 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1745546214.884334 1595206 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1745546214.884339 1595206 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1745546214.884342 1595206 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1745546214.884345 1595206 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1745546214.884348 1595206 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1745546214.884352 1595206 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1745546214.884355 1595206 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1745546214.884358 1595206 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1745546214.884361 1595206 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-25 01:56:54.884366: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.887690 1595206 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1745546214.887702 1595206 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1745546214.887706 1595206 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1745546214.887709 1595206 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1745546214.887713 1595206 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1745546214.887716 1595206 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1745546214.887721 1595206 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1745546214.887724 1595206 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1745546214.887727 1595206 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1745546214.887730 1595206 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-25 01:56:54.887735: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.891055 1595206 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1745546214.891066 1595206 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1745546214.891070 1595206 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1745546214.891073 1595206 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1745546214.891076 1595206 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1745546214.891079 1595206 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1745546214.891082 1595206 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1745546214.891084 1595206 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1745546214.891087 1595206 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1745546214.891090 1595206 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-25 01:56:54.891095: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.894385 1595206 buffer_comparator.cc:145] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1745546214.894396 1595206 buffer_comparator.cc:145] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1745546214.894401 1595206 buffer_comparator.cc:145] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1745546214.894404 1595206 buffer_comparator.cc:145] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1745546214.894408 1595206 buffer_comparator.cc:145] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1745546214.894411 1595206 buffer_comparator.cc:145] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1745546214.894414 1595206 buffer_comparator.cc:145] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1745546214.894417 1595206 buffer_comparator.cc:145] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1745546214.894420 1595206 buffer_comparator.cc:145] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1745546214.894423 1595206 buffer_comparator.cc:145] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-25 01:56:54.894428: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.897741 1595206 buffer_comparator.cc:145] Difference at 0: 876.475, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1745546214.897752 1595206 buffer_comparator.cc:145] Difference at 1: 1292.4, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1745546214.897756 1595206 buffer_comparator.cc:145] Difference at 2: 1239.8, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1745546214.897759 1595206 buffer_comparator.cc:145] Difference at 3: 1830.71, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1745546214.897762 1595206 buffer_comparator.cc:145] Difference at 4: 1857.47, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1745546214.897765 1595206 buffer_comparator.cc:145] Difference at 5: 1551.94, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1745546214.897768 1595206 buffer_comparator.cc:145] Difference at 6: 1022.45, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1745546214.897771 1595206 buffer_comparator.cc:145] Difference at 8: 1214.29, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1745546214.897774 1595206 buffer_comparator.cc:145] Difference at 9: 1173.34, expected 1052.46</span></span>
<span class="line"><span>E0000 00:00:1745546214.897779 1595206 buffer_comparator.cc:145] Difference at 10: 1732.94, expected 1556.04</span></span>
<span class="line"><span>2025-04-25 01:56:54.897784: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.901101 1595206 buffer_comparator.cc:145] Difference at 0: 876.475, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1745546214.901112 1595206 buffer_comparator.cc:145] Difference at 1: 1292.4, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1745546214.901116 1595206 buffer_comparator.cc:145] Difference at 2: 1239.8, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1745546214.901119 1595206 buffer_comparator.cc:145] Difference at 3: 1830.71, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1745546214.901122 1595206 buffer_comparator.cc:145] Difference at 4: 1857.47, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1745546214.901125 1595206 buffer_comparator.cc:145] Difference at 5: 1551.94, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1745546214.901128 1595206 buffer_comparator.cc:145] Difference at 6: 1022.45, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1745546214.901131 1595206 buffer_comparator.cc:145] Difference at 8: 1214.29, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1745546214.901134 1595206 buffer_comparator.cc:145] Difference at 9: 1173.34, expected 1052.46</span></span>
<span class="line"><span>E0000 00:00:1745546214.901137 1595206 buffer_comparator.cc:145] Difference at 10: 1732.94, expected 1556.04</span></span>
<span class="line"><span>2025-04-25 01:56:54.901141: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.904398 1595206 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1745546214.904409 1595206 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1745546214.904413 1595206 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1745546214.904416 1595206 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1745546214.904419 1595206 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1745546214.904422 1595206 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1745546214.904425 1595206 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1745546214.904428 1595206 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1745546214.904431 1595206 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1745546214.904434 1595206 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-04-25 01:56:54.904439: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.907764 1595206 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1745546214.907775 1595206 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1745546214.907779 1595206 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1745546214.907782 1595206 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1745546214.907785 1595206 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1745546214.907788 1595206 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1745546214.907791 1595206 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1745546214.907794 1595206 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1745546214.907797 1595206 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1745546214.907800 1595206 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-04-25 01:56:54.907805: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.911161 1595206 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1745546214.911172 1595206 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1745546214.911176 1595206 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1745546214.911179 1595206 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1745546214.911182 1595206 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1745546214.911185 1595206 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1745546214.911188 1595206 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1745546214.911191 1595206 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1745546214.911194 1595206 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1745546214.911197 1595206 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-04-25 01:56:54.911202: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.914575 1595206 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1745546214.914587 1595206 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1745546214.914590 1595206 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1745546214.914593 1595206 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1745546214.914596 1595206 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1745546214.914600 1595206 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1745546214.914603 1595206 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1745546214.914606 1595206 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1745546214.914609 1595206 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1745546214.914612 1595206 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-04-25 01:56:54.914617: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.917976 1595206 buffer_comparator.cc:145] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1745546214.917988 1595206 buffer_comparator.cc:145] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1745546214.917991 1595206 buffer_comparator.cc:145] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1745546214.917994 1595206 buffer_comparator.cc:145] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1745546214.917997 1595206 buffer_comparator.cc:145] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1745546214.918000 1595206 buffer_comparator.cc:145] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1745546214.918003 1595206 buffer_comparator.cc:145] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1745546214.918005 1595206 buffer_comparator.cc:145] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1745546214.918008 1595206 buffer_comparator.cc:145] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1745546214.918011 1595206 buffer_comparator.cc:145] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-25 01:56:54.918016: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1172] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1745546214.921339 1595206 buffer_comparator.cc:145] Difference at 540: 1203.79, expected 1064.85</span></span>
<span class="line"><span>Test Loss: 81.764893	Test Acc: 10.3000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
